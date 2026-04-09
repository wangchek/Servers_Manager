#!/usr/bin/env python3
"""Auto-deploy tool — generate deploy scripts and launch SharPA server for specified nodes"""

import argparse
import re
import sys
import time
from datetime import datetime

from rich.console import Console

from ssh_fetcher import _run_ssh, list_tmux_sessions
from config import (
    CAPTURE_LINES, TRAIN_SCRIPT_ROOT, DEPLOY_ROOT,
    CKPT_ROOT, SERVER_PY, DEPLOY_PORT, CONDA_PREFIX, CONDA_ENV,
)

console = Console()

DEPLOY_TEMPLATE = f"""source {CONDA_PREFIX}/bin/activate
conda activate {CONDA_ENV}

python {{server_py}} --ckpt_path {{ckpt_path}} --port {{port}}
"""


def get_run_id(node, session_name):
    """Extract run_id from the training script"""
    # Search all date subdirectories for the script
    cmd = f"find {TRAIN_SCRIPT_ROOT} -name '{session_name}.sh' -type f | head -1"
    script_path = _run_ssh(node, cmd)
    if not script_path or not script_path.strip():
        return None
    script_path = script_path.strip()
    output = _run_ssh(node, f"grep '^run_id=' {script_path}")
    if not output:
        return None
    m = re.search(r"run_id=(\S+)", output)
    if m:
        return m.group(1).strip()
    return None


def get_latest_ckpt(node, run_id):
    """Find the latest checkpoint file path"""
    ckpt_dir = f"{CKPT_ROOT}/{run_id}/checkpoints"
    output = _run_ssh(node, f"ls {ckpt_dir}")
    if not output:
        return None

    files = [f.strip() for f in output.strip().split("\n") if f.strip()]
    # Filter for files matching steps_N_pytorch_model.pt format
    ckpt_files = []
    for f in files:
        m = re.match(r"steps_(\d+)_pytorch_model\.pt", f)
        if m:
            ckpt_files.append((int(m.group(1)), f))

    if not ckpt_files:
        return None

    ckpt_files.sort(key=lambda x: x[0])
    latest = ckpt_files[-1]
    return f"{ckpt_dir}/{latest[1]}", latest[0]


def deploy_node(node, port, dry_run):
    """Deploy a single node; returns (success, deploy_session_name) or (False, None)"""
    console.print(f"\n[bold cyan]{'='*50}")
    console.print(f"[bold cyan]  {node}")
    console.print(f"[bold cyan]{'='*50}")

    # 1. Get tmux sessions
    sessions = list_tmux_sessions(node)
    if not sessions:
        console.print(f"  [red]No tmux sessions found (offline?)")
        return False, None

    console.print(f"  tmux sessions: {', '.join(sessions)}")

    # 2. Find training session and run_id
    run_id = None
    train_session = None
    for session in sessions:
        rid = get_run_id(node, session)
        if rid:
            run_id = rid
            train_session = session
            break

    if not run_id:
        console.print(f"  [red]Could not find run_id from training scripts")
        return False, None

    console.print(f"  train session: {train_session}")
    console.print(f"  run_id: {run_id}")

    # 3. Find latest checkpoint
    result = get_latest_ckpt(node, run_id)
    if not result:
        console.print(f"  [red]No checkpoints found for run_id={run_id}")
        return False, None

    ckpt_path, step_num = result
    console.print(f"  latest ckpt: step {step_num:,}")
    console.print(f"  ckpt path: {ckpt_path}")

    # 4. Generate deploy script
    script_content = DEPLOY_TEMPLATE.format(
        server_py=SERVER_PY,
        ckpt_path=ckpt_path,
        port=port,
    )

    date_str = datetime.now().strftime("%m%d")
    deploy_dir = f"{DEPLOY_ROOT}/{date_str}"
    script_name = f"{run_id}_{step_num // 1000}k.sh"
    script_path = f"{deploy_dir}/{script_name}"

    console.print(f"  deploy script: {script_path}")
    console.print(f"  [dim]--- script content ---")
    for line in script_content.strip().split("\n"):
        console.print(f"  [dim]{line}")
    console.print(f"  [dim]--- end ---")

    if dry_run:
        console.print(f"  [yellow]DRY RUN — skipping execution")
        return True, None

    # 5. Write deploy script to server
    _run_ssh(node, f"mkdir -p {deploy_dir}")
    # Write file using heredoc
    write_cmd = f"cat > {script_path} << 'DEPLOY_EOF'\n{script_content}DEPLOY_EOF"
    _run_ssh(node, write_cmd)
    _run_ssh(node, f"chmod +x {script_path}")

    # 6. Start tmux session
    deploy_session = f"deploy_{run_id}"
    # Kill existing session with the same name first
    _run_ssh(node, f"tmux kill-session -t {deploy_session} 2>/dev/null")
    _run_ssh(node, f"tmux new-session -d -s {deploy_session} 'bash {script_path}'")

    console.print(f"  [green]Started tmux session: {deploy_session}")
    console.print(f"  [green]Server starting on port {port}")
    return True, deploy_session


def fetch_deploy_output(node, deploy_session):
    """Capture output from the deploy tmux session"""
    cmd = f"tmux capture-pane -t {deploy_session} -p -S -{CAPTURE_LINES}"
    output = _run_ssh(node, cmd, timeout=15)
    if not output:
        return None
    # Remove blank lines
    lines = [l for l in output.split("\n") if l.strip()]
    return "\n".join(lines) if lines else None


# Keywords indicating server.py has finished starting
_READY_KEYWORDS = ["server running", "Serving on", "WebSocket"]
# Keywords indicating server error
_ERROR_KEYWORDS = ["Error", "Traceback", "CUDA out of memory", "RuntimeError"]


def wait_for_server(node, deploy_session, timeout, poll_interval=10):
    """Poll until server is ready; returns (output, status)
    status: 'ready' / 'error' / 'timeout'
    """
    elapsed = 0
    while elapsed < timeout:
        output = fetch_deploy_output(node, deploy_session)
        if output:
            # Check if server started successfully
            for kw in _READY_KEYWORDS:
                if kw.lower() in output.lower():
                    return output, "ready"
            # Check for errors
            for kw in _ERROR_KEYWORDS:
                if kw in output:
                    return output, "error"
        time.sleep(poll_interval)
        elapsed += poll_interval
        console.print(f"  [dim]...waiting ({elapsed}s/{timeout}s)")

    # Timed out; return current output
    output = fetch_deploy_output(node, deploy_session)
    return output, "timeout"


def kill_deploy_sessions(nodes):
    """Kill all deploy_ prefixed tmux sessions on specified nodes"""
    for node in nodes:
        console.print(f"\n[bold cyan]{node}[/bold cyan]")
        sessions = list_tmux_sessions(node)
        if not sessions:
            console.print(f"  [red]Offline or no tmux sessions")
            continue

        deploy_sessions = [s for s in sessions if s.startswith("deploy_")]
        if not deploy_sessions:
            console.print(f"  [dim]No deploy sessions found")
            continue

        for session in deploy_sessions:
            _run_ssh(node, f"tmux kill-session -t {session}")
            console.print(f"  [green]Killed: {session}")


def main():
    parser = argparse.ArgumentParser(
        description="Deploy SharPA server with latest checkpoint"
    )
    parser.add_argument(
        "nodes", nargs="+",
        help="Nodes to deploy/kill (e.g. node47 node49)"
    )
    parser.add_argument(
        "--port", type=int, default=DEPLOY_PORT,
        help=f"Server port (default: {DEPLOY_PORT})"
    )
    parser.add_argument(
        "--kill", action="store_true",
        help="Kill all deploy_ tmux sessions on specified nodes"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only show what would be done, don't execute"
    )
    parser.add_argument(
        "--timeout", type=int, default=300,
        help="Max seconds to wait for server ready (default: 300)"
    )
    args = parser.parse_args()

    console.print("[bold]Deploy SharPA Server[/bold]")

    if args.kill:
        kill_deploy_sessions(args.nodes)
        return

    if args.dry_run:
        console.print("[yellow]DRY RUN MODE[/yellow]")

    deployed = []  # (node, deploy_session)
    success = 0
    failed = 0
    for node in args.nodes:
        ok, deploy_session = deploy_node(node, args.port, args.dry_run)
        if ok:
            success += 1
            if deploy_session:
                deployed.append((node, deploy_session))
        else:
            failed += 1

    # After deployment, poll and wait for each server to start
    if deployed:
        console.print(f"\n[bold]Waiting for servers to start (timeout: {args.timeout}s)...[/bold]")

        for node, session in deployed:
            console.print(f"\n[bold cyan]--- {node} / {session} ---[/bold cyan]")
            output, status = wait_for_server(node, session, args.timeout)
            if status == "ready":
                console.print(f"  [bold green]Server is ready![/bold green]")
            elif status == "error":
                console.print(f"  [bold red]Server encountered an error![/bold red]")
            else:
                console.print(f"  [bold yellow]Timeout — server may still be loading[/bold yellow]")
            if output:
                console.print(output)
            else:
                console.print("[yellow]No output[/yellow]")

    console.print(f"\n[bold]Done: {success} deployed, {failed} failed[/bold]")


if __name__ == "__main__":
    main()
