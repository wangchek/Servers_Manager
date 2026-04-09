#!/usr/bin/env python3
"""Auto Train — reads config.json, deploys watchers on target nodes,
and automatically starts training once data is ready.

Watchers run in tmux on the server side, no persistent local connection needed.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Optional, List

from rich.console import Console
from rich.table import Table

from ssh_fetcher import _run_ssh, list_tmux_sessions, capture_tmux_pane
from log_parser import parse_output
from config import (
    LDA_DIR, TRAIN_SCRIPT_ROOT,
    CONDA_PREFIX, CONDA_ENV, DEFAULT_ROBOT_TYPE,
    DATA_READY_MARKER, GIT_BRANCH,
    BASE_VLM_PATH, PRETRAINED_CKPT, WANDB_ENTITY,
)
from batch_train import TRAIN_SCRIPT_TEMPLATE

console = Console()

WATCHER_SESSION_PREFIX = "auto_"
WATCHER_DIR = "/root/auto_train"  # Remote directory for watcher scripts


def _build_watcher_script(task_name: str, data_path: str,
                          robot_type: str, interval: int) -> str:
    """Generate the server-side watcher bash script."""
    date_str = datetime.now().strftime("%m%d")
    script_dir = f"{TRAIN_SCRIPT_ROOT}/{date_str}"
    script_path = f"{script_dir}/{task_name}.sh"
    mixtures_path = f"{LDA_DIR}/lda/dataloader/gr00t_lerobot/mixtures.py"
    ready_path = f"/root/{data_path}/{DATA_READY_MARKER}"

    # Training script content (escaped for heredoc nesting)
    train_content = TRAIN_SCRIPT_TEMPLATE.format(
        task_name=task_name,
        conda_prefix=CONDA_PREFIX,
        conda_env=CONDA_ENV,
        lda_dir=LDA_DIR,
        base_vlm_path=BASE_VLM_PATH,
        pretrained_ckpt=PRETRAINED_CKPT,
        wandb_entity=WANDB_ENTITY,
    )

    watcher = f'''#!/bin/bash
# Auto-generated watcher for task: {task_name}
# Polls for data readiness, then launches training

TASK_NAME="{task_name}"
DATA_PATH="{data_path}"
ROBOT_TYPE="{robot_type}"
READY_FILE="{ready_path}"
INTERVAL={interval}
SCRIPT_DIR="{script_dir}"
SCRIPT_PATH="{script_path}"
MIXTURES_PATH="{mixtures_path}"
LDA_DIR="{LDA_DIR}"
CONDA_PREFIX="{CONDA_PREFIX}"
CONDA_ENV="{CONDA_ENV}"

log() {{
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}}

log "Watcher started for $TASK_NAME"
log "Watching: $READY_FILE"
log "Poll interval: ${{INTERVAL}}s"

# Phase 1: Wait for data readiness
while true; do
    if [ -f "$READY_FILE" ]; then
        log "Data READY: $READY_FILE found!"
        break
    fi
    log "Waiting for data... ($READY_FILE not found)"
    sleep $INTERVAL
done

# Phase 2: Add mixture entry (if not exists)
if grep -q "\\"$TASK_NAME\\"" "$MIXTURES_PATH" 2>/dev/null; then
    log "Mixture '$TASK_NAME' already exists"
else
    log "Adding mixture entry for '$TASK_NAME'"
    python3 -c "
import base64
entry = base64.b64decode('{_b64_mixture_entry(task_name, data_path, robot_type)}').decode()
path = '$MIXTURES_PATH'
f = open(path, 'r'); content = f.read(); f.close()
idx = content.rfind('}}')
content = content[:idx] + entry + content[idx:]
f = open(path, 'w'); f.write(content); f.close()
print('OK')
"
    # git sync
    log "Syncing mixtures.py via git..."
    cd "$LDA_DIR" && git add "$MIXTURES_PATH" && git commit -m "auto_train: add $TASK_NAME" 2>/dev/null
    cd "$LDA_DIR" && git pull --rebase origin {GIT_BRANCH} 2>/dev/null && git push origin {GIT_BRANCH} 2>/dev/null
fi

# Phase 3: Generate training script
mkdir -p "$SCRIPT_DIR"
cat > "$SCRIPT_PATH" << 'TRAINSCRIPT_EOF'
{train_content}TRAINSCRIPT_EOF
chmod +x "$SCRIPT_PATH"
log "Training script written: $SCRIPT_PATH"

# Phase 4: Kill old training sessions (exclude deploy_ and auto_ prefixes)
for sess in $(tmux list-sessions -F '#{{session_name}}' 2>/dev/null); do
    case "$sess" in
        deploy_*|auto_*) continue ;;
        *) tmux kill-session -t "$sess" 2>/dev/null; log "Killed old session: $sess" ;;
    esac
done

# Phase 5: Launch training
tmux new-session -d -s "$TASK_NAME" "bash $SCRIPT_PATH"
log "Training launched: tmux session '$TASK_NAME'"
log "Watcher complete. Training is running."
'''
    return watcher


def _b64_mixture_entry(task_name: str, data_path: str, robot_type: str) -> str:
    """Generate a base64-encoded mixture entry."""
    import base64
    entry = (
        f'\n    "{task_name}": [\n'
        f'        ("{data_path}", 1.0, "{robot_type}"),\n'
        f'    ],\n'
    )
    return base64.b64encode(entry.encode()).decode()


def deploy_watcher(node: str, task_name: str, data_path: str,
                   robot_type: str, interval: int, dry_run: bool = False) -> bool:
    """Deploy a watcher on the target node."""
    session_name = f"{WATCHER_SESSION_PREFIX}{task_name}"
    watcher_script = _build_watcher_script(task_name, data_path, robot_type, interval)
    script_path = f"{WATCHER_DIR}/{task_name}_watcher.sh"

    if dry_run:
        console.print(f"  [yellow]DRY RUN[/yellow] would deploy watcher to {node}:{script_path}")
        return True

    # Create directory
    _run_ssh(node, f"mkdir -p {WATCHER_DIR}")

    # Write watcher script
    write_cmd = f"cat > {script_path} << 'WATCHER_EOF'\n{watcher_script}WATCHER_EOF"
    result = _run_ssh(node, write_cmd, timeout=15)
    if result is None:
        console.print(f"  [red]Failed to write watcher script[/red]")
        return False
    _run_ssh(node, f"chmod +x {script_path}")

    # Kill existing watcher session with the same name
    sessions = list_tmux_sessions(node)
    if session_name in sessions:
        _run_ssh(node, f"tmux kill-session -t {session_name} 2>/dev/null")

    # Start watcher tmux session
    result = _run_ssh(node, f"tmux new-session -d -s {session_name} 'bash {script_path}'")
    if result is None:
        console.print(f"  [red]Failed to start watcher tmux session[/red]")
        return False

    console.print(f"  [green]Watcher deployed: tmux session '{session_name}'[/green]")
    return True


def check_watchers(tasks):
    """Check watcher status on each node."""
    table = Table(title="Watcher Status")
    table.add_column("Node", no_wrap=True)
    table.add_column("Task", no_wrap=True)
    table.add_column("Watcher", no_wrap=True)
    table.add_column("Status", no_wrap=False)

    for task in tasks:
        node = task["node"]
        task_name = task["task_name"]
        session_name = f"{WATCHER_SESSION_PREFIX}{task_name}"

        sessions = list_tmux_sessions(node)
        if session_name in sessions:
            # Watcher still running, capture output for status
            output = capture_tmux_pane(node, session_name)
            if output:
                lines = [l.strip() for l in output.strip().split("\n") if l.strip()]
                last_lines = lines[-3:] if len(lines) >= 3 else lines
                status = "\n".join(last_lines)
            else:
                status = "running (no output yet)"
            table.add_row(node, task_name, "[green]running[/green]", status)
        elif task_name in sessions:
            # Watcher finished, training session is running
            output = capture_tmux_pane(node, task_name)
            info = parse_output(task_name, output) if output else None
            if info and info.has_data:
                status = f"step {info.step}, loss {info.loss:.4f}" if info.loss else f"step {info.step}"
            else:
                status = "training (initializing...)"
            table.add_row(node, task_name, "[green]done[/green]", f"Training: {status}")
        else:
            table.add_row(node, task_name, "[red]not found[/red]", "-")

    console.print(table)


def kill_watchers(tasks):
    """Kill all watchers."""
    for task in tasks:
        node = task["node"]
        task_name = task["task_name"]
        session_name = f"{WATCHER_SESSION_PREFIX}{task_name}"
        _run_ssh(node, f"tmux kill-session -t {session_name} 2>/dev/null")
        console.print(f"  Killed {node}/{session_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Auto Train — deploy watchers on servers that wait for data readiness and automatically start training"
    )
    parser.add_argument(
        "config", nargs="?",
        help="Training task JSON config file (same format as batch_train.py)"
    )
    parser.add_argument(
        "--interval", type=int, default=300,
        help="Watcher polling interval in seconds (default: 300)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without actually deploying"
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Check status of deployed watchers"
    )
    parser.add_argument(
        "--kill", action="store_true",
        help="Kill all watchers"
    )
    args = parser.parse_args()

    if not args.config:
        parser.error("Config file path required (e.g., configs/0409_train.json)")

    with open(args.config) as f:
        config = json.load(f)
    tasks = config["tasks"]

    if args.check:
        check_watchers(tasks)
        return

    if args.kill:
        kill_watchers(tasks)
        return

    # Deploy watchers
    console.print("[bold]Auto Train — Deploy Watchers[/bold]")
    console.print(f"Config: {args.config} ({len(tasks)} tasks)")
    console.print(f"Poll interval: {args.interval}s")
    if args.dry_run:
        console.print("[yellow]DRY RUN[/yellow]")

    table = Table(title="Tasks")
    table.add_column("Node", no_wrap=True)
    table.add_column("Task", no_wrap=True)
    table.add_column("Data Path", no_wrap=True)
    for t in tasks:
        table.add_row(t["node"], t["task_name"], t["data_path"])
    console.print(table)

    success = 0
    for task in tasks:
        node = task["node"]
        task_name = task["task_name"]
        data_path = task["data_path"]
        robot_type = task.get("robot_type", DEFAULT_ROBOT_TYPE)

        console.print(f"\n[bold cyan]{node} / {task_name}[/bold cyan]")
        ok = deploy_watcher(node, task_name, data_path, robot_type,
                           args.interval, dry_run=args.dry_run)
        if ok:
            success += 1

    console.print(f"\n[bold green]Deployed {success}/{len(tasks)} watchers[/bold green]")
    console.print("Watchers run on servers — safe to close your Mac.")
    console.print(f"Check status: python auto_train.py {args.config} --check")


if __name__ == "__main__":
    main()
