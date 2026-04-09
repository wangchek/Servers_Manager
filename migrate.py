#!/usr/bin/env python3
"""Migration tool — moves checkpoints from /root/checkpoints to /workplace/checkpoints"""

import argparse
import json
import os
from datetime import datetime

from rich.console import Console
from rich.table import Table

from ssh_fetcher import _run_ssh, list_tmux_sessions
from config import NODES

console = Console()

SRC_CKPT = "/root/checkpoints"
DST_CKPT = "/workplace/checkpoints"
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".migrate_state.json")


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    console.print(f"[dim]{ts}[/dim] {msg}")


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def is_training_running(node: str) -> bool:
    """Check if a training process is running on the node."""
    result = _run_ssh(node, "pgrep -f train_LDA.py", timeout=10)
    return result is not None and result.strip() != ""


def get_active_run_id(node: str) -> str:
    """Get the currently training run_id (tmux session name, excluding deploy_/auto_ prefixes)."""
    sessions = list_tmux_sessions(node)
    for sess in sessions:
        if sess.startswith("deploy_") or sess.startswith("auto_"):
            continue
        return sess
    return ""


def list_runs(node: str) -> list:
    """List all checkpoint run directories on a node."""
    result = _run_ssh(node, f"ls -d {SRC_CKPT}/*/ 2>/dev/null | xargs -n1 basename", timeout=10)
    if not result:
        return []
    return [r.strip() for r in result.strip().split("\n") if r.strip()]


def get_ckpt_status(node: str) -> str:
    """Check the current checkpoint status on a node."""
    # Check if already a symlink
    result = _run_ssh(node, f"readlink {SRC_CKPT} 2>/dev/null", timeout=10)
    if result and result.strip() == DST_CKPT:
        return "symlinked"

    # Check if /workplace/checkpoints exists
    result = _run_ssh(node, f"test -d {DST_CKPT} && echo YES", timeout=10)
    has_dst = result is not None and "YES" in result

    # Check if /root/checkpoints is a real directory (not a symlink)
    result = _run_ssh(node, f"test -d {SRC_CKPT} -a ! -L {SRC_CKPT} && echo YES", timeout=10)
    has_src = result is not None and "YES" in result

    if has_src and has_dst:
        return "copied"  # Both exist, pending cleanup
    elif has_src:
        return "not_migrated"
    elif has_dst:
        return "dst_only"  # Destination exists but source removed (abnormal state)
    else:
        return "empty"


def get_ckpt_size(node: str, path: str) -> str:
    """Get directory size."""
    result = _run_ssh(node, f"du -sh {path} 2>/dev/null | cut -f1", timeout=30)
    return result.strip() if result else "-"


def _rsync_run(node: str, run_id: str, dry_run: bool = False) -> bool:
    """Rsync a single run directory to /workplace, then remove the source."""
    src = f"{SRC_CKPT}/{run_id}"
    dst = f"{DST_CKPT}/{run_id}"
    size = get_ckpt_size(node, src)

    if dry_run:
        log(f"    [yellow]DRY RUN[/yellow] rsync {run_id}/ ({size}) -> {DST_CKPT}/")
        return True

    log(f"    Copying {run_id}/ ({size})...")
    result = _run_ssh(
        node,
        f"mkdir -p {dst} && rsync -a {src}/ {dst}/ && rm -rf {src} && echo OK",
        timeout=7200,
        allow_fail=True,
    )
    if result and "OK" in result:
        log(f"    [green]{run_id} migrated and source removed[/green]")
        return True
    else:
        log(f"    [red]{run_id} rsync/cleanup failed: {(result or '')[-200:]}[/red]")
        return False


def migrate_node(node: str, dry_run: bool = False) -> bool:
    """Migrate checkpoints for a single node."""
    log(f"[bold cyan]{node}[/bold cyan]")

    status = get_ckpt_status(node)
    training = is_training_running(node)

    if status == "symlinked":
        log(f"  [green]Already migrated (symlink exists)[/green]")
        return True

    if status == "empty":
        log(f"  [dim]No checkpoints to migrate[/dim]")
        return True

    runs = list_runs(node)
    if not runs:
        log(f"  [dim]No run directories found[/dim]")
        return True

    active_run = get_active_run_id(node) if training else ""
    src_size = get_ckpt_size(node, SRC_CKPT)
    log(f"  Source: {SRC_CKPT} ({src_size})")
    log(f"  Runs: {', '.join(runs)}")
    log(f"  Training: {'[yellow]' + active_run + '[/yellow]' if active_run else '[dim]idle[/dim]'}")

    _run_ssh(node, f"mkdir -p {DST_CKPT}")

    # Migrate old runs (not currently training)
    old_runs = [r for r in runs if r != active_run]
    if old_runs:
        log(f"  Migrating {len(old_runs)} old runs...")
        for run_id in old_runs:
            _rsync_run(node, run_id, dry_run=dry_run)

    if active_run:
        log(f"  [yellow]Keeping active run '{active_run}' in place[/yellow]")
        state = load_state()
        state[node] = {
            "status": "partial",
            "active_run": active_run,
            "migrated_at": datetime.now().isoformat(),
        }
        save_state(state)
    else:
        # No training running, create symlink after migrating all runs
        remaining = list_runs(node)
        if not remaining:
            return _do_cleanup(node, dry_run)
        else:
            log(f"  [yellow]Remaining runs: {', '.join(remaining)}[/yellow]")

    return True


def _do_cleanup(node: str, dry_run: bool = False) -> bool:
    """Cleanup: remove source directory and create symlink."""
    if dry_run:
        log(f"  [yellow]DRY RUN[/yellow] would rm -rf {SRC_CKPT} && ln -s {DST_CKPT} {SRC_CKPT}")
        return True

    log(f"  Removing source and creating symlink...")
    result = _run_ssh(
        node,
        f"rm -rf {SRC_CKPT} && ln -s {DST_CKPT} {SRC_CKPT} && echo OK",
        timeout=600,
        allow_fail=True,
    )
    if result and "OK" in result:
        log(f"  [green]Symlink created: {SRC_CKPT} -> {DST_CKPT}[/green]")
        # Update state
        state = load_state()
        state[node] = {"status": "done", "completed_at": datetime.now().isoformat()}
        save_state(state)
        return True
    else:
        log(f"  [red]Cleanup failed: {result}[/red]")
        return False


def cleanup_nodes(nodes, dry_run=False):
    """Check nodes pending cleanup; migrate remaining runs and create symlink after training ends."""
    for node in nodes:
        log(f"[bold cyan]{node}[/bold cyan]")
        status = get_ckpt_status(node)

        if status == "symlinked":
            log(f"  [green]Already done[/green]")
            continue

        training = is_training_running(node)
        if training:
            active = get_active_run_id(node)
            log(f"  [yellow]Training still running ({active}), skipping[/yellow]")
            continue

        # Training finished, migrate remaining runs
        remaining = list_runs(node)
        if remaining:
            log(f"  Training finished, migrating remaining: {', '.join(remaining)}")
            _run_ssh(node, f"mkdir -p {DST_CKPT}")
            for run_id in remaining:
                _rsync_run(node, run_id, dry_run=dry_run)

        # Check if all runs have been migrated
        still_there = list_runs(node)
        if not still_there:
            _do_cleanup(node, dry_run)
        else:
            log(f"  [yellow]Still remaining: {', '.join(still_there)}[/yellow]")


def show_status(nodes):
    """Show migration status for all nodes."""
    table = Table(title="Migration Status")
    table.add_column("Node", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Training", no_wrap=True)
    table.add_column("Src Runs", no_wrap=False)
    table.add_column("Src Size", no_wrap=True)
    table.add_column("Dst Size", no_wrap=True)

    for node in nodes:
        status = get_ckpt_status(node)
        training = is_training_running(node)
        active = get_active_run_id(node) if training else ""

        status_display = {
            "symlinked": "[green]done (symlinked)[/green]",
            "copied": "[yellow]partial, pending cleanup[/yellow]",
            "not_migrated": "[red]not migrated[/red]",
            "empty": "[dim]no checkpoints[/dim]",
            "dst_only": "[yellow]dst only[/yellow]",
        }.get(status, status)

        training_display = f"[yellow]{active}[/yellow]" if active else "[dim]idle[/dim]"

        if status in ("not_migrated", "copied"):
            runs = list_runs(node)
            runs_display = ", ".join(
                f"[yellow]{r}[/yellow]" if r == active else r for r in runs
            ) if runs else "-"
            src_size = get_ckpt_size(node, SRC_CKPT)
        else:
            runs_display = "-"
            src_size = "-"

        dst_size = get_ckpt_size(node, DST_CKPT) if status != "not_migrated" else "-"

        table.add_row(node, status_display, training_display, runs_display, src_size, dst_size)

    console.print(table)


def main():
    parser = argparse.ArgumentParser(
        description="Migrate checkpoints: /root/checkpoints -> /workplace/checkpoints"
    )
    parser.add_argument(
        "--nodes", nargs="*", default=NODES,
        help=f"Nodes to operate on (default: {NODES[0]}-{NODES[-1]})"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without actually executing"
    )
    parser.add_argument(
        "--cleanup", action="store_true",
        help="Check pending nodes and perform cleanup after training ends"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show migration status for all nodes"
    )
    args = parser.parse_args()

    if args.status:
        show_status(args.nodes)
        return

    if args.cleanup:
        cleanup_nodes(args.nodes, dry_run=args.dry_run)
        return

    # Default: perform migration
    console.print("[bold]Checkpoint Migration[/bold]")
    console.print(f"  {SRC_CKPT} -> {DST_CKPT}")
    console.print(f"  Nodes: {', '.join(args.nodes)}")
    if args.dry_run:
        console.print("  [yellow]DRY RUN[/yellow]")

    success = 0
    for node in args.nodes:
        ok = migrate_node(node, dry_run=args.dry_run)
        if ok:
            success += 1

    console.print(f"\n[bold]{success}/{len(args.nodes)} nodes processed[/bold]")
    pending = [n for n in args.nodes if get_ckpt_status(n) == "copied"]
    if pending:
        console.print(f"[yellow]Pending cleanup: {', '.join(pending)}[/yellow]")
        console.print("Run: python migrate.py --cleanup")


if __name__ == "__main__":
    main()
