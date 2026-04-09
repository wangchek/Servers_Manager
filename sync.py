#!/usr/bin/env python3
"""Sync LDA code across nodes (especially mixtures.py) via git commit/pull/push"""

import argparse
import sys

from rich.console import Console

from ssh_fetcher import _run_ssh
from config import NODES, LDA_DIR, MIXTURES_REL_PATH, GIT_BRANCH

console = Console()


def run_git(node, git_cmd, timeout=30):
    """Run a git command in the remote node's LDA directory; returns (success, output)"""
    full_cmd = f"cd {LDA_DIR} && git {git_cmd} 2>&1"
    output = _run_ssh(node, full_cmd, timeout=timeout, allow_fail=True)
    if output is None:
        return False, "SSH connection failed or timed out"
    return True, output.strip()


def resolve_conflict(node):
    """Auto-resolve mixtures.py conflicts: keep all new entries from both sides"""
    content = _run_ssh(node, f"cat {LDA_DIR}/{MIXTURES_REL_PATH}", timeout=15)
    if not content or "<<<<<<<" not in content:
        return False

    console.print(f"  [yellow]Resolving conflict in mixtures.py...")

    # Remove conflict markers, keep content from both sides
    lines = content.split("\n")
    resolved = []
    in_conflict = False
    section = None  # 'ours' or 'theirs'

    for line in lines:
        if line.startswith("<<<<<<<"):
            in_conflict = True
            section = "ours"
            continue
        elif line.startswith("=======") and in_conflict:
            section = "theirs"
            continue
        elif line.startswith(">>>>>>>") and in_conflict:
            in_conflict = False
            section = None
            continue
        else:
            resolved.append(line)

    resolved_content = "\n".join(resolved)

    # Write resolved content back to file
    write_cmd = f"cat > {LDA_DIR}/{MIXTURES_REL_PATH} << 'SYNC_EOF'\n{resolved_content}\nSYNC_EOF"
    _run_ssh(node, write_cmd, timeout=15)

    ok, out = run_git(node, f"add {MIXTURES_REL_PATH}")
    if not ok:
        console.print(f"  [red]git add failed: {out}")
        return False
    ok, out = run_git(node, "rebase --continue", timeout=30)
    if not ok:
        console.print(f"  [red]rebase --continue failed: {out}")
        return False

    console.print(f"  [green]Conflict resolved")
    return True


def sync_node(node, dry_run):
    """Sync a single node"""
    console.print(f"\n[bold cyan]{'='*50}")
    console.print(f"[bold cyan]  {node}")
    console.print(f"[bold cyan]{'='*50}")

    # 1. Check for uncommitted mixtures.py changes
    ok, status = run_git(node, f"status --short {MIXTURES_REL_PATH}")
    if not ok:
        console.print(f"  [red]Cannot connect or git error: {status}")
        return False

    has_changes = status != ""
    if has_changes:
        console.print(f"  [yellow]Has uncommitted changes in mixtures.py")
    else:
        console.print(f"  [dim]No local changes in mixtures.py")

    if dry_run:
        console.print(f"  [yellow]DRY RUN — would commit, pull, push")
        return True

    # 2. If there are changes, commit first
    if has_changes:
        ok, out = run_git(node, f"add {MIXTURES_REL_PATH}")
        if not ok:
            console.print(f"  [red]git add failed: {out}")
            return False
        ok, out = run_git(node, f'commit -m "update mixtures.py from {node}"')
        if not ok:
            console.print(f"  [red]git commit failed: {out}")
            return False
        console.print(f"  [green]Committed local changes")

    # 3. pull --rebase
    console.print(f"  Pulling from origin...")
    ok, pull_result = run_git(node, f"pull --rebase origin {GIT_BRANCH}", timeout=30)
    if not ok:
        console.print(f"  [red]Pull failed: {pull_result}")
        if "CONFLICT" in pull_result or "conflict" in pull_result:
            resolved = resolve_conflict(node)
            if not resolved:
                console.print(f"  [red]Failed to auto-resolve conflict, manual fix needed")
                run_git(node, "rebase --abort")
                return False
        else:
            return False
    elif "Already up to date" in pull_result:
        console.print(f"  [dim]Already up to date")
    else:
        console.print(f"  [green]Pulled successfully")

    # 4. push
    console.print(f"  Pushing to origin...")
    ok, push_result = run_git(node, f"push origin {GIT_BRANCH}", timeout=30)
    if not ok:
        if "rejected" in push_result:
            console.print(f"  [red]Push rejected (another node pushed first?)")
            console.print(f"  [yellow]Retrying pull + push...")
            run_git(node, f"pull --rebase origin {GIT_BRANCH}", timeout=30)
            ok2, push_result2 = run_git(node, f"push origin {GIT_BRANCH}", timeout=30)
            if ok2:
                console.print(f"  [green]Retry push succeeded")
            else:
                console.print(f"  [red]Retry push also failed: {push_result2}")
                return False
        else:
            console.print(f"  [red]Push failed: {push_result}")
            return False
    elif "Everything up-to-date" in push_result:
        console.print(f"  [dim]Nothing to push")
    else:
        console.print(f"  [green]Pushed successfully")

    return True


def main():
    parser = argparse.ArgumentParser(description="Sync LDA code across nodes via git")
    parser.add_argument(
        "nodes", nargs="*", default=NODES,
        help=f"Nodes to sync (default: all {NODES[0]}-{NODES[-1]})"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only show what would be done"
    )
    args = parser.parse_args()

    console.print("[bold]Sync LDA Code[/bold]")
    if args.dry_run:
        console.print("[yellow]DRY RUN MODE[/yellow]")

    console.print(f"Nodes: {', '.join(args.nodes)}")
    console.print(f"[dim]Syncing sequentially to avoid push conflicts\n")

    success = 0
    failed = 0
    for node in args.nodes:
        ok = sync_node(node, args.dry_run)
        if ok:
            success += 1
        else:
            failed += 1

    console.print(f"\n[bold]Done: {success} synced, {failed} failed[/bold]")


if __name__ == "__main__":
    main()
