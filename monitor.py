#!/usr/bin/env python3
"""Training monitor main program — monitors training status across configured nodes"""

import argparse
import time
import sys

from rich.console import Console
from rich.live import Live

from config import NODES, DEFAULT_INTERVAL
from ssh_fetcher import fetch_all_nodes
from log_parser import parse_output
from display import build_table


def collect_and_display(console: Console, show_gpu: bool = False) -> tuple:
    """Fetch and parse data from all nodes"""
    all_data = fetch_all_nodes(NODES, fetch_gpu=show_gpu)

    train_infos = {}
    for node_data in all_data:
        node = node_data["node"]
        infos = []
        for session in node_data.get("sessions", []):
            info = parse_output(session["name"], session["output"])
            infos.append(info)
        train_infos[node] = infos

    return all_data, train_infos


def run_once(console: Console, show_gpu: bool = False):
    """Single execution mode"""
    console.print("[bold]Fetching training status from all nodes...[/bold]\n")
    all_data, train_infos = collect_and_display(console, show_gpu)
    table = build_table(all_data, train_infos, show_gpu=show_gpu)
    console.print(table)


def run_live(console: Console, interval: int, show_gpu: bool = False):
    """Live refresh mode"""
    with Live(console=console, refresh_per_second=1, screen=True) as live:
        while True:
            all_data, train_infos = collect_and_display(console, show_gpu)
            table = build_table(all_data, train_infos, show_gpu=show_gpu)
            live.update(table)
            time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Training Monitor for configured nodes")
    parser.add_argument(
        "--interval", "-i", type=int, default=DEFAULT_INTERVAL,
        help=f"Refresh interval in seconds (default: {DEFAULT_INTERVAL})"
    )
    parser.add_argument(
        "--once", "-1", action="store_true",
        help="Run once and exit (no live refresh)"
    )
    parser.add_argument(
        "--gpu", "-g", action="store_true",
        help="Show GPU utilization and memory usage"
    )
    args = parser.parse_args()

    console = Console()

    if args.once:
        run_once(console, show_gpu=args.gpu)
    else:
        try:
            run_live(console, args.interval, show_gpu=args.gpu)
        except KeyboardInterrupt:
            console.print("\n[bold yellow]Monitor stopped.[/bold yellow]")
            sys.exit(0)


if __name__ == "__main__":
    main()
