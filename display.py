"""Real-time terminal display with live refresh"""

import re
from datetime import datetime
from rich.table import Table
from rich.text import Text
from log_parser import TrainInfo


def _parse_gpu_cards(gpu_raw):
    """Parse per-GPU info, returning [(idx, util%, used_mb, total_mb), ...]"""
    if not gpu_raw:
        return []
    cards = []
    for line in gpu_raw.strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            cards.append((int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])))
    return cards


def _parse_gpu_procs(proc_raw):
    """Parse GPU process info, returning a deduplicated list of process descriptions"""
    if not proc_raw:
        return []
    # Format: pid,mem lines ... ===SEP=== ... pid|cmdline lines
    parts = proc_raw.split("===SEP===")
    if len(parts) < 2:
        return []
    cmd_section = parts[1].strip()
    procs = []
    seen = set()
    for line in cmd_section.split("\n"):
        line = line.strip()
        if not line or "|" not in line:
            continue
        pid, cmdline = line.split("|", 1)
        cmdline = cmdline.strip()
        if not cmdline:
            continue
        # Extract key info: script name or key arguments
        label = _simplify_cmdline(cmdline)
        if label not in seen:
            seen.add(label)
            procs.append(label)
    return procs


def _simplify_cmdline(cmdline):
    """Extract a short description from a full cmdline"""
    # train_LDA.py type
    m = re.search(r'(train_\w+\.py)', cmdline)
    if m:
        # Look for data_mix argument
        dm = re.search(r'--datasets\.vla_data\.data_mix\s+(\S+)', cmdline)
        if dm:
            return f"{m.group(1)} ({dm.group(1)})"
        # Look for run_id
        ri = re.search(r'--run_id\s+(\S+)', cmdline)
        if ri:
            return f"{m.group(1)} ({ri.group(1)})"
        return m.group(1)
    # server.py type
    m = re.search(r'(server\.py)', cmdline)
    if m:
        ckpt = re.search(r'--ckpt_path\s+\S+/(\S+?)(?:/checkpoints)?(?:\s|$)', cmdline)
        if ckpt:
            return f"server.py ({ckpt.group(1)})"
        return "server.py"
    # Other python scripts
    m = re.search(r'python\S*\s+(\S+\.py)', cmdline)
    if m:
        return m.group(1)
    # Fallback: take the first 60 characters
    return cmdline[:60]


def _gpu_summary(cards):
    """Generate a GPU summary string"""
    if not cards:
        return "-"
    avg_util = sum(c[1] for c in cards) / len(cards)
    total_used = sum(c[2] for c in cards) / 1024
    total_mem = sum(c[3] for c in cards) / 1024
    return f"{avg_util:.0f}% | {total_used:.0f}/{total_mem:.0f}G"


def _gpu_detail(cards):
    """Generate a per-GPU detail string"""
    if not cards:
        return "-"
    lines = []
    for idx, util, used, total in cards:
        lines.append(f"GPU{idx}: {util}% {used // 1024}G/{total // 1024}G")
    return "\n".join(lines)


def build_table(all_data: list, train_infos: dict, show_gpu: bool = False) -> Table:
    """Build a rich Table displaying training status for all nodes"""
    table = Table(
        title=f"Training Monitor  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        show_lines=True,
        expand=False,
    )

    table.add_column("Node", style="bold cyan", no_wrap=True)
    table.add_column("Session", no_wrap=True)
    table.add_column("Step", justify="right", no_wrap=True)
    table.add_column("Progress", justify="right", no_wrap=True)
    table.add_column("Loss", justify="right", no_wrap=True)
    table.add_column("LR", justify="right", no_wrap=True)
    table.add_column("Speed", justify="right", no_wrap=True)
    table.add_column("Epoch", justify="right", no_wrap=True)
    table.add_column("Elapsed", justify="center", no_wrap=True)
    table.add_column("ETA", justify="center", no_wrap=True)
    if show_gpu:
        table.add_column("GPU", no_wrap=True, min_width=18)
        table.add_column("Processes", no_wrap=False, min_width=20)

    for node_data in all_data:
        node = node_data["node"]

        if show_gpu:
            cards = _parse_gpu_cards(node_data.get("gpu"))
            gpu_str = _gpu_summary(cards)
            gpu_detail = _gpu_detail(cards)
            procs = _parse_gpu_procs(node_data.get("gpu_procs"))
            proc_str = "\n".join(procs) if procs else "-"
        else:
            gpu_str = None
            proc_str = None

        if node_data["status"] == "offline":
            row = [node, Text("OFFLINE", style="bold red"),
                   "-", "-", "-", "-", "-", "-", "-", "-"]
            if show_gpu:
                row.extend(["-", "-"])
            table.add_row(*row)
            continue

        infos = train_infos.get(node, [])
        if not infos:
            row = [node, Text("No tmux session", style="yellow"),
                   "-", "-", "-", "-", "-", "-", "-", "-"]
            if show_gpu:
                row.extend([gpu_str, proc_str])
            table.add_row(*row)
            continue

        first_row = True
        for info in infos:
            if not info.has_data:
                row = [node, info.session,
                       Text("No train data", style="yellow"),
                       "-", "-", "-", "-", "-", "-", "-"]
                if show_gpu:
                    row.extend([gpu_str if first_row else "", proc_str if first_row else ""])
                first_row = False
                table.add_row(*row)
                continue

            # Format step
            if info.total_steps > 0:
                step_str = f"{info.step:,}/{info.total_steps:,}"
            else:
                step_str = f"{info.step:,}"

            # Format progress
            if info.progress_pct > 0:
                progress_str = f"{info.progress_pct:.1f}%"
            else:
                progress_str = "-"

            loss_str = f"{info.loss:.6f}" if info.loss > 0 else "-"
            lr_str = f"{info.learning_rate:.2e}" if info.learning_rate > 0 else "-"
            epoch_str = f"{info.epoch:.0f}" if info.epoch > 0 else "-"

            row = [
                node,
                Text(info.session, style="green"),
                step_str,
                progress_str,
                loss_str,
                lr_str,
                info.speed or "-",
                epoch_str,
                info.elapsed or "-",
                info.remaining or "-",
            ]
            if show_gpu:
                row.extend([gpu_str if first_row else "", proc_str if first_row else ""])
            first_row = False
            table.add_row(*row)

    return table
