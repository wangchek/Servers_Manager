"""SSH connection to servers, capturing tmux session output"""

import subprocess
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import SSH_TIMEOUT, CAPTURE_LINES


def _run_ssh(node: str, cmd: str, timeout: int = None, allow_fail: bool = False) -> Optional[str]:
    """Execute a command on a remote server via SSH, returning stdout or None on failure.

    When allow_fail=True, returns stdout+stderr even if the command returns non-zero
    (useful for capturing error messages).
    """
    ssh_timeout = timeout or SSH_TIMEOUT
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout={}".format(SSH_TIMEOUT),
             "-o", "StrictHostKeyChecking=no",
             "-o", "BatchMode=yes",
             node, cmd],
            capture_output=True, text=True, timeout=ssh_timeout + 5
        )
        if result.returncode == 0:
            return result.stdout
        if allow_fail:
            return (result.stdout or "") + (result.stderr or "")
        return None
    except (subprocess.TimeoutExpired, Exception):
        return None


def list_tmux_sessions(node: str) -> list[str]:
    """List all tmux session names on a remote server"""
    output = _run_ssh(node, "tmux list-sessions -F '#{session_name}'")
    if not output:
        return []
    return [line.strip() for line in output.strip().split("\n") if line.strip()]


def capture_tmux_pane(node: str, session: str) -> Optional[str]:
    """Capture the last N lines of output from a given tmux session"""
    cmd = f"tmux capture-pane -t {session} -p -S -{CAPTURE_LINES}"
    return _run_ssh(node, cmd)


def fetch_gpu_info(node: str) -> Optional[str]:
    """Fetch per-GPU utilization and memory usage for a node"""
    cmd = "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null"
    return _run_ssh(node, cmd)


def fetch_gpu_processes(node: str) -> Optional[str]:
    """Fetch information about processes occupying GPUs on a node"""
    cmd = (
        "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null && "
        "echo '===SEP===' && "
        "nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u | "
        "while read pid; do echo \"$pid|$(cat /proc/$pid/cmdline 2>/dev/null | tr '\\0' ' ' | head -c 200)\"; done"
    )
    return _run_ssh(node, cmd, timeout=15)


def fetch_node(node: str, fetch_gpu: bool = False) -> dict:
    """Fetch all tmux session output and GPU info for a single node"""
    result = {"node": node, "status": "offline", "sessions": [], "gpu": None, "gpu_procs": None}
    sessions = list_tmux_sessions(node)
    if not sessions:
        if fetch_gpu:
            gpu_out = fetch_gpu_info(node)
            if gpu_out:
                result["status"] = "online"
                result["gpu"] = gpu_out.strip()
                proc_out = fetch_gpu_processes(node)
                if proc_out:
                    result["gpu_procs"] = proc_out.strip()
        return result

    result["status"] = "online"
    for session in sessions:
        output = capture_tmux_pane(node, session)
        result["sessions"].append({
            "name": session,
            "output": output or "",
        })

    if fetch_gpu:
        gpu_out = fetch_gpu_info(node)
        if gpu_out:
            result["gpu"] = gpu_out.strip()
        proc_out = fetch_gpu_processes(node)
        if proc_out:
            result["gpu_procs"] = proc_out.strip()

    return result


def fetch_all_nodes(nodes: list, fetch_gpu: bool = False) -> list:
    """Fetch data from all nodes concurrently"""
    results = []
    with ThreadPoolExecutor(max_workers=len(nodes)) as executor:
        futures = {executor.submit(fetch_node, node, fetch_gpu): node for node in nodes}
        for future in as_completed(futures):
            results.append(future.result())
    # Sort by node name
    results.sort(key=lambda x: x["node"])
    return results
