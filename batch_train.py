#!/usr/bin/env python3
"""Batch training tool — launch training tasks on multiple nodes based on a config file"""

import argparse
import base64
import json
import sys
import time
from datetime import datetime

from rich.console import Console
from rich.table import Table

from ssh_fetcher import _run_ssh, list_tmux_sessions, capture_tmux_pane
from log_parser import parse_output
from config import (
    LDA_DIR, MIXTURES_REL_PATH, TRAIN_SCRIPT_ROOT,
    DEFAULT_ROBOT_TYPE, CONDA_PREFIX, CONDA_ENV,
    BASE_VLM_PATH, PRETRAINED_CKPT, WANDB_ENTITY,
)

console = Console()

# Training script template, consistent with existing cap_r2_2hz.sh
TRAIN_SCRIPT_TEMPLATE = r"""source {conda_prefix}/bin/activate
conda activate {conda_env}
export WANDB_API_KEY=${{WANDB_API_KEY}}
export NCCL_IB_TIMEOUT=10000
export NCCL_IB_RETRY_CNT=10000
export NCCL_IB_AR_THRESHOLD=0

Framework_name=QwenMMDiT
base_vlm={base_vlm_path}
vision_encoder_path=/root

freeze_module_list='qwen_vl_interface,action_model.vision_encoder'
DIT_TYPE="DiT-L"

data_root_dir=/root
data_mix={task_name}

obs_horizon=1
state_dim=58
action_dim=138
max_num_embodiments=32
num_layers=16
use_delta_action=true
positional_embeddings=null
TRAINING_TASK_WEIGHTS="[1,1,1,1]"

seed=42

repeated_diffusion_steps=1

future_obs_index=16
run_root_dir=/workplace/checkpoints
run_id={task_name}

pretrained_checkpoint={pretrained_ckpt}
post_train=true

only_policy=true
policy_and_video_gen=false
only_wo_video_gen=false

export WANDB_MODE=online
wandb_entity={wandb_entity}

output_dir=${{run_root_dir}}/${{run_id}}
mkdir -p ${{output_dir}}
cp $0 ${{output_dir}}/

accelerate launch \
  --config_file {lda_dir}/lda/config/deepseeds/deepspeed_zero2.yaml \
  --num_machines 1 \
  --num_processes 8 \
  {lda_dir}/lda/training/train_LDA.py \
  --config_yaml {lda_dir}/lda/config/training/LDA_pretrain.yaml \
  --seed ${{seed}} \
  --framework.name ${{Framework_name}} \
  --framework.qwenvl.base_vlm ${{base_vlm}} \
  --framework.action_model.vision_encoder_path ${{vision_encoder_path}} \
  --framework.action_model.action_model_type ${{DIT_TYPE}} \
  --framework.action_model.max_num_embodiments ${{max_num_embodiments}} \
  --framework.action_model.state_dim ${{state_dim}} \
  --framework.action_model.action_dim ${{action_dim}} \
  --framework.action_model.obs_horizon ${{obs_horizon}} \
  --framework.action_model.future_obs_index ${{future_obs_index}} \
  --framework.action_model.only_policy ${{only_policy}} \
  --framework.action_model.policy_and_video_gen ${{policy_and_video_gen}} \
  --framework.action_model.only_wo_video_gen ${{only_wo_video_gen}} \
  --framework.action_model.diffusion_model_cfg.num_layers ${{num_layers}} \
  --framework.action_model.diffusion_model_cfg.positional_embeddings ${{positional_embeddings}} \
  --datasets.vla_data.use_delta_action ${{use_delta_action}} \
  --datasets.vla_data.data_root_dir ${{data_root_dir}} \
  --datasets.vla_data.training_task_weights ${{TRAINING_TASK_WEIGHTS}} \
  --datasets.vla_data.data_mix ${{data_mix}} \
  --datasets.vla_data.per_device_batch_size 32 \
  --trainer.freeze_modules ${{freeze_module_list}} \
  --trainer.post_train ${{post_train}} \
  --trainer.max_train_steps 1000000 \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 1000 \
  --trainer.repeated_diffusion_steps ${{repeated_diffusion_steps}} \
  --trainer.learning_rate.base 4e-5 \
  --trainer.pretrained_checkpoint ${{pretrained_checkpoint}} \
  --run_root_dir ${{run_root_dir}} \
  --run_id ${{run_id}} \
  --wandb_project lda-post-train \
  --wandb_entity ${{wandb_entity}} \
  --is_debug False
"""


def check_mixture_exists(node, task_name):
    """Check whether the given task_name already exists in mixtures.py"""
    output = _run_ssh(node, f'grep -c \'"{task_name}"\' {LDA_DIR}/{MIXTURES_REL_PATH}')
    if output and output.strip() != "0":
        return True
    return False


def add_mixture_entry(node, task_name, data_path, robot_type):
    """Append a new entry to the end of DATASET_NAMED_MIXTURES in mixtures.py"""
    entry = (
        f'\n    "{task_name}": [\n'
        f'        ("{data_path}", 1.0, "{robot_type}"),\n'
        f'    ],\n'
    )
    # Use base64 to pass content, completely avoiding nested quoting issues
    b64_entry = base64.b64encode(entry.encode()).decode()
    mixtures_path = f"{LDA_DIR}/{MIXTURES_REL_PATH}"
    py_cmd = (
        f"python3 -c \""
        f"import base64; "
        f"entry = base64.b64decode('{b64_entry}').decode(); "
        f"path = '{mixtures_path}'; "
        f"f = open(path, 'r'); content = f.read(); f.close(); "
        f"idx = content.rfind('}}'); "
        f"content = content[:idx] + entry + content[idx:]; "
        f"f = open(path, 'w'); f.write(content); f.close(); "
        f"print('OK')"
        f"\""
    )
    result = _run_ssh(node, py_cmd, timeout=15)
    return result and "OK" in result


def prepare_task(task, date_str, dry_run):
    """Prepare a single task: check mixture, generate script. Returns an action summary."""
    node = task["node"]
    task_name = task["task_name"]
    data_path = task["data_path"]
    robot_type = task.get("robot_type", DEFAULT_ROBOT_TYPE)

    info = {"node": node, "task_name": task_name, "actions": [], "ok": True}

    # Check mixture
    if check_mixture_exists(node, task_name):
        info["actions"].append(f"mixture '{task_name}' already exists")
    else:
        info["actions"].append(f"ADD mixture '{task_name}' -> {data_path} ({robot_type})")
        if not dry_run:
            ok = add_mixture_entry(node, task_name, data_path, robot_type)
            if not ok:
                info["actions"][-1] += " [FAILED]"
                info["ok"] = False
                return info

    # Generate training script
    script_dir = f"{TRAIN_SCRIPT_ROOT}/{date_str}"
    script_path = f"{script_dir}/{task_name}.sh"
    script_content = TRAIN_SCRIPT_TEMPLATE.format(
        task_name=task_name,
        conda_prefix=CONDA_PREFIX,
        conda_env=CONDA_ENV,
        lda_dir=LDA_DIR,
        base_vlm_path=BASE_VLM_PATH,
        pretrained_ckpt=PRETRAINED_CKPT,
        wandb_entity=WANDB_ENTITY,
    )

    info["script_path"] = script_path
    info["actions"].append(f"CREATE script {script_path}")

    if not dry_run:
        _run_ssh(node, f"mkdir -p {script_dir}")
        write_cmd = f"cat > {script_path} << 'TRAIN_EOF'\n{script_content}TRAIN_EOF"
        _run_ssh(node, write_cmd, timeout=15)
        _run_ssh(node, f"chmod +x {script_path}")

    # Scan all tmux sessions on the node to get current training status
    sessions = list_tmux_sessions(node)
    running = []
    for sess in sessions:
        output = capture_tmux_pane(node, sess)
        train_info = parse_output(sess, output) if output else None
        running.append({"name": sess, "training": train_info})
    info["running_sessions"] = running
    info["actions"].append(f"START new tmux session '{task_name}'")

    return info


def execute_task(task_info):
    """Execute task: kill user-selected sessions, then start new training"""
    node = task_info["node"]
    task_name = task_info["task_name"]
    script_path = task_info["script_path"]

    # Kill sessions the user chose to stop
    for sess_name in task_info.get("kill_sessions", []):
        _run_ssh(node, f"tmux kill-session -t {sess_name} 2>/dev/null")
        console.print(f"  [yellow]Killed session: {sess_name}")

    # Start new session
    _run_ssh(node, f"tmux new-session -d -s {task_name} 'bash {script_path}'")
    console.print(f"  [green]Started: tmux session '{task_name}' on {node}")


def main():
    parser = argparse.ArgumentParser(description="Batch training launcher")
    parser.add_argument(
        "config", help="Path to tasks JSON config file"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only show what would be done"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    tasks = config["tasks"]
    date_str = datetime.now().strftime("%m%d")

    console.print("[bold]Batch Training Launcher[/bold]")
    if args.dry_run:
        console.print("[yellow]DRY RUN MODE[/yellow]")
    console.print(f"Config: {args.config}")
    console.print(f"Date: {date_str}")
    console.print(f"Tasks: {len(tasks)}\n")

    # Phase 1: Prepare all tasks
    task_infos = []
    for task in tasks:
        console.print(f"[bold cyan]{task['node']} / {task['task_name']}[/bold cyan]")
        info = prepare_task(task, date_str, args.dry_run)
        for action in info["actions"]:
            console.print(f"  {action}")
        task_infos.append(info)

    if args.dry_run:
        console.print(f"\n[yellow]DRY RUN — no changes made[/yellow]")
        return

    # Phase 2: Summarize and confirm
    # Show status of all running sessions across nodes
    has_running = [info for info in task_infos if info.get("running_sessions")]
    if has_running:
        console.print("\n[bold yellow]== Current Training Status ==[/bold yellow]")
        status_table = Table(title="Running Sessions")
        status_table.add_column("Node", no_wrap=True)
        status_table.add_column("Session", no_wrap=True)
        status_table.add_column("Step", no_wrap=True)
        status_table.add_column("Progress", no_wrap=True)
        status_table.add_column("Loss", no_wrap=True)
        status_table.add_column("LR", no_wrap=True)
        status_table.add_column("Speed", no_wrap=True)
        status_table.add_column("ETA", no_wrap=True)

        for info in has_running:
            for rs in info["running_sessions"]:
                t = rs["training"]
                if t and t.has_data:
                    status_table.add_row(
                        info["node"],
                        rs["name"],
                        f"{t.step}/{t.total_steps}" if t.total_steps else str(t.step),
                        f"{t.progress_pct:.0f}%" if t.progress_pct else "-",
                        f"{t.loss:.4f}" if t.loss else "-",
                        f"{t.learning_rate:.2e}" if t.learning_rate else "-",
                        t.speed or "-",
                        t.remaining or "-",
                    )
                else:
                    status_table.add_row(
                        info["node"], rs["name"],
                        "-", "-", "-", "-", "-", "-",
                    )
        console.print(status_table)

        # Ask per-node, per-session whether to stop existing training
        console.print("\n[bold]Choose per-session whether to stop existing training:[/bold]")
        for info in has_running:
            info["kill_sessions"] = []
            for rs in info["running_sessions"]:
                t = rs["training"]
                status_str = ""
                if t and t.has_data:
                    parts = []
                    if t.step:
                        parts.append(f"step {t.step}")
                    if t.loss:
                        parts.append(f"loss {t.loss:.4f}")
                    if t.speed:
                        parts.append(t.speed)
                    status_str = f" ({', '.join(parts)})" if parts else ""

                answer = input(
                    f"  Stop {info['node']}/{rs['name']}{status_str}? [y/N] "
                ).strip().lower()
                if answer == "y":
                    info["kill_sessions"].append(rs["name"])
                    info["actions"].append(f"KILL tmux session '{rs['name']}'")
                else:
                    info["actions"].append(f"KEEP tmux session '{rs['name']}'")

    # Show final action summary
    table = Table(title="Actions Summary")
    table.add_column("Node")
    table.add_column("Task")
    table.add_column("Actions")
    for info in task_infos:
        table.add_row(info["node"], info["task_name"], "\n".join(info["actions"]))
    console.print(table)

    answer = input("\nProceed with the above actions? [y/N] ").strip().lower()
    if answer != "y":
        console.print("[yellow]Aborted[/yellow]")
        return

    # Phase 3: Execute
    launched = []
    for info in task_infos:
        if not info["ok"]:
            console.print(f"  [red]Skipping {info['node']}/{info['task_name']} due to earlier error")
            continue
        execute_task(info)
        launched.append(info)

    if not launched:
        return

    # Phase 4: Wait and verify that training started successfully
    console.print(f"\n[bold]Waiting 30s for training to initialize...[/bold]")
    time.sleep(30)

    console.print("[bold]Checking launch status...[/bold]")
    all_ok = True
    for info in launched:
        node = info["node"]
        task_name = info["task_name"]
        output = capture_tmux_pane(node, task_name)
        if not output:
            console.print(f"  [red]{node}/{task_name}: session not found (may have crashed)")
            all_ok = False
            continue

        # Check for errors
        has_error = False
        for keyword in ["Traceback", "Error", "FAILED", "exitcode: 1"]:
            if keyword in output:
                has_error = True
                break

        if has_error:
            console.print(f"  [red]{node}/{task_name}: ERROR detected!")
            # Show last few lines to help debug
            lines = output.strip().split("\n")
            for line in lines[-10:]:
                console.print(f"    [dim]{line}")
            all_ok = False
        else:
            # Try to parse training status
            train_info = parse_output(task_name, output)
            if train_info.has_data:
                console.print(f"  [green]{node}/{task_name}: running (step {train_info.step})")
            else:
                console.print(f"  [green]{node}/{task_name}: initializing...")

    if all_ok:
        console.print(f"\n[bold green]All {len(launched)} tasks launched successfully.[/bold green]")
    else:
        console.print(f"\n[bold red]Some tasks had errors. Check output above.[/bold red]")


if __name__ == "__main__":
    main()
