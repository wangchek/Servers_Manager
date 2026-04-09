"""Global configuration — constants shared by all modules"""

import os

# ============================================================
# Local settings
# ============================================================

# List of monitored servers (corresponding to Host aliases in ~/.ssh/config)
_nodes_env = os.environ.get("NODES", "")
NODES = [n.strip() for n in _nodes_env.split(",") if n.strip()] if _nodes_env else [f"node{i}" for i in range(45, 52)]

# Default refresh interval (seconds)
DEFAULT_INTERVAL = 30

# SSH timeout (seconds)
SSH_TIMEOUT = 10

# Number of lines to capture from tmux capture-pane
CAPTURE_LINES = 80

# ============================================================
# Remote server paths
# ============================================================

# LDA project root directory
LDA_DIR = os.environ.get("LDA_DIR", "/root/LDA")

# Conda environment
CONDA_PREFIX = os.environ.get("CONDA_PREFIX", "/root/miniconda3")
CONDA_ENV = f"{CONDA_PREFIX}/envs/LDA"

# mixtures.py relative path (relative to LDA_DIR)
MIXTURES_REL_PATH = "lda/dataloader/gr00t_lerobot/mixtures.py"

# Training script root directory
TRAIN_SCRIPT_ROOT = f"{LDA_DIR}/scripts/run_scripts/post-train"

# Deploy script root directory
DEPLOY_ROOT = f"{LDA_DIR}/scripts/deploy"

# Checkpoint root directory
CKPT_ROOT = os.environ.get("CKPT_ROOT", "/workplace/checkpoints")

# SharPA server.py path
SERVER_PY = f"{LDA_DIR}/examples/sharpa/server.py"

# Deploy server default port
DEPLOY_PORT = 10093

# ============================================================
# Git sync
# ============================================================

# Remote repository branch name
GIT_BRANCH = os.environ.get("GIT_BRANCH", "main")

# ============================================================
# Training default parameters
# ============================================================

# Default robot_type
DEFAULT_ROBOT_TYPE = "galbot_sharpa_no_history"

# Base VLM model path (on remote server)
BASE_VLM_PATH = os.environ.get("BASE_VLM_PATH", "/root/Qwen3-VL-4B-Instruct")

# Pretrained checkpoint path (on remote server)
PRETRAINED_CKPT = os.environ.get("PRETRAINED_CKPT", "/root/pretrained/LDA-pretrain.pt")

# Weights & Biases entity name
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "")

# ============================================================
# Auto-training (auto_train)
# ============================================================

# Dataset root directory (on remote server)
DATA_ROOT = os.environ.get("DATA_ROOT", "/root/data/RealWorld")

# Data-ready marker file (relative to dataset directory)
DATA_READY_MARKER = "meta/modality.json"
