"""Parse training logs to extract step, loss, lr, speed, and other metrics"""

import re
from dataclasses import dataclass


@dataclass
class TrainInfo:
    """Training status data"""
    session: str = ""
    step: int = 0
    total_steps: int = 0
    progress_pct: float = 0.0
    elapsed: str = ""
    remaining: str = ""
    speed: str = ""
    loss: float = 0.0
    learning_rate: float = 0.0
    epoch: float = 0.0
    has_data: bool = False


# tqdm progress bar, supporting both it/s and s/it formats:
#   6%|▊  | 56100/1000000 [18:28:32<199:31:14,  1.31it/s, ...]
#   6%|▊  | 56800/1000000 [18:42:18<164:56:45,  1.32s/it, ...]
_TQDM_PATTERN = re.compile(
    r"(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*"
    r"\[(\d+:\d+:\d+)<(\d+:\d+:\d+),\s*([\d.]+)(it/s|s/it)"
)

# Step/Loss/LR/Epoch — searched across the entire text since Step lines may span multiple lines.
# Note: the value after 'loss': may be interrupted by a right-aligned filename (e.g., train_LDA.py:309),
# with the actual number on the next line, so [\s\S]{0,60}? is used to allow cross-line matching.
_STEP_PATTERN = re.compile(r"Step\s+(\d+)")
_LOSS_PATTERN = re.compile(r"'loss':[\s\S]{0,60}?([\d]+\.[\d.eE\-+]+)")
_LR_PATTERN = re.compile(r"'learning_rate':[\s\S]{0,60}?([\d]+\.[\d.eE\-+]+)")
_EPOCH_PATTERN = re.compile(r"'epoch':[\s\S]{0,60}?([\d]+\.[\d.eE\-+]+)")


def parse_output(session_name: str, output: str) -> TrainInfo:
    """Parse training information from tmux output"""
    info = TrainInfo(session=session_name)
    if not output:
        return info

    # ---- Parse tqdm progress bar (take the last match) ----
    tqdm_matches = list(_TQDM_PATTERN.finditer(output))
    if tqdm_matches:
        m = tqdm_matches[-1]
        info.progress_pct = float(m.group(1))
        info.step = int(m.group(2))
        info.total_steps = int(m.group(3))
        info.elapsed = m.group(4)
        info.remaining = m.group(5)
        info.speed = m.group(6) + " " + m.group(7)
        info.has_data = True

    # ---- Parse Step/Loss/LR/Epoch ----
    # Step lines are multi-line, so find all Step occurrences in the entire text and take the last one
    step_matches = list(_STEP_PATTERN.finditer(output))
    if step_matches:
        last_step_match = step_matches[-1]
        info.step = int(last_step_match.group(1))
        info.has_data = True

        # From this Step position, extract a chunk of text to parse loss/lr/epoch
        # (these fields appear within a few lines after Step)
        chunk = output[last_step_match.start():]
        # Truncate at the next Step or at most 500 characters
        next_step = _STEP_PATTERN.search(chunk[10:])
        if next_step:
            chunk = chunk[:next_step.start() + 10]
        else:
            chunk = chunk[:500]

        m_loss = _LOSS_PATTERN.search(chunk)
        if m_loss:
            info.loss = float(m_loss.group(1))

        m_lr = _LR_PATTERN.search(chunk)
        if m_lr:
            info.learning_rate = float(m_lr.group(1))

        m_epoch = _EPOCH_PATTERN.search(chunk)
        if m_epoch:
            info.epoch = float(m_epoch.group(1))

    return info
