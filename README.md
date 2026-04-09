# Server Training Monitor & Deploy

Monitor and manage training tasks running in tmux sessions across multiple GPU servers. Supports one-click deployment, batch training, and code synchronization.

## Dependencies

```bash
pip install rich
```

## Training Monitor

```bash
python monitor.py              # Live refresh (default every 30 seconds)
python monitor.py --interval 10  # Refresh every 10 seconds
python monitor.py --once        # Single query
```

Press `Ctrl+C` to exit live mode.

### Monitored Fields

| Field | Description |
|-------|-------------|
| Node | Server node name |
| Session | tmux session name |
| Step | Current step / total steps |
| Progress | Training progress percentage |
| Loss | Current loss value |
| LR | Learning rate |
| Speed | Training speed (it/s or s/it) |
| Epoch | Current epoch |
| Elapsed | Elapsed time |
| ETA | Estimated remaining time |

## One-Click Deploy

Finds the latest checkpoint from a training tmux session and launches the SharPA server. Automatically polls until the server is ready and prints the IP/port.

```bash
python deploy.py <node1> <node2>           # Deploy and wait for server ready
python deploy.py <node> --timeout 600      # Custom timeout (default 300s)
python deploy.py <node1> <node2> --dry-run # Preview
python deploy.py <node1> <node2> --kill    # Stop deploy sessions
```

## Batch Training

Launch training tasks in batch via a JSON config file. Automatically: adds dataset entries to `mixtures.py`, generates training scripts, stops old training, and starts new training.

```bash
python batch_train.py tasks.json           # Start training (with confirmation)
python batch_train.py tasks.json --dry-run # Preview
```

### Config File Format (`tasks.json`)

```json
{
    "tasks": [
        {
            "node": "node01",
            "task_name": "my_task_2hz",
            "data_path": "data/RealWorld/My_Task_Post_Train_100_2hz"
        },
        {
            "node": "node02",
            "task_name": "my_task_v2_2hz",
            "data_path": "data/RealWorld/My_Task_V2_Post_Train_50_2hz"
        }
    ]
}
```

- `task_name` is used as both the mixtures.py key and the run_id
- `data_path` is relative to `data_root_dir` on the remote server
- `robot_type` is optional (see `config.py` for default)

## Code Sync

Synchronize code across nodes (especially `mixtures.py`) via git commit -> pull --rebase -> push, with automatic conflict handling.

```bash
python sync.py                    # Sync all nodes
python sync.py <node1> <node2>    # Sync specific nodes only
python sync.py --dry-run          # Preview
```

## Auto Train

Reads a config JSON and deploys a watcher (tmux session) on each target node. The watcher polls locally for data readiness (`meta/modality.json`), then automatically stops old training and launches new training. Runs entirely on the server — safe to close your laptop after deploying.

```bash
python auto_train.py config.json              # Deploy watchers
python auto_train.py config.json --interval 60 # Custom poll interval
python auto_train.py config.json --check       # Check watcher status
python auto_train.py config.json --kill        # Stop all watchers
python auto_train.py config.json --dry-run     # Preview
```

## Checkpoint Migration

Migrate checkpoints from source to destination (e.g. `/root/checkpoints` to `/workplace/checkpoints`). Active training runs are left in place; only old runs are moved. After training ends, `--cleanup` moves the rest and creates a symlink.

```bash
python migrate.py --status                # Show migration status per node
python migrate.py --nodes <node>          # Migrate specific node
python migrate.py --dry-run               # Preview
python migrate.py --cleanup               # Finalize after training ends
```

## Prerequisites

- SSH host aliases configured in `~/.ssh/config` for all target nodes
- Training tasks running inside tmux sessions on each server
- Logs contain tqdm progress bars and Step/Loss formatted output

## Configuration

Edit `config.py` to customize:

- `NODES` — List of server nodes to monitor
- `DEFAULT_INTERVAL` — Default refresh interval (seconds)
- `SSH_TIMEOUT` — SSH connection timeout
- `CAPTURE_LINES` — Number of tmux lines to capture

## File Structure

```
monitor.py         Training monitor (live-refresh table)
deploy.py          One-click SharPA server deployment
batch_train.py     Batch training launcher
auto_train.py      Auto-train (deploy server-side watchers)
sync.py            Multi-node code synchronization
migrate.py         Checkpoint migration tool
config.py          Server configuration
ssh_fetcher.py     SSH connection and tmux output capture
log_parser.py      Training log parser
display.py         Terminal table rendering
configs/           Training task JSON configs
```
