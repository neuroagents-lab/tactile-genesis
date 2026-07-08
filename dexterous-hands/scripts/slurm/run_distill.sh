#!/bin/bash

#SBATCH --output=logs/%j-%x.out
#SBATCH --job-name=dex_run
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=general
#SBATCH --time=2-0:00:00
#SBATCH --mem=10G
#SBATCH --mail-type=END

# Distillation runner used by scripts/submit_distill.sh. Same SBATCH preamble as
# babel/run.sh, but auto-retries once when main.py exits with
# STALE_CHECKPOINT_EXIT_CODE=77 -- i.e. --resume found a checkpoint whose
# student state_dict no longer matches the live model. main.py wipes the log
# dir before exiting, so this wrapper just re-execs the same command and the
# second attempt finds no checkpoint and starts fresh from the teacher.
# One-off jobs should keep using babel/run.sh (no retry policy needed).

if [ $# -lt 1 ]; then
    echo "Usage: sbatch run_distill.sh train <args...>" >&2
    exit 1
fi

echo "Running on node: $(hostname)"
echo "Args: $@"

set -a
source .env
set +a

MODE="$1"
shift
if [ "$MODE" != "train" ]; then
    echo "run_distill.sh expects MODE=train (got '$MODE'). Use babel/run.sh for other modes." >&2
    exit 2
fi

STALE_EXIT=77   # mirrors STALE_CHECKPOINT_EXIT_CODE in main.py
MAX_TRIES=2
attempt=1
while :; do
    echo "=== run_distill.sh: train attempt $attempt/$MAX_TRIES ==="
    python main.py --mode=train --opt_storage="${DB_URL}" "$@"
    rc=$?
    if [ "$rc" -eq "$STALE_EXIT" ] && [ "$attempt" -lt "$MAX_TRIES" ]; then
        attempt=$((attempt + 1))
        echo "=== run_distill.sh: STALE_CHECKPOINT exit; main.py cleaned log dir, retrying fresh ==="
        continue
    fi
    exit "$rc"
done
