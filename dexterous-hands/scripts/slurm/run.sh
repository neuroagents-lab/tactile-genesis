#!/bin/bash

#SBATCH --output=logs/%j-%x.out
#SBATCH --job-name=dex_run
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=general
#SBATCH --time=2-0:00:00
#SBATCH --mem=10G
#SBATCH --mail-type=END

if [ $# -lt 1 ]; then
    echo "Usage: sbatch run.sh <mode> <args...>"
    echo "  mode: train, play, or path to script"
    exit 1
fi

set -e

echo "Running on node: $(hostname)"
echo "Args: $@"

# Export environment variables from .env file
set -a  # automatically export all variables
source .env
set +a


MODE=$1
shift  # Remove first argument, rest are passed to Python script

case $MODE in
    train)
        python main.py --mode=train --opt_storage=${DB_URL} "$@"
        ;;
    play)
        python main.py --mode=play "$@"
        ;;
    optimize | opt)
        python main.py --mode=optimize --opt_storage=${DB_URL} "$@"
        ;;
    rollout)
        python main.py --mode=rollout-benchmark --opt_storage=${DB_URL} "$@"
        ;;
    *)
        python $MODE "$@"
        ;;
esac