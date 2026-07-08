#!/bin/bash
#
# Submit a distillation sensor sweep described by an entry in
# conf/distill_configs.yaml. scripts/expand_distill_config.py expands the
# selected entry into parallel SENSORS / TACTILE_ENCODERS arrays (one slurm job
# per pair). Run names take the form `<BASE>-<tactile_encoder>` -- the encoder
# is the only thing we add to the prefix because main.py's build_run_name
# already appends task/robot/sensors/config_stem.
#
# Every job in the selected subset is submitted; there is no run-status check.
#
# The <mode> argument picks the subset of the config's menu to submit:
#   types        -> baseline sweep across sensor_types at the pinned baseline
#                   (sensor_types x [pinned_resolution] x [pinned_noisy]).
#                   tactile_encoder auto-picked per sensor.
#   models       -> baseline sweep across tactile_encoders against
#                   models_sensor_types (models_sensor_types x
#                   [pinned_resolution] x [pinned_noisy] x tactile_encoders).
#                   No `none` baseline (encoder doesn't apply when sensorless).
#   <type>       -> focused sweep on that sensor type
#                   ([type] x sensor_resolutions x noisy_modes).
#                   tactile_encoder auto-picked per sensor.
#
# Usage:
#   bash scripts/submit_distill.sh <mode> <config_name>
# Examples:
#   bash scripts/submit_distill.sh types  screwdriver-xhand1
#   bash scripts/submit_distill.sh models in_palm_rotate-xhand1
#   bash scripts/submit_distill.sh force_torque screwdriver-xhand1

set -a

BASE_RUN_NAME="STUDENT"
CONFIG_FILE="conf/distill_configs.yaml"

if [ "$#" -lt 2 ]; then
    echo "usage: bash scripts/submit_distill.sh <mode> <config_name>" >&2
    exit 2
fi

MODE="$1"
CONFIG_NAME="$2"

source .env

# Pull TASK / ROBOT / CHECKPOINT / TASK_FLAGS / SENSORS / TACTILE_ENCODERS from the YAML entry.
eval "$(python scripts/expand_distill_config.py "$CONFIG_FILE" "$CONFIG_NAME" "$MODE")"

echo "Config:     $CONFIG_FILE :: $CONFIG_NAME"
echo "Mode:       $MODE"
echo "Task/Robot: $TASK / $ROBOT"
echo "Teacher:    $CHECKPOINT"
echo "Run prefix: $BASE_RUN_NAME-<tactile_encoder>"
echo "Submitting ${#SENSORS[@]} jobs:"

# Student architecture: tac_mlp head whose per-group encoders are picked by
# --tactile_encoder and --encoder (see TACTILE_ENCODER_CFGS / GROUP_ENCODER_CFGS
# in src/model_config.py). The expand script picks --tactile_encoder per job
# (auto for `types`/focused modes, swept for `models` mode).
for i in "${!SENSORS[@]}"; do
    sensor="${SENSORS[i]}"
    tact_enc="${TACTILE_ENCODERS[i]}"
    sensor_name=$(echo "$sensor" | tr / -)
    echo "  ${sensor}  ->  ${tact_enc}"
    sbatch --partition=preempt -J "stu-${sensor_name}-${tact_enc}" babel/run_distill.sh train \
        --resume --no_loaded_video --no_checkpoint_video --stage 2 --model=tac_mlp \
        --tactile_encoder="${tact_enc}" --encoder=rnn \
        --sensors="${sensor}" --run_name="${BASE_RUN_NAME}-${tact_enc}" ${TASK_FLAGS}
done
