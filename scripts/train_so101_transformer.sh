#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT_DIR/.venv312/bin/python}"
MODE="${1:-all}"

# 625k total gives an exact 500k / 62.5k / 62.5k (80/10/10) split.
TOTAL_TRAJECTORIES="${TOTAL_TRAJECTORIES:-625000}"
HORIZON="${HORIZON:-100}"
PARALLEL_ENVS="${PARALLEL_ENVS:-64}"
ACTION_SCALE="${ACTION_SCALE:-0.01}"
ACTION_HOLD_STEPS="${ACTION_HOLD_STEPS:-20}"
SEED="${SEED:-42}"
TRAINING_STEPS="${TRAINING_STEPS:-${EPOCHS:-500000}}"
BATCH_SIZE="${BATCH_SIZE:-512}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
GENERATION_PRINT_INTERVAL="${GENERATION_PRINT_INTERVAL:-10000}"
VALIDATION_INTERVAL="${VALIDATION_INTERVAL:-5000}"
VALIDATION_WINDOWS="${VALIDATION_WINDOWS:-4096}"
PRINT_INTERVAL="${PRINT_INTERVAL:-1000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-10000}"
TENSORBOARD_PORT="${TENSORBOARD_PORT:-6006}"
AUTO_RESUME="${AUTO_RESUME:-1}"
REGENERATE="${REGENERATE:-0}"

RUN_NAME="${RUN_NAME:-so101_transformer_${TOTAL_TRAJECTORIES}}"
TRAINING_CONFIG="${TRAINING_CONFIG:-so101}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/$RUN_NAME}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/outputs/tensorboard/$RUN_NAME}"
TRAIN_DATASET="$OUTPUT_DIR/train.hdf5"
VALIDATION_DATASET="$OUTPUT_DIR/validation.hdf5"
TEST_DATASET="$OUTPUT_DIR/test.hdf5"
CHECKPOINT="$OUTPUT_DIR/model.pt"
LATEST_CHECKPOINT="${CHECKPOINT%.pt}.latest.pt"
PID_FILE="$OUTPUT_DIR/pipeline.pid"

if [[ ! -x "$PYTHON" ]]; then
    echo "Python interpreter is not executable: $PYTHON" >&2
    exit 1
fi
if (( TOTAL_TRAJECTORIES < 10 )); then
    echo "TOTAL_TRAJECTORIES must be at least 10 for an 80/10/10 split" >&2
    exit 1
fi
if (( HORIZON < 10 )); then
    echo "HORIZON must be at least 10 for the Transformer history window" >&2
    exit 1
fi

TRAIN_TRAJECTORIES=$((TOTAL_TRAJECTORIES * 80 / 100))
VALIDATION_TRAJECTORIES=$((TOTAL_TRAJECTORIES * 10 / 100))
TEST_TRAJECTORIES=$((TOTAL_TRAJECTORIES - TRAIN_TRAJECTORIES - VALIDATION_TRAJECTORIES))
TRAIN_WINDOWS=$((TRAIN_TRAJECTORIES * (HORIZON - 10 + 1)))
STEPS_PER_PASS=$(((TRAIN_WINDOWS + BATCH_SIZE - 1) / BATCH_SIZE))
SAMPLED_WINDOWS=$((TRAINING_STEPS * BATCH_SIZE))

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$MODE" != "tensorboard" ]]; then
    if [[ -s "$PID_FILE" ]]; then
        running_pid="$(<"$PID_FILE")"
        if kill -0 "$running_pid" 2>/dev/null; then
            echo "Pipeline is already running with PID $running_pid" >&2
            exit 1
        fi
    fi
    echo "$$" > "$PID_FILE"
    trap 'rm -f "$PID_FILE"' EXIT
fi

print_config() {
    cat <<EOF
SO-101 Transformer NeRD pipeline
  mode:                  $MODE
  trajectories:          $TOTAL_TRAJECTORIES
  split:                 $TRAIN_TRAJECTORIES / $VALIDATION_TRAJECTORIES / $TEST_TRAJECTORIES
  horizon:               $HORIZON
  parallel environments: $PARALLEL_ENVS
    action hold steps:     $ACTION_HOLD_STEPS
    optimizer steps:        $TRAINING_STEPS
    batch size:             $BATCH_SIZE
    windows per data pass:  $TRAIN_WINDOWS
    steps per data pass:    $STEPS_PER_PASS
    sampled windows:        $SAMPLED_WINDOWS
  output:                 $OUTPUT_DIR
  TensorBoard logs:       $LOG_DIR
EOF
}

generate_split() {
    local split="$1"
    local count="$2"
    local split_seed="$3"
    local path="$4"
    local partial_path="${path}.partial"

    if [[ -s "$path" && "$REGENERATE" != "1" ]]; then
        echo "Using existing $split dataset: $path"
        return
    fi
    rm -f "$partial_path"

    "$PYTHON" "$ROOT_DIR/examples/example_robot_nerd_train.py" \
        --generate-only \
        --splits "$split" \
        --robot-id so101 \
        --training-config "$TRAINING_CONFIG" \
        --num-envs "$PARALLEL_ENVS" \
        --num-trajectories "$count" \
        --horizon "$HORIZON" \
        --seed "$split_seed" \
        --action-scale "$ACTION_SCALE" \
        --action-hold-steps "$ACTION_HOLD_STEPS" \
        --generation-print-interval "$GENERATION_PRINT_INTERVAL" \
        "--${split}-dataset-path" "$partial_path"
    mv "$partial_path" "$path"
}

generate_data() {
    generate_split train "$TRAIN_TRAJECTORIES" "$SEED" "$TRAIN_DATASET"
    generate_split validation "$VALIDATION_TRAJECTORIES" "$((SEED + 1))" "$VALIDATION_DATASET"
    generate_split test "$TEST_TRAJECTORIES" "$((SEED + 2))" "$TEST_DATASET"
}

train_model() {
    "$PYTHON" -m training.train \
        --config "$TRAINING_CONFIG" \
        --train-dataset "$TRAIN_DATASET" \
        --validation-dataset "$VALIDATION_DATASET" \
        --test-dataset "$TEST_DATASET" \
        --checkpoint "$CHECKPOINT" \
        --steps "$TRAINING_STEPS" \
        --batch-size "$BATCH_SIZE" \
        --learning-rate "$LEARNING_RATE" \
        --seed "$SEED"

    echo "Checkpoint: $CHECKPOINT"
    echo "View losses: $0 tensorboard"
}

start_tensorboard() {
    exec "$ROOT_DIR/.venv312/bin/tensorboard" \
        --logdir "$LOG_DIR" \
        --host 0.0.0.0 \
        --port "$TENSORBOARD_PORT"
}

print_config
case "$MODE" in
    generate)
        generate_data
        ;;
    train)
        train_model
        ;;
    all)
        generate_data
        train_model
        ;;
    tensorboard)
        start_tensorboard
        ;;
    *)
        echo "Usage: $0 [generate|train|all|tensorboard]" >&2
        exit 2
        ;;
esac