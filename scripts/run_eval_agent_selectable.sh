#!/bin/bash

# Usage: ./scripts/run_eval_agent_selectable.sh <agent> [additional args]
# Examples:
#   ./scripts/run_eval_agent_selectable.sh solver_checker_with_tools --model "Qwen2.5-Math-1.5B" --round "test_tools" --dataset "gsm8k" --count 10 --max_iterations 5
#   MODEL=Qwen2.5-Math-1.5B ./scripts/run_eval_agent_selectable.sh majority_vote --count 50

set -e

if [ $# -lt 1 ]; then
  echo "Usage: $0 <agent> [additional args passed to evaluation.eval_agent]"
  exit 1
fi

# Force GPU selection before anything else
export CUDA_VISIBLE_DEVICES=2

tag="$1"
shift

# Defaults (can be overridden by passing the corresponding flags in extra args or via environment)
DEFAULT_MODEL="${MODEL:-Qwen2.5-Math-1.5B}"
DEFAULT_DATASET="${DATASET:-gsm8k}"
DEFAULT_ROUND="${ROUND:-test_${tag}}"
DEFAULT_COUNT="${COUNT:-10}"
DEFAULT_DETAILED="${DETAILED:-false}"
DEFAULT_SAVE_INTERVAL="${SAVE_INTERVAL:-10}"

EXTRA_ARGS="$@"

# Helper to check whether a flag is present in EXTRA_ARGS
has_flag() {
  local flag="$1"
  if [ -z "$EXTRA_ARGS" ]; then
    return 1
  fi
  echo "$EXTRA_ARGS" | grep -q -- "\b${flag}\b" && return 0 || return 1
}

# Build argument string, only append defaults for flags not provided by user
ARGS="--agent \"${tag}\""

if has_flag --model; then
  ARGS="$ARGS $EXTRA_ARGS"
else
  ARGS="$ARGS --model \"${DEFAULT_MODEL}\" $EXTRA_ARGS"
fi

if has_flag --dataset; then :; else ARGS="$ARGS --dataset \"${DEFAULT_DATASET}\""; fi
if has_flag --round; then :; else ARGS="$ARGS --round \"${DEFAULT_ROUND}\""; fi
if has_flag --count; then :; else ARGS="$ARGS --count \"${DEFAULT_COUNT}\""; fi
if has_flag --detailed; then :; else ARGS="$ARGS --detailed \"${DEFAULT_DETAILED}\""; fi
if has_flag --save_interval; then :; else ARGS="$ARGS --save_interval \"${DEFAULT_SAVE_INTERVAL}\""; fi

# Basic model existence check when model flag isn't provided in EXTRA_ARGS
if ! has_flag --model; then
  MODEL_PATH="pretrained_models/${DEFAULT_MODEL}"
  if [ ! -d "$MODEL_PATH" ]; then
    echo "Warning: Model not found at $MODEL_PATH"
    echo "Either place the model under 'pretrained_models/${DEFAULT_MODEL}' or pass --model with a path/name that exists."
  fi
fi

CMD="python -m evaluation.eval_agent $ARGS"

echo "Exported: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Running command:"
echo "$CMD"

eval $CMD
