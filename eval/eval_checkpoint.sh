#!/bin/bash
# eval_checkpoint.sh — Evaluate a GRPO checkpoint on LiveCodeBench
#
# Controls which model is used for candidate generation vs ranking:
#   GENERATOR   "base" or "ckpt" (default: ckpt)
#   RANKER      "base" or "ckpt" (default: ckpt)
#
# Modes:
#   GENERATOR=ckpt RANKER=ckpt  → GRPO model does both (standard eval)
#   GENERATOR=base RANKER=ckpt  → base generates, GRPO ranks
#   GENERATOR=base RANKER=base  → pure baseline
#   GENERATOR=ckpt RANKER=base  → GRPO generates, base ranks
#
# Environment variables:
#   S3_CKPT_DIR        S3 path to checkpoint (e.g. s3://.../global_step_10)
#   BASE_MODEL_S3      S3 or local path to base model HF weights
#   EVAL_TASK          Benchmark task (default: LiveCodeBenchSelectionID)
#   LCB_VERSION        v5 or v6 (default: v6)
#   N_ROLLOUTS         Number of candidate rollouts (default: 4)
#   TP_SIZE            Tensor parallel size (default: 4)
#   GPU_MEM            GPU memory utilization (default: 0.90)
#   REPEAT_IDX         Repeat index for seed offset (default: 0)
#   SELECTION_MODE     "ranking" or "selection" (default: ranking)
#   SAMPLING_PARAMS    vLLM sampling params string
#   RESULTS_DIR        Local directory for results
#
# Prerequisites:
#   - verl installed (for checkpoint merging)
#   - evalchemy installed (for eval framework)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ─── Parameters ───
GENERATOR=${GENERATOR:-ckpt}
RANKER=${RANKER:-ckpt}
REPEAT_IDX=${REPEAT_IDX:-0}
N_REPEAT=${N_REPEAT:-1}
CKPT_STEP=${CKPT_STEP:-0}
S3_CKPT_DIR=${S3_CKPT_DIR:-}
BASE_MODEL_S3=${BASE_MODEL_S3:-}
SELECTION_MODE=${SELECTION_MODE:-ranking}
LCB_VERSION=${LCB_VERSION:-v6}
EVAL_TASK=${EVAL_TASK:-LiveCodeBenchSelectionID}
N_ROLLOUTS=${N_ROLLOUTS:-4}
TP_SIZE=${TP_SIZE:-4}
GPU_MEM=${GPU_MEM:-0.90}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-}
SAMPLING_PARAMS=${SAMPLING_PARAMS:-"temperature=0.6,top_p=0.95,top_k=20,min_p=0"}
RESULTS_DIR=${RESULTS_DIR:-./output/eval_results}

MODEL_ARGS_SUFFIX=""
if [ -n "$MAX_MODEL_LEN" ]; then
    MODEL_ARGS_SUFFIX=",max_model_len=$MAX_MODEL_LEN"
fi

# Seed with repeat offset for reproducibility across repeats
SEED_OFFSET=$((REPEAT_IDX * 1000))
SEED="$((0 + SEED_OFFSET)),$((1234 + SEED_OFFSET)),$((1234 + SEED_OFFSET)),$((1234 + SEED_OFFSET))"

# Selection mode flag
SELECTION_MODE_ARG=""
if [ "$SELECTION_MODE" = "ranking" ]; then
    SELECTION_MODE_ARG="--selection_mode ranking"
fi

# LCB-specific args
LCB_ARGS=""
if [[ "$EVAL_TASK" == *"LiveCodeBench"* ]]; then
    LCB_ARGS="--lcb_version $LCB_VERSION"
fi

# Model directories
CKPT_MODEL_DIR=/tmp/eval_hf_model
BASE_MODEL_DIR=/tmp/eval_base_model
NEED_CKPT=false
NEED_BASE=false

[[ "$GENERATOR" == "ckpt" || "$RANKER" == "ckpt" ]] && NEED_CKPT=true
[[ "$GENERATOR" == "base" || "$RANKER" == "base" ]] && NEED_BASE=true

echo "=========================================="
echo "Eval GRPO Checkpoint"
echo "  EVAL_TASK:    $EVAL_TASK"
echo "  Generator:    $GENERATOR"
echo "  Ranker:       $RANKER"
echo "  Repeat:       $REPEAT_IDX"
echo "  Seed:         $SEED"
echo "  N_ROLLOUTS:   $N_ROLLOUTS"
echo "  TP_SIZE:      $TP_SIZE"
echo "  LCB_VERSION:  $LCB_VERSION"
echo "  SEL_MODE:     $SELECTION_MODE"
echo "=========================================="

# ─── Step 1: Prepare models ───
echo "Step 1: Preparing models (need_ckpt=$NEED_CKPT, need_base=$NEED_BASE)"

if [ "$NEED_CKPT" = true ]; then
    if [ -z "$S3_CKPT_DIR" ]; then
        echo "ERROR: S3_CKPT_DIR required when GENERATOR=ckpt or RANKER=ckpt"
        exit 1
    fi

    echo "  Merging distributed checkpoint → HF model"
    cd "$PROJECT_ROOT/third_party/verl"
    pip install -e . -q

    mkdir -p /tmp/ckpt_download/actor/dist_ckpt
    echo "  Downloading from $S3_CKPT_DIR"
    aws s3 sync "${S3_CKPT_DIR}/actor/dist_ckpt/" /tmp/ckpt_download/actor/dist_ckpt --quiet

    # Download base model for architecture config
    mkdir -p /tmp/ckpt_download/actor/huggingface
    if [ -n "$BASE_MODEL_S3" ]; then
        aws s3 sync "$BASE_MODEL_S3" /tmp/ckpt_download/actor/huggingface --quiet
    else
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('${BASE_MODEL_HF:-Qwen/Qwen3-30B-A3B-Instruct-2507}', local_dir='/tmp/ckpt_download/actor/huggingface')
"
    fi

    python -m verl.model_merger merge --backend megatron \
        --local_dir /tmp/ckpt_download/actor \
        --target_dir "$CKPT_MODEL_DIR" \
        --use_cpu_initialization

    cd "$PROJECT_ROOT"
fi

if [ "$NEED_BASE" = true ]; then
    if [ -d /tmp/ckpt_download/actor/huggingface ] && [ -n "$(ls -A /tmp/ckpt_download/actor/huggingface 2>/dev/null)" ]; then
        BASE_MODEL_DIR=/tmp/ckpt_download/actor/huggingface
    elif [ -n "$BASE_MODEL_S3" ]; then
        mkdir -p "$BASE_MODEL_DIR"
        aws s3 sync "$BASE_MODEL_S3" "$BASE_MODEL_DIR" --quiet
    else
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('${BASE_MODEL_HF:-Qwen/Qwen3-30B-A3B-Instruct-2507}', local_dir='$BASE_MODEL_DIR')
"
    fi
fi

# Resolve model paths
GENERATOR_MODEL_DIR="$( [ "$GENERATOR" = "ckpt" ] && echo "$CKPT_MODEL_DIR" || echo "$BASE_MODEL_DIR" )"
RANKER_MODEL_DIR="$( [ "$RANKER" = "ckpt" ] && echo "$CKPT_MODEL_DIR" || echo "$BASE_MODEL_DIR" )"

echo "  Generator model: $GENERATOR_MODEL_DIR"
echo "  Ranker model:    $RANKER_MODEL_DIR"

# ─── Step 2: Setup evalchemy ───
echo "Step 2: Setting up evalchemy"
cd "$PROJECT_ROOT/third_party/evalchemy"
if [ ! -d .venv ]; then
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -e . -q
else
    source .venv/bin/activate
fi

# ─── Step 3: Generate candidates if split mode ───
if [ "$GENERATOR" != "$RANKER" ]; then
    echo "Step 3: Generating candidates with $GENERATOR model"
    CANDIDATES_LOCAL=/tmp/eval_candidates

    python -m eval.eval \
        --model vllm \
        --model_args "pretrained=$GENERATOR_MODEL_DIR,tensor_parallel_size=$TP_SIZE,gpu_memory_utilization=$GPU_MEM$MODEL_ARGS_SUFFIX" \
        --tasks "$EVAL_TASK" \
        --batch_size auto \
        --max_tokens 65536 \
        --n_rollouts "$N_ROLLOUTS" \
        --sampling_params "$SAMPLING_PARAMS" \
        $SELECTION_MODE_ARG \
        $LCB_ARGS \
        --seed "$SEED" \
        --output_path logs_gen \
        --n_repeat "$N_REPEAT"

    # Copy candidates for ranker
    mkdir -p "$CANDIDATES_LOCAL"
    cp -r "${RESULTS_DIR}/${EVAL_TASK}_repeat_"* "$CANDIDATES_LOCAL/" 2>/dev/null || true
    export CANDIDATES_DIR="$CANDIDATES_LOCAL"
    echo "  CANDIDATES_DIR=$CANDIDATES_DIR"
else
    echo "Step 3: Skipped (generator == ranker)"
fi

# ─── Step 4: Run eval ───
echo "Step 4: Running eval with $RANKER model"

python -m eval.eval \
    --model vllm \
    --model_args "pretrained=$RANKER_MODEL_DIR,tensor_parallel_size=$TP_SIZE,gpu_memory_utilization=$GPU_MEM$MODEL_ARGS_SUFFIX" \
    --tasks "$EVAL_TASK" \
    --batch_size auto \
    --max_tokens 65536 \
    --n_rollouts "$N_ROLLOUTS" \
    --sampling_params "$SAMPLING_PARAMS" \
    $SELECTION_MODE_ARG \
    $LCB_ARGS \
    --seed "$SEED" \
    --output_path "$RESULTS_DIR/step_${CKPT_STEP}/repeat_${REPEAT_IDX}" \
    --n_repeat "$N_REPEAT"

echo "=========================================="
echo "Eval complete! (generator=$GENERATOR, ranker=$RANKER, repeat=$REPEAT_IDX)"
echo "Results: $RESULTS_DIR/step_${CKPT_STEP}/repeat_${REPEAT_IDX}"
echo "=========================================="
