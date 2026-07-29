#!/bin/bash
# run_grpo.sh — Launch GRPO training for code ranking task
#
# This script uses verl (https://github.com/volcengine/verl) with Megatron backend
# to train a code selector model via GRPO (Group Relative Policy Optimization).
#
# Prerequisites:
#   - verl installed: cd third_party/verl && pip install -e .
#   - Model weights available locally (see configs/paths.yaml)
#   - Training data prepared (see training/prepare_data.py)
#
# Usage:
#   # Single node (8 GPUs)
#   bash training/run_grpo.sh
#
#   # Multi-node via Ray (set RAY_ADDRESS)
#   RAY_ADDRESS=auto bash training/run_grpo.sh
#
# All configurable values are read from configs/training.yaml and configs/paths.yaml.
# Override via environment variables when needed.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VERL_ROOT="$PROJECT_ROOT/third_party/verl"

# ══════════════════════════════════════════════════════════════
# Load configuration from YAML (with env var overrides)
# ══════════════════════════════════════════════════════════════
parse_yaml() {
    python3 -c "
import yaml, os, sys
with open('$1') as f:
    cfg = yaml.safe_load(f)
def flatten(d, prefix=''):
    for k, v in d.items():
        key = f'{prefix}{k}'.upper()
        if isinstance(v, dict):
            flatten(v, f'{prefix}{k}_')
        else:
            print(f'{key}={v}')
flatten(cfg)
"
}

# Read configs (environment variables take precedence)
CONFIG_DIR="$PROJECT_ROOT/configs"

# Model
MODEL=${MODEL:-$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_DIR/paths.yaml')); print(c['model']['local_cache'])")}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_DIR/paths.yaml')); print(c['output']['checkpoint_dir'])")}
S3_CKPT_BASE=${S3_CKPT_BASE:-$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_DIR/paths.yaml')); print(c['output']['checkpoint_s3'])")}
DATA_DIR=${DATA_DIR:-$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_DIR/paths.yaml')); print(c['output']['processed_data_dir'])")}

# Training hyperparameters
EXPERIMENT_NAME=${EXPERIMENT_NAME:-$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_DIR/training.yaml')); print(c['experiment']['name'])")}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_DIR/training.yaml')); print(c['training']['total_epochs'])")}
TRAIN_BATCH=${TRAIN_BATCH:-$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_DIR/training.yaml')); print(c['training']['train_batch_size'])")}
LR=${LR:-$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_DIR/training.yaml')); print(c['training']['learning_rate'])")}
KL_LOSS_COEF=${KL_LOSS_COEF:-$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_DIR/training.yaml')); print(c['training']['kl_loss_coef'])")}
CLIP_RATIO=${CLIP_RATIO:-$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_DIR/training.yaml')); print(c['training']['clip_ratio_low'])")}
PPO_EPOCHS=${PPO_EPOCHS:-$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_DIR/training.yaml')); print(c['training']['ppo_epochs'])")}
ROLLOUT_N=${ROLLOUT_N:-$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_DIR/training.yaml')); print(c['training']['rollout_n'])")}
REWARD_MODE=${REWARD_MODE:-$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_DIR/training.yaml')); print(c['training']['reward_mode'])")}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_DIR/training.yaml')); print(c['training']['max_prompt_length'])")}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_DIR/training.yaml')); print(c['training']['max_response_length'])")}
SAVE_FREQ=${SAVE_FREQ:-$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_DIR/training.yaml')); print(c['training']['save_freq'])")}
TEST_FREQ=${TEST_FREQ:-$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_DIR/training.yaml')); print(c['training']['test_freq'])")}

# Cluster
NNODES=${NNODES:-$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_DIR/training.yaml')); print(c['cluster']['num_nodes'])")}
N_GPUS=${N_GPUS:-$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_DIR/training.yaml')); print(c['cluster']['gpus_per_node'])")}
TRAIN_TP=${TRAIN_TP:-$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_DIR/training.yaml')); print(c['cluster']['train_tp'])")}
TRAIN_PP=${TRAIN_PP:-$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_DIR/training.yaml')); print(c['cluster']['train_pp'])")}
TRAIN_EP=${TRAIN_EP:-$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_DIR/training.yaml')); print(c['cluster']['train_ep'])")}
TRAIN_CP=${TRAIN_CP:-$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_DIR/training.yaml')); print(c['cluster']['train_cp'])")}
GEN_TP=${GEN_TP:-$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_DIR/training.yaml')); print(c['cluster']['gen_tp'])")}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_DIR/training.yaml')); print(c['cluster']['gpu_memory_utilization'])")}
USE_DIST_CHECKPOINTING=${USE_DIST_CHECKPOINTING:-$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_DIR/training.yaml')); print(c['cluster']['use_dist_checkpointing'])")}

export REWARD_MODE

# Derived values
train_prompt_mini_bsz=16
train_prompt_micro_bsz=2
ROLLOUT_TEMPERATURE=1.0
ROLLOUT_TOP_P=1.0
ROLLOUT_TOP_K=-1
MAX_MODEL_LEN=$(($MAX_PROMPT_LENGTH + $MAX_RESPONSE_LENGTH))
USE_DYNAMIC_BSZ=True
PPO_MAX_TOKEN_LEN_PER_GPU=$(($MAX_PROMPT_LENGTH + $MAX_RESPONSE_LENGTH))

# Megatron dist_ckpt paths
S3_MODEL_KEY=$(basename "$MODEL")
MCORE_MODEL_PATH="/mnt/models/${S3_MODEL_KEY}_mcore_dist_ckpt"

# Multi-node detection
if [ -n "$RAY_ADDRESS" ]; then
    NNODES_GT1=1
else
    [ "$NNODES" -gt 1 ] 2>/dev/null && NNODES_GT1=1 || NNODES_GT1=""
fi

mkdir -p "$CHECKPOINT_DIR"

# ══════════════════════════════════════════════════════════════
# Pre-flight checks
# ══════════════════════════════════════════════════════════════
echo "=========================================="
echo "Code Ranking GRPO Training (Megatron)"
echo "=========================================="
echo "Model:             $MODEL"
echo "Data dir:          $DATA_DIR"
echo "Checkpoint dir:    $CHECKPOINT_DIR"
echo "Experiment:        $EXPERIMENT_NAME"
echo "GPUs per node:     $N_GPUS"
echo "Nodes:             $NNODES"
echo "Total GPUs:        $((NNODES * N_GPUS))"
echo "Batch size:        $TRAIN_BATCH"
echo "Epochs:            $TOTAL_EPOCHS"
echo "LR:                $LR"
echo "Reward mode:       $REWARD_MODE"
echo "Parallelism:       TP=$TRAIN_TP PP=$TRAIN_PP EP=$TRAIN_EP CP=$TRAIN_CP"
echo "Rollout TP:        $GEN_TP"
echo "Max seq len:       $MAX_MODEL_LEN"
echo ""

# ══════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════
python3 -m verl.trainer.main_ppo \
    --config-name="ppo_megatron_trainer" \
    algorithm.adv_estimator=grpo \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_batch_size=$TRAIN_BATCH \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    custom_reward_function.path="$SCRIPT_DIR/reward_score.py" \
    custom_reward_function.name=reward_func \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$train_prompt_mini_bsz \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$train_prompt_micro_bsz \
    actor_rollout_ref.actor.ppo_epochs=$PPO_EPOCHS \
    actor_rollout_ref.actor.clip_ratio_low=$CLIP_RATIO \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.use_dynamic_bsz=$USE_DYNAMIC_BSZ \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TRAIN_TP \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$TRAIN_PP \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=$TRAIN_EP \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1 \
    actor_rollout_ref.actor.megatron.context_parallel_size=$TRAIN_CP \
    actor_rollout_ref.actor.megatron.sequence_parallel=True \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    actor_rollout_ref.actor.megatron.grad_offload=True \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=$USE_DIST_CHECKPOINTING \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path=$MCORE_MODEL_PATH \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_activation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_dropout_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=alltoall \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=$TRAIN_TP \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=$TRAIN_PP \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=$TRAIN_EP \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=1 \
    actor_rollout_ref.ref.megatron.context_parallel_size=$TRAIN_CP \
    actor_rollout_ref.ref.megatron.sequence_parallel=True \
    actor_rollout_ref.ref.megatron.param_offload=True \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=$USE_DIST_CHECKPOINTING \
    actor_rollout_ref.ref.megatron.dist_checkpointing_path=$MCORE_MODEL_PATH \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$train_prompt_micro_bsz \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$USE_DYNAMIC_BSZ \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$GEN_TP \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.temperature=$ROLLOUT_TEMPERATURE \
    actor_rollout_ref.rollout.top_p=$ROLLOUT_TOP_P \
    actor_rollout_ref.rollout.top_k=$ROLLOUT_TOP_K \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$train_prompt_micro_bsz \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$USE_DYNAMIC_BSZ \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.nccl_timeout=14400 \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=True \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='Code Selection' \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.default_local_dir="$CHECKPOINT_DIR" \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=$NNODES \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.max_actor_ckpt_to_keep=1 \
    +trainer.s3_checkpoint_dir="$S3_CKPT_BASE" \
    ${NNODES_GT1:++ray_kwargs.ray_init.address=auto} \
    ${NNODES_GT1:++ray_kwargs.ray_init.ignore_reinit_error=true}

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
