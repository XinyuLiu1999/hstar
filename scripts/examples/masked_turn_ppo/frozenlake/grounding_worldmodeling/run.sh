#!/bin/bash

set -e

# Configuration
PORT=5000
CUDA_DEVICES=${CUDA_DEVICES:-0}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EXPERIMENT_NAME=$(echo $SCRIPT_DIR | rev | cut -d'/' -f1-3 | rev | tr '/' '-')

echo "Experiment name: $EXPERIMENT_NAME"
echo "Using port: $PORT"
echo "Using CUDA devices: $CUDA_DEVICES"

# Set proxy bypass
export NO_PROXY="localhost,127.0.0.1,::1"
export no_proxy="localhost,127.0.0.1,::1"

# Create directories
mkdir -p "data/$EXPERIMENT_NAME"

# Configure environment
export WANDB_API_KEY=e4bb266d6e5a159a1280afa4a476720e92a6dbe7
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONHASHSEED=0

# Activate conda environment
source /root/miniconda3/etc/profile.d/conda.sh || {
    echo "ERROR: Failed to source conda"
    exit 1
}
conda activate vagen || {
    echo "ERROR: Failed to activate vagen environment"
    exit 1
}

# Check if port is in use
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "WARNING: Port $PORT is already in use. Killing existing process..."
    kill $(lsof -t -i:$PORT) 2>/dev/null || true
    sleep 2
fi

# Start server
cd $SCRIPT_DIR
python -m vagen.server.server server.port=$PORT > server_$EXPERIMENT_NAME.log 2>&1 &
SERVER_PID=$!
echo "Server started with PID: $SERVER_PID"

# Cleanup on exit
trap "echo 'Cleaning up...'; kill $SERVER_PID 2>/dev/null || true" EXIT INT TERM

# Wait for server to be ready
echo "Waiting for server to start..."
for i in {1..30}; do
    if curl -s --noproxy localhost http://localhost:$PORT/ > /dev/null 2>&1; then
        echo "✓ Server is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "ERROR: Server failed to start within 60 seconds"
        echo "Check server_$EXPERIMENT_NAME.log for details"
        exit 1
    fi
    echo "Waiting... ($i/30)"
    sleep 2
done

set -x

# Create dataset
python -m vagen.env.create_dataset \
    --yaml_path "$SCRIPT_DIR/env_config.yaml" \
    --train_path "data/$EXPERIMENT_NAME/train.parquet" \
    --test_path "data/$EXPERIMENT_NAME/test.parquet"


# Then start the training
python3 -m vagen.trainer.main_ppo \
    algorithm.adv_estimator=turn_wise_gae \
    algorithm.high_level_gamma=1.0 \
    data.train_files=data/$EXPERIMENT_NAME/train.parquet \
    data.val_files=data/$EXPERIMENT_NAME/test.parquet \
    data.train_batch_size=8 \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    data.max_trajectory_length=8192 \
    data.image_key=images \
    data.truncation=left \
    actor_rollout_ref.model.path=/root/GP/hstar/models/HVS-3B-sft-only \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=mse \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.temperature=0.7 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=/root/GP/hstar/models/HVS-3B-sft-only \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    critic.use_reward_mask=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb', 'console'] \
    trainer.project_name='vagen_new' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=150 \
    trainer.test_freq=20 \
    trainer.total_training_steps=280 \
    rollout_manager.max_turns=5 \
    rollout_manager.window_size=2 \
    rollout_manager.use_multi_turn_reward=False \
    rollout_manager.use_loss_mask=True \
    rollout_manager.use_gae_mask=True \
    trainer.val_before_train=True \
    trainer.val_generations_to_log_to_wandb=8 \
    rollout_manager.n_trajectory=1 \
    rollout_manager.use_service=True \
    rollout_manager.timeout=300 \
    rollout_manager.base_url="http://localhost:$PORT" \
    2>&1 | tee $EXPERIMENT_NAME.log