#!/bin/bash
#SBATCH --job-name=sft
#SBATCH --account=infra01
#SBATCH --time=10:00:00
#SBATCH --exclusive
#SBATCH --nodes=24
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=5
#SBATCH --mem=460800
#SBATCH --partition=normal
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --exclude=nid006634,nid006701,nid006948,nid006588,nid006629,nid006910,nid007254,nid007078,nid006619,nid006840,nid006905,nid006941,nid006947,nid006922,nid007074,nid007131,nid007189,nid007129,nid007184,nid007176,nid007177,nid007183,nid007090,nid007551,nid007531,nid007539,nid007558,nid006988,nid006990,nid006987,nid006989,nid007363,nid006606,nid007410,nid007096,nid007566,nid006774,nid007343,nid006867,nid007323,nid007489,nid006676,nid006677,nid007411,nid006848,nid006681,nid007626,nid007612,nid006887,nid006577,nid006729,nid006831,nid007520,nid007589,nid007614,nid006955,nid007592,nid007344,nid007374,nid007134,nid007628,nid007382,nid007141,nid007155,nid007286,nid006589,nid007024,nid007025

set -ex

# Cluster and infrastructure
MASTER_PORT=30000
REPO_DIR="/iopsstor/scratch/cscs/msantelmo/SSFT"
WORK_DIR="${REPO_DIR}/verl_sft"

# Paths
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/outputs/sft_1}"

## Apertus v1 from scratch
# baseline
MODEL_PATH="/iopsstor/scratch/cscs/msantelmo/checkpoints/Apertus-8B-2509"
TOKENIZER_PATH="/iopsstor/scratch/cscs/msantelmo/checkpoints/Apertus-8B-Instruct-2509"
DATASET_PATH="/iopsstor/scratch/cscs/msantelmo/SSFT/data/sft_1/sft0+teacher-baseline"
LEARNING_RATE="5e-5"
# # cap-filter-fill
# MODEL_PATH="/iopsstor/scratch/cscs/msantelmo/checkpoints/Apertus-8B-2509"
# TOKENIZER_PATH="/iopsstor/scratch/cscs/msantelmo/checkpoints/Apertus-8B-Instruct-2509"
# DATASET_PATH="/iopsstor/scratch/cscs/msantelmo/SSFT/data/sft_1/cap-filter-fill-apertus-8b-2509-sft0-step11776-mix-sft0"
# LEARNING_RATE="5e-5"
# # cap-filter-fill - smaller lr
# MODEL_PATH="/iopsstor/scratch/cscs/msantelmo/checkpoints/Apertus-8B-2509"
# TOKENIZER_PATH="/iopsstor/scratch/cscs/msantelmo/checkpoints/Apertus-8B-Instruct-2509"
# DATASET_PATH="/iopsstor/scratch/cscs/msantelmo/SSFT/data/sft_1/cap-filter-fill-apertus-8b-2509-sft0-step11776-mix-sft0"
# LEARNING_RATE="1e-5"

CUSTOM_CLS_NAME="ApertusSFTDataset"
MODEL_DTYPE="bfloat16"

# Data configuration
MAX_LENGTH=32_768
TRAIN_BATCH_SIZE=512
VAL_BATCH_SIZE=512
ROLLOUT_BATCH_SIZE=64
ROLLOUT_MAX_LENGTH=4096
TRAIN_FILE="train.parquet"
VAL_PATH="/iopsstor/scratch/cscs/msantelmo/SSFT/data/sft_0/val.parquet"
ROLLOUT_PATH="/iopsstor/scratch/cscs/msantelmo/SSFT/data/sft_1/test_think.parquet"  # NOTE: we use enable_thinking=true in evals
USE_DYNAMIC_BSZ=true
SEQ_PARALLEL=2  # set to >1 to enable sequence parallelism
MAX_TOKEN_LEN_PER_GPU=16_384

# Training configuration
WARMUP_STYLE="linear"
LR_WARMUP_STEPS_RATIO=0.03
TOTAL_EPOCHS=2
if [[ "$DATASET_PATH" == *"sft0+"* ]] || [[ "$DATASET_PATH" == *"mix-sft0"* ]]; then
    TEST_FREQ=512
    SAVE_FREQ=512
else
    TEST_FREQ=128
    SAVE_FREQ=256
fi
ROLLOUT_NODES=8
ROLLOUT_MAX_CONCURRENT_REQUESTS=2048
MAX_CKPT_TO_KEEP=5
TOTAL_TRAINING_STEPS=null
WEIGHT_DECAY=0.0

# Experiment configuration
PROJECT_NAME="${PROJECT_NAME:-ssft}"

# Multi-turn configuration
ENABLE_MULTITURN=true
MESSAGES_KEY="messages"
TOOLS_KEY="tools"
ENABLE_THINKING_KEY="enable_thinking"
CUSTOM_CLS_PATH="verl/utils/dataset/multiturn_sft_dataset.py"

# Rollout evaluation
ROLLOUT_MODEL_PATH="$MODEL_PATH"
ROLLOUT_TEMPERATURE=0.7
ROLLOUT_TOP_P=0.95
ROLLOUT_NUM_SAMPLES=64

# Logging
LOGGER='["console","wandb"]'
WANDB_MODE="${WANDB_MODE:-online}"

# Generate run name and experiment name from key parameters
MODEL_NAME=${MODEL_ALIAS:-$(basename "$MODEL_PATH")}
DATASET_NAME=$(basename "$DATASET_PATH")
RUN_NAME="${RUN_NAME:-${MODEL_NAME}__${DATASET_NAME}__sp${SEQ_PARALLEL}-lr${LEARNING_RATE}-bs${TRAIN_BATCH_SIZE}-warmup${WARMUP_STYLE}-lr_warmup_steps_ratio${LR_WARMUP_STEPS_RATIO}__$(date '+%Y%m%d-%H%M%S')}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"
LOG_DIR="${RUN_DIR}/logs"

# Environment
VERL_ENVIRONMENT="/capstor/store/cscs/swissai/infra01/reasoning/imgs/projects/verl_swiss:1/env.toml"
SGLANG_ROUTER_ENVIRONMENT="/capstor/store/cscs/swissai/infra01/reasoning/users/nathanrchn/images/sglang_router/env.toml"

# The run directory must exist before Slurm resolves its stdout/stderr paths.
# Launch this file as a shell script; it submits itself after resolving the run.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    mkdir -p "$LOG_DIR"
    sbatch \
        --output="$LOG_DIR/slurm_%j.out" \
        --error="$LOG_DIR/slurm_%j.err" \
        --export=ALL,PROJECT_NAME="$PROJECT_NAME",RUN_NAME="$RUN_NAME",RUN_DIR="$RUN_DIR",OUTPUT_ROOT="$OUTPUT_ROOT" \
        "$0"
    echo "Run directory: $RUN_DIR"
    exit 0
fi

nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 --nodelist=$head_node hostname -i)

save_path="$RUN_DIR"
dataset_path=$DATASET_PATH

mkdir -p "$save_path" "$LOG_DIR"
cp "$0" "$save_path/sft_1.sh"

# Build project wheel once on the head node and store it in a shared location
WHEEL_DIR="$save_path/wheels"
mkdir -p "$WHEEL_DIR"
srun --nodes=1 --ntasks=1 --nodelist=$head_node --container-writable --environment=$VERL_ENVIRONMENT --kill-on-bad-exit=1 --output=$LOG_DIR/wheel_build.log --error=$LOG_DIR/wheel_build.err \
    bash --norc --noprofile -c "\
set -ex
cd $WORK_DIR
pip wheel . --no-cache-dir --no-deps -w $WHEEL_DIR"
PACKAGE_WHEEL=$(ls -t "$WHEEL_DIR"/*.whl | head -n1)

TRAINING_NODES=$((SLURM_NNODES - ROLLOUT_NODES))
rollout_nodes=("${nodes[@]:$TRAINING_NODES:$ROLLOUT_NODES}")
rollout_node_ips=()
for node in "${rollout_nodes[@]}"; do
    rollout_node_ips+=("$(srun --nodes=1 --ntasks=1 --nodelist=$node hostname -i)")
done
ROLLOUT_URL="http://${rollout_node_ips[0]}:30000"

if [ "$SEQ_PARALLEL" -gt 1 ]; then
    SEQ_PARALLEL_FLAG="engine.ulysses_sequence_parallel_size=$SEQ_PARALLEL"
else
    SEQ_PARALLEL_FLAG=""
fi


# Rename slurm job to have the run name in the Slurm queue
scontrol update JobId=$SLURM_JOB_ID Name="$RUN_NAME"

for local_rank in $(seq 0 $((TRAINING_NODES - 1))); do
    node=${nodes[$local_rank]}

    if [ "$local_rank" -eq 0 ]; then
        verifier_install_command="pip install evaluate math-verify latex2sympy2-extended pylatexenc antlr4-python3-runtime==4.9.3"
        verifier_check_command="python3 -c 'import emoji, langdetect, math_verify, nltk, syllapy'"
    else
        verifier_install_command=""
        verifier_check_command=""
    fi

    srun --nodes=1 --ntasks=1 --nodelist=$node --container-writable --environment=$VERL_ENVIRONMENT --kill-on-bad-exit=1 --output=$LOG_DIR/node_${local_rank}.log --error=$LOG_DIR/node_${local_rank}.err \
        bash --norc --noprofile -c "\
set -ex

echo $node

export no_proxy="0.0.0.0,$no_proxy"
export NO_PROXY="0.0.0.0,$NO_PROXY"

export NCCL_DEBUG=INFO
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED="1"
export RAY_DEDUP_LOGS="0"
export WANDB_MODE="$WANDB_MODE"
export WANDB_DIR="$save_path/wandb"

cd $WORK_DIR

$verifier_install_command

pip install --no-deps /capstor/store/cscs/swissai/infra01/reasoning/users/nathanrchn/wheels/nltk-3.9.2-py3-none-any.whl
pip install --no-deps /capstor/store/cscs/swissai/infra01/reasoning/users/nathanrchn/wheels/emoji-2.15.0-py3-none-any.whl
pip install --no-deps /capstor/store/cscs/swissai/infra01/reasoning/users/nathanrchn/wheels/syllapy-0.7.2-py3-none-any.whl
pip install --no-deps /capstor/store/cscs/swissai/infra01/reasoning/users/nathanrchn/wheels/langdetect-1.0.9-py3-none-any.whl
pip install --no-deps /capstor/store/cscs/swissai/infra01/reasoning/users/nathanrchn/wheels/immutabledict-4.2.2-py3-none-any.whl

pip install $PACKAGE_WHEEL --no-cache-dir --no-deps --force-reinstall

$verifier_check_command

torchrun --nnodes=$TRAINING_NODES --nproc_per_node=4 --node_rank=$local_rank --master_addr=$head_node_ip --master_port=$MASTER_PORT \
    -m verl.trainer.sft_trainer \
    data.train_files=$dataset_path/$TRAIN_FILE \
    data.val_files=$VAL_PATH \
    data.rollout_files=$ROLLOUT_PATH \
    data.messages_key=$MESSAGES_KEY \
    data.tools_key=$TOOLS_KEY \
    data.enable_thinking_key=$ENABLE_THINKING_KEY \
    data.custom_cls.path=$CUSTOM_CLS_PATH \
    data.custom_cls.name=$CUSTOM_CLS_NAME \
    data.max_length=$MAX_LENGTH \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.rollout_batch_size=$ROLLOUT_BATCH_SIZE \
    +data.rollout_max_size=$ROLLOUT_MAX_LENGTH \
    +data.rollout_sampling_params.temperature=$ROLLOUT_TEMPERATURE \
    +data.rollout_sampling_params.top_p=$ROLLOUT_TOP_P \
    +data.rollout_sampling_params.n=$ROLLOUT_NUM_SAMPLES \
    data.rollout_max_concurrent_requests=$ROLLOUT_MAX_CONCURRENT_REQUESTS \
    data.use_dynamic_bsz=$USE_DYNAMIC_BSZ \
    data.max_token_len_per_gpu=$MAX_TOKEN_LEN_PER_GPU \
    +data.apply_chat_template_kwargs.truncation=true \
    +data.add_generation_prompt=false \
    model.path=$MODEL_PATH \
    model.tokenizer_path=$TOKENIZER_PATH \
    model.use_remove_padding=true \
    model.enable_gradient_checkpointing=true \
    engine.model_dtype=$MODEL_DTYPE \
    engine.strategy=fsdp \
    optim.lr=$LEARNING_RATE \
    optim.warmup_style=$WARMUP_STYLE \
    optim.weight_decay=$WEIGHT_DECAY \
    optim.lr_warmup_steps_ratio=$LR_WARMUP_STEPS_RATIO \
    rollout_url=$ROLLOUT_URL \
    checkpoint.save_contents='["model","optimizer","extra","hf_model"]' \
    trainer.default_local_dir=$save_path \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$RUN_NAME \
    trainer.test_freq=$TEST_FREQ \
    trainer.save_freq=$SAVE_FREQ \
    trainer.max_ckpt_to_keep=$MAX_CKPT_TO_KEEP \
    trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.logger=$LOGGER \
    hydra.run.dir=$save_path/hydra $SEQ_PARALLEL_FLAG" &
done

WORKER_URLS=""
for node_idx in $(seq 0 $((ROLLOUT_NODES - 1))); do
    node=${rollout_nodes[$node_idx]}
    node_ip=${rollout_node_ips[$node_idx]}

    for i in $(seq 0 3); do
        port=$((50000 + i))
        WORKER_URLS="${WORKER_URLS} http://${node_ip}:${port}"
        srun --nodes=1 --ntasks=1 --nodelist=$node --container-writable --environment=$VERL_ENVIRONMENT --kill-on-bad-exit=1 --gpus-per-task=1 --cpus-per-task=50 --gpu-bind=map_gpu:${i} --overlap --output=$LOG_DIR/node_rollout_${node_idx}_$i.log --error=$LOG_DIR/node_rollout_${node_idx}_$i.err \
            bash --norc --noprofile -c "\
set -ex

export no_proxy=\"0.0.0.0,$no_proxy\"
export NO_PROXY=\"0.0.0.0,$NO_PROXY\"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$i
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

python -m sglang.launch_server --model-path=$ROLLOUT_MODEL_PATH --tokenizer-path=$TOKENIZER_PATH --dtype=$MODEL_DTYPE --host=0.0.0.0 --port=$port --decode-log-interval=1 --skip-server-warmup --random-seed=42 --grammar-backend=llguidance --mem-fraction-static=0.6 --max-running-requests=60" &
    done
done

srun --nodes=1 --ntasks=1 --nodelist=${rollout_nodes[0]} --container-writable --environment=$SGLANG_ROUTER_ENVIRONMENT --kill-on-bad-exit=1 --cpus-per-task=50 --overlap --output=$LOG_DIR/node_rollout_router.log --error=$LOG_DIR/node_rollout_router.err \
    bash --norc --noprofile -c "\
set -ex

export no_proxy=\"0.0.0.0,$no_proxy\"
export NO_PROXY=\"0.0.0.0,$NO_PROXY\"

python -m sglang_router.launch_router --host=0.0.0.0 --port=30000 --worker-urls $WORKER_URLS --model-path=$ROLLOUT_MODEL_PATH" &

wait -n
scancel $SLURM_JOB_ID

echo "[FINISHED]"
