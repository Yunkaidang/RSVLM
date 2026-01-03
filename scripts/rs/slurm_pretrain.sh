#!/bin/bash
set -x

# wandb login
export CUDA_DEVICE_MAX_CONNECTIONS=1
export GPUS_PER_NODE=8
export NNODES=1
export MASTER_PORT=29502
export CPUS_PER_TASK=32
DATA_DIR="/data0/yunkai/VHM_VersaD"
export DATA_PATH=${DATA_DIR}
export LIST_FILE=${DATA_DIR}/list_pretrain.json

PROJECT_ROOT="/home/data/dangyunkai/donghao/MF-RSVLM"
MODEL_ROOT="${PROJECT_ROOT}/models"

export CKPT_PATH="${MODEL_ROOT}/llava-mlp2x/mm_projector.bin"

current_script="$0"
basename=$(basename "$current_script" .sh)
version_flag=$(echo "$basename" | rev | cut -d'_' -f1 | rev)

export SAVE_PATH=${PROJECT_ROOT}/checkpoints/vhm-7b_pretrained
export TUNE_ENTIRE_MODEL=true
export TUNE_VIT_FROM=-1
export BASE_LR=2e-5
export GRADIENT_ACCU_STEPS=1

MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}

mkdir -p "${SAVE_PATH}"

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 \
PYTHONPATH="$(dirname $0)/../..":$PYTHONPATH \
torchrun --nnodes ${NNODES} --nproc_per_node ${GPUS_PER_NODE} --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} \
    ${PROJECT_ROOT}/vhm/train/train_mem.py \
    --deepspeed ${PROJECT_ROOT}/scripts/zero3.json \
    --model_name_or_path ${MODEL_ROOT}/vicuna-7b-v1.5 \
    --version plain \
    --data_path ${DATA_PATH} \
    --list_file ${LIST_FILE} \
    --vision_tower ${MODEL_ROOT}/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --pretrain_mm_mlp_adapter ${CKPT_PATH} \
    --tune_entire_model ${TUNE_ENTIRE_MODEL} \
    --tune_vit_from_layer ${TUNE_VIT_FROM} \
    --mm_vision_select_layer 24 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --fp16 True \
    --output_dir ${SAVE_PATH} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${GRADIENT_ACCU_STEPS} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate ${BASE_LR} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --run_name ${SAVE_PATH} |& tee -a "$(dirname $0)/pretrain_$(date +%Y%m%d_%H%M%S).log"
