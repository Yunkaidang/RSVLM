#!/bin/bash
set -x

# wandb login

export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
export GPUS_PER_NODE=7
export NNODES=1
export MASTER_PORT=29504
export CPUS_PER_TASK=32
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
DATA_DIR="/data0/yunkai/VHM_dataset_sft"
export DATA_PATH=${DATA_DIR}
export LIST_FILE=/data0/yunkai/VHM_dataset_sft/list_sft_with_vrsbench.json

CKPT="/home/data/dangyunkai/donghao/VHM_full_deepstack/checkpoints/vhm-7b_pretrained"
export SAVE_PATH=/home/data/dangyunkai/donghao/VHM_full_deepstack/checkpoints/vhm-7b_sft_468

export CKPT_PATH=${CKPT}
export VIT_PATH=/home/data/dangyunkai/donghao/VHM_full_deepstack/checkpoints/vhm-7b_pretrained/vision_tower


# Force DeepStack to inject at LLM layers 2, 4, 6, and 8 during this run without mutating the base checkpoint permanently.
CONFIG_JSON="${CKPT_PATH}/config.json"
CONFIG_BACKUP="${CONFIG_JSON}.layer_backup"

restore_config() {
    if [[ -f "${CONFIG_BACKUP}" ]]; then
        mv -f "${CONFIG_BACKUP}" "${CONFIG_JSON}"
    fi
}

if [[ -f "${CONFIG_JSON}" ]]; then
    cp "${CONFIG_JSON}" "${CONFIG_BACKUP}"
    trap restore_config EXIT
    python - "${CONFIG_JSON}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    config = json.load(f)
target_layers = [4,6,8]
if config.get("deepstack_injection_layers") != target_layers:
    config["deepstack_injection_layers"] = target_layers
    with open(path, "w", encoding="utf-8") as out:
        json.dump(config, out, indent=2)
        out.write("\n")
PY
else
    echo "DeepStack config not found at ${CONFIG_JSON}" >&2
    exit 1
fi

export LEARNING_RATE=2e-5
export TUNE_ENTIRE_MODEL=true
export TUNE_VIT_FROM=-1

MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 \
PYTHONPATH="$(dirname $0)/../..":$PYTHONPATH \
torchrun --nnodes 1 --nproc_per_node $GPUS_PER_NODE --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} \
    /home/data/dangyunkai/donghao/VHM_full_deepstack/vhm/train/train_mem.py \
    --model_name_or_path ${CKPT_PATH} \
    --deepspeed /home/data/dangyunkai/donghao/VHM_full_deepstack/scripts/zero3.json \
    --version v1 \
    --data_path ${DATA_PATH} \
    --vision_tower ${VIT_PATH} \
    --list_file ${LIST_FILE} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer 24 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --deepstack_detail_layers 0.33 0.66 1.0 \
    --deepstack_window_scales 0.5 1.0 \
    --deepstack_downsample_factor 2 \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${SAVE_PATH} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False\
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --run_name ${SAVE_PATH} |& tee -a "$(dirname $0)/finetune_$(date +%Y%m%d_%H%M%S).log"


