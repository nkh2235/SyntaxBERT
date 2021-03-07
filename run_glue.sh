#!/bin/bash
MODEL_DIR=$1
TASK_NAME="SST-2"
CHECKPOINT_DIR="./checkpoint/bert_base"
export PYTHONPATH="$(pwd)"

# python -m torch.distributed.launch --nproc_per_node 4 search_hparam.py \
python search_hparam.py \
  --task_name $TASK_NAME \
  --model_type synbert \
  --model_name_or_path ${CHECKPOINT_DIR} \
  --do_train \
  --do_eval \
  --do_lower_case \
  --max_seq_length 256 \
  --warmup_steps 100 \
  --per_gpu_train_batch_size 8 \
  --per_gpu_eval_batch_size 4 \
  --weight_decay 0.01 \
  --adam_epsilon 1e-8 \
  --max_grad_norm 100.0 \
  --num_train_epochs 3.0 \
  --save_steps -1 \
  --overwrite_output_dir \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 \
  --seed 42 \
  --data_dir dataset/glue_data/${TASK_NAME} \
  --output_dir ./savedir/${TASK_NAME} \
  --available_gpus 4,5,6,7 \
  --need_gpus 1 \
  --conf_file ./confs/conf.json \
