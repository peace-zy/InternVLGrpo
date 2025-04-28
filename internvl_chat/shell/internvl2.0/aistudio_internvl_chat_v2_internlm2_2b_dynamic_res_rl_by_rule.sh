
# 青云多机配置
set -x
set -e

# 设置时区为上海
#cat /etc/issue
#apk add tzdata
rm -f /etc/localtime
ln -s /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
date

#pip config set global.index-url http://nexus.kcs.ke.com/repository/kcs-pip-proxy/simple
#pip config set global.trusted-host nexus.kcs.ke.com

rm -rf ~/.cache/pip
conda clean --all -y

pip install --no-cache-dir timm==0.9.12
#pip install --no-cache-dir timm==1.0.12
#pip install --no-cache-dir deepspeed==0.15.4
pip install --no-cache-dir opencv-python-headless==4.10.0.84
pip install --no-cache-dir imageio==2.36.1
pip install --no-cache-dir decord==0.6.0
pip install --no-cache-dir shapely==2.0.6
pip install --no-cache-dir matplotlib==3.9.2
pip install --no-cache-dir pypinyin==0.53.0
#pip install --no-cache-dir transformers==4.44.2


# pip install --no-cache-dir transformers==4.46.3
# sed -i '4098i\
#         for key, value in model.named_parameters():\
#             logger.info(f"instaned model with {key} of dtype {value.dtype} on device {value.device} with shape {value.shape}")' /opt/conda/lib/python3.10/site-packages/transformers/modeling_utils.py


export NCCL_SOCKET_IFNAME=bond0
export GLOO_SOCKET_IFNAME=bond0
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_11
export NCCL_ALGO=nvls
export NCCL_COLLNET_ENABLE=1
export NCCL_IB_QPS_PER_CONNECTION=2
export CUDA_DEVICE_MAX_CONNECTIONS=1


is_multinode=true
is_multinode=false
if [ $is_multinode = true ]; then
    nproc_per_node=8
    nnodes=${WORLD_SIZE}
    node_rank=${RANK}
    master_addr=$(cat /etc/aistudio/master-host)
    master_port=6000
else
    nproc_per_node=8
    nnodes=1
    node_rank=0
    master_addr=localhost
    master_port=23458
fi

# node_rank=$1
# echo ${node_rank}
#########################################

Model_name_or_path=model
meta_path=filtered_badcase.json
OUTPUT_DIR=output

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "目录不存在，正在创建..."
    mkdir -p "$OUTPUT_DIR"
else
    echo "目录已存在"
fi
#########################################
# number of gpus: 256
# batch size per gpu: 4
# gradient accumulation steps: 1
# total batch size: 1024
# epoch: 1
#########################################
PER_DEVICE_BATCH_SIZE=2
GRADIENT_ACC=4
save_steps=2000
max_len=14400
epochs=5
epochs=1
#epochs=2
#epochs=1

DISTRIBUTED_ARGS="--nproc_per_node $nproc_per_node --nnodes $nnodes --node_rank $node_rank --master_addr $master_addr --master_port $master_port"
LOG_FILE="${OUTPUT_DIR}/training_log_node_${RANK}.txt"
torchrun ${DISTRIBUTED_ARGS}  internvl/train/internvl_chat_finetune_with_rl_by_rule.py \
  --model_name_or_path $Model_name_or_path \
  --conv_style "internlm2-chat" \
  --output_dir ${OUTPUT_DIR} \
  --meta_path $meta_path \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 6 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.1 \
  --pad2square False \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone False \
  --vision_select_layer -1 \
  --use_data_resampling False \
  --dataloader_num_workers 32 \
  --bf16 True \
  --num_train_epochs $epochs \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps $save_steps \
  --save_total_limit 1 \
  --learning_rate 2e-6 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length $max_len \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "zero_stage1_config.json" \
  --report_to "tensorboard" \
  --max_completion_length 20000 \
  --do_sample False \
  --top_k 10 \
  --top_p 1.0 \
  --temperature 0.0 \
  --beta 0.00 \
  --reward_threshold 1 \
  --epsilon 0.04 \
  2>&1 | tee ${LOG_FILE}
  #--reward_threshold 0.999 \
  #--reward_threshold 0.95 \
  #--beta 0.00 \
  #--epsilon 0.1 \
  #--online_rl_by_rule True \
  #--use_vllm True \
  #--deepspeed "zero_stage3_config.json" \
  #--report_to "tensorboard" > ${LOG_FILE} 2>&1
sleep 100d
set +x
set +e
