# 青云多机配置
set -x
set -e

# 设置时区为上海
#cat /etc/issue
#apk add tzdata
rm -f /etc/localtime
ln -s /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
date

rm -rf ~/.cache/pip
conda clean --all -y
# pip config set global.trusted-host nexus.kcs.ke.com
# pip config set global.index-url http://nexus.kcs.ke.com/repository/kcs-pip-proxy/simple

pip install --no-cache-dir timm==0.9.12 #旧环境 0.9.12 loss从0.3下降
pip install --no-cache-dir opencv-python-headless==4.10.0.84
pip install --no-cache-dir imageio==2.36.1
pip install --no-cache-dir decord==0.6.0


# update env
#pip install --no-cache-dir timm==1.0.12
# to be tested
#pip install --no-cache-dir deepspeed==0.16.2


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
#is_multinode=false
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

Model_name_or_path=InternVL2_2B
meta_path=sft_version4_without_space_2nd.json
OUTPUT_DIR=InternVL2_2B_v4_without_space

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "目录不存在，正在创建..."
    mkdir -p "$OUTPUT_DIR"
else
    echo "目录已存在"
fi
#########################################
# number of gpus: 8
# batch size per gpu: 4
# gradient accumulation steps: 4
# total batch size: 128
# epoch: 1
#########################################
PER_DEVICE_BATCH_SIZE=2
PER_DEVICE_BATCH_SIZE=3
GRADIENT_ACC=1
GRADIENT_ACC=4

save_steps=5000
max_len=14400
epochs=5
epochs=3

DISTRIBUTED_ARGS="--nproc_per_node $nproc_per_node --nnodes $nnodes --node_rank $node_rank --master_addr $master_addr --master_port $master_port"
LOG_FILE="${OUTPUT_DIR}/training_log_node_${RANK}.txt"
torchrun ${DISTRIBUTED_ARGS}  internvl/train/internvl_chat_finetune.py \
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
  2>&1 | tee ${LOG_FILE}
 
  #--learning_rate 4e-5 \
  #--learning_rate 2e-5 \
  #--warmup_ratio 0.03 \
  #--report_to "tensorboard" > ${LOG_FILE} 2>&1
  #--deepspeed "zero_stage1_config.json" \
sleep 100d
set +x
set +e
