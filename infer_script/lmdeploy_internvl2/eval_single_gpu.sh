test_image_dir=images
checkpoint=InternVL2_8B
result_path=outputs_2B
tp=1
batch_size=1
num_gpus=1
max_new_tokens=20000
repetition_penalty=1.0
#repetition_penalty=1.1
temperature=0.0
quant_policy=0
#quant_policy=8

version='3'

if [ ! -d "$result_path" ]; then
    echo "目录不存在，正在创建..."
    mkdir -p "$result_path"
else
    echo "目录已存在"
fi

python lmdeploy_internvl2_evaluation.py \
        --checkpoint=${checkpoint} \
        --test_image_dir=${test_image_dir} \
        --outpath=${result_path} \
        --batch_size=${batch_size} \
        --tp=${tp} \
        --temperature=${temperature} \
        --max_new_tokens ${max_new_tokens} \
        --repetition_penalty ${repetition_penalty} \
        --version=${version} \
        --quant_policy=${quant_policy}

