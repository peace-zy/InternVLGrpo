checkpoint=InternVL2_2B

#input_file 为jsonl文件，每一行需要包含id、image、prompt三个字段。
test_file=11000000440476_0fa2d32a8e15a7388aa5b89ec0383418_composite.png

tp=1
batch_size=1
max_new_tokens=20000
result_path=outputs_debug

repetition_penalty=1.0
temperature=0.0
version='3'

if [ ! -d "$result_path" ]; then
    echo "目录不存在，正在创建..."
    mkdir -p "$result_path"
else
    echo "目录已存在"
fi

python lmdeploy_internvl2_evaluation.py \
    --checkpoint=$checkpoint \
    --test_file=${test_file} \
    --outpath=${result_path} \
    --batch_size=${batch_size} \
    --tp=${tp} \
    --max_new_tokens ${max_new_tokens} \
    --repetition_penalty ${repetition_penalty} \
    --version=${version} \
    2>&1 | tee "log_eval_gpu${tp}.txt"
    #2>&1 | tee -a "log_eval.txt"
