#input_file 为jsonl文件，每一行需要包含id、image、prompt三个字段。

tp=1
batch_size=1
num_gpus=8

max_new_tokens=20000
input_file=test_without_space.json
checkpoint=Model/InternVL2_2B
result_path=outputs

repetition_penalty=1.0
#repetition_penalty=1.1
temperature=0.0
quant_policy=8
quant_policy=0

version='4'

if [ ! -d "$result_path" ]; then
    echo "目录不存在，正在创建..."
    mkdir -p "$result_path"
else
    echo "目录已存在"
fi
for gpu_index in $(seq 0 $((num_gpus - 1))); do
    echo "gpu_index: ${gpu_index}"
    CUDA_VISIBLE_DEVICES=${gpu_index} python lmdeploy_internvl2_evaluation_mulgpu.py \
        --checkpoint=$checkpoint \
        --test_json_file=${input_file} \
        --outpath=${result_path} \
        --batch_size=${batch_size} \
        --tp=${tp} \
        --temperature=${temperature} \
        --max_new_tokens ${max_new_tokens} \
        --repetition_penalty ${repetition_penalty} \
        --version=${version} \
        --gpu_index=${gpu_index} \
        --num_gpus=${num_gpus} \
        --quant_policy=${quant_policy} \
        &
        #--use_crop \
done

# 等待所有后台任务完成
wait
#--do_sample \
#--spaces_between_special_tokens \
checkpoint_name=$(basename "$checkpoint")
files_to_merge=($(find "$result_path" -type f -name "*${checkpoint_name}_gpu*"))

# 如果没有找到文件，退出
if [ ${#files_to_merge[@]} -eq 0 ]; then
    echo "未找到相关文件"
    exit 1
fi

# 获取第一个文件名并去除 _gpu0~7 后缀
first_file_name=$(basename "${files_to_merge[0]}")
echo "first_file_name: $first_file_name"
merged_file_name=$(echo "$first_file_name" | sed 's/_gpu[0-7]//g')
echo "merged_file_name: $merged_file_name"
# 合并文件
merged_file_path="$result_path/$merged_file_name"
merged_file_path=$(realpath "$merged_file_path")
echo "合并文件到: $merged_file_path"
cat "${files_to_merge[@]}" > "$merged_file_path"

# 删除原始文件
echo "删除原始文件..."
for file in "${files_to_merge[@]}"; do
    echo ${file}
    rm "$file"
done

echo "合并完成"
  
