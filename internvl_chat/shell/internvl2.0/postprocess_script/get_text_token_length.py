import json
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor,as_completed
from transformers import AutoTokenizer
from tqdm import tqdm
'''
from constants import (BOX_END_TOKEN, BOX_START_TOKEN,
                                      IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
                                      IMG_START_TOKEN, QUAD_END_TOKEN,
                                      QUAD_START_TOKEN, REF_END_TOKEN,
                                      REF_START_TOKEN)
'''

os.environ['TOKENIZERS_PARALLELISM'] = 'true'
def process_line(line, tokenizer):
    try:
        data = json.loads(line)
        frame_id = data['id']

        # Compute token length using the tokenizer
        conversations = '\n'.join([temp['value'] for temp in data['conversations']])
        str_length = len(conversations)
        token_length = tokenizer(
            conversations, return_tensors='pt', padding=False, truncation=False,
        ).input_ids.size(1)

        input_ids = tokenizer(conversations, return_tensors='pt', padding=False, truncation=False).input_ids
        text = tokenizer.decode(input_ids[0], spaces_between_special_tokens=False)
        # force_image_size = 448
        # patch_size = 14
        # down_sample_ratio = 0.5
        # num_image_token = int((force_image_size // patch_size) ** 2 * (down_sample_ratio ** 2))
        num_image_token = 256
        max_dynamic_patch = 6
        token_length = token_length + num_image_token * (
                                        max_dynamic_patch + 1)
        # if token_length > tokenizer.model_max_length:
        #     print(f"\nframe_id={frame_id} token长度大于阈值: {token_length} > {tokenizer.model_max_length}\n")
        return 0, frame_id, token_length, 'success'

    except Exception as e:
        print(f"处理行时出错: {e}")
        return 1, frame_id, None, e

def process_in_parallel(submit_data, tokenizer, process_id, thread_num=20):
    out_lines = []
    with ThreadPoolExecutor(max_workers=thread_num) as executor:
        futures = [executor.submit(process_line, line, tokenizer) for line in submit_data]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f'Processing in parallel {process_id}', position=process_id):
            status, frame_id, token_length, error = future.result()
            out_lines.append({'frame_id': frame_id, 'token_length': token_length, 'status': status, 'error': error})
    return out_lines

def init_tokenizer(model_name_or_path='models/OpenGVLab/InternVL2-8B'):
    max_seq_length = 14400
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, add_eos_token=False, trust_remote_code=True, use_fast=False)
    tokenizer.tokenizer_path = model_name_or_path
    tokenizer.model_max_length = max_seq_length
    '''
    token_list = [IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
                  QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN,
                  REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN]
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    '''
    return tokenizer

def get_text_token_length(
        json_file,
        model_name_or_path='models/OpenGVLab/InternVL2-8B',
        process_num=30, thread_num=20, output_path='check_token_output'):
    tokenizer = init_tokenizer(model_name_or_path)

    #tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    #tokenizer.padding_side = "left"

    os.makedirs(output_path, exist_ok=True)
    with open(json_file, 'r') as f:
        dataset = json.load(f)

    for i, (data_name, data_info) in enumerate(dataset.items()):
        annotation = data_info['annotation']
        if not annotation:
            print(f'dataset_name={data_name}, annotation={annotation} is empty!')
            continue
        submit_num = 10000

        with open(annotation, 'r') as f:
            lines = f.readlines()
        #lines = lines[:100]
        print(f"{data_name} num={len(lines)}")
        with ProcessPoolExecutor(max_workers=process_num) as executor, \
            open(f'{output_path}/{data_name}_output.txt', 'w') as out_f:
            futures = []
            process_id = 0
            for i in range(0, len(lines), submit_num):
                submit_data = lines[i:i + submit_num]
                futures.append(executor.submit(process_in_parallel, submit_data, tokenizer, process_id, thread_num))
                process_id += 1
            for future in tqdm(as_completed(futures), total=len(futures), desc=f'Processing {data_name}'):
                out_lines = future.result()
                for line in out_lines:
                    out_f.write(json.dumps(line, ensure_ascii=False) + '\n')

def main():
    '''
    json_file = 'dataset/r2v_sft_version3.json'
    model_name_or_path = 'models/OpenGVLab/InternVL2-8B'
    output_path = 'output_version3'
    process_num = 30
    thread_num = 20
    get_text_token_length(
        json_file=json_file,
        model_name_or_path=model_name_or_path,
        process_num=process_num,
        thread_num=thread_num,
        output_path=output_path
    )
    '''

    jsonl_file = 'dataset/train_data.jsonl'
    model_name_or_path = 'Model/OpenGVLab/InternVL2-2B-add-token'
    output_path = 'debug'
    tokenizer = init_tokenizer(model_name_or_path)
    os.makedirs(output_path, exist_ok=True)

    with open(jsonl_file, 'r') as f:
        for line in f:
            status, frame_id, token_length, error = process_line(line, tokenizer)


    return

if __name__ == '__main__':
    main()
