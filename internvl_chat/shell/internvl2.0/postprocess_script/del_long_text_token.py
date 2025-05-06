import json
from tqdm import tqdm
import copy
import os

def del_long_text_token(json_file, token_info_path, output_path, len_thres=14400):
    os.makedirs(output_path, exist_ok=True)

    with open(json_file, 'r') as f:
        dataset = json.load(f)
    new_dataset = copy.deepcopy(dataset)

    for data_name, data_info in dataset.items():
        annotation = data_info['annotation']
        if not annotation:
            print(f'dataset_name={data_name}, annotation={annotation} is empty!')
            continue
        length = data_info['length']
        data_dict = {}
        long_data = []
        with open(annotation, 'r') as f:
            for line in tqdm(f, total=length, desc=f'Loading {data_name}'):
                data = json.loads(line)
                frame_id = data['id']
                data_dict[frame_id] = line

        token_info_file = f'{token_info_path}/{data_name}_output.txt'
        with open(token_info_file, 'r') as f:
            for line in tqdm(f.readlines(), desc=f'Checking {data_name}'):
                data = json.loads(line)
                frame_id = data['frame_id']
                token_length = data['token_length']
                if token_length > len_thres:
                    print(f"frame_id={frame_id} token长度大于阈值: {token_length} > {len_thres}")
                    d = data_dict.pop(frame_id)
                    long_data.append(d)

        new_dataset[data_name]['length'] = len(data_dict)

        if long_data:
            print(f"dataset={data_name}, long_data num={len(long_data)}")
            dirname, filename = os.path.split(annotation)
            new_annotation = f'{dirname}/filter_long_token_{filename}'
            new_dataset[data_name]['annotation'] = new_annotation
            with open(new_annotation, 'w') as out_f:
                for frame_id, data in data_dict.items():
                    out_f.write(data)
            output_file = f'{output_path}/{data_name}.jsonl'
            with open(output_file, 'w') as out_f:
                for d in long_data:
                    out_f.write(d)
    with open(json_file, 'w') as out_f:
        json.dump(new_dataset, out_f, ensure_ascii=False, indent=4)

def main():
    json_file = 'dataset/filtered_r2v_sft_version2.json'
    token_info_path = 'text_token_length_output'
    output_path = 'long_text_token_output'
    len_thres = 14400
    del_long_text_token(
        json_file=json_file,
        token_info_path=token_info_path,
        output_path=output_path,
        len_thres=len_thres
    )
if __name__ == '__main__':
    main()
