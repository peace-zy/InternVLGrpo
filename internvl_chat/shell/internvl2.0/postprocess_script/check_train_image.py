import json
import os
from matplotlib import ft2font
from regex import T
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from PIL import Image
import copy

def check_image_file(line, image_root):
    data = json.loads(line)
    image = data['image']
    image_file = os.path.join(image_root, image)
    try:
        Image.open(image_file)
    except Exception as e:
        os.remove(image_file)
        return 1, f'image={image_file} open error'
    return 0, 'success'

def check_data(data_name, data_info, position):
    annotation = data_info['annotation']
    image_root = data_info['root']
    file_num = 0
    if not annotation or not image_root:
        print(f'dataset_name={data_name}, annotation={annotation}, image_root={image_root} is empty!')
        return file_num
    
    #check_num = 1000
    del_line_idx = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {}
        with open(annotation, 'r') as f:
            for idx, line in enumerate(f):
                futures[executor.submit(check_image_file, line, image_root)] = idx
        file_num = len(futures)
        for future in tqdm(as_completed(futures), total=len(futures), desc=f'Processing {data_name}', position=position):
            status, message = future.result()
            if status:
                print(f'dataset_name={data_name}, annotation={annotation}, error={message}')
                del_line_idx.append(futures[future])
                file_num -= 1
    print(f'dataset_name={data_name}, annotation={annotation}, file_num={file_num}, del_line_num={len(del_line_idx)}')
    if del_line_idx:
        with open(annotation, 'r') as f, open(annotation + '.bak', 'w') as out_f:
            for idx, line in enumerate(f):
                if idx not in del_line_idx:
                    out_f.write(line)
        os.rename(annotation + '.bak', annotation)
    return file_num

def check_train_image(json_file, out_json_file):
    with open(json_file, 'r') as f:
        dataset = json.load(f)
    max_workers = len(dataset)
    new_dataset = copy.deepcopy(dataset)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i, (data_name, data_info) in enumerate(dataset.items()):
            futures[executor.submit(check_data, data_name, data_info, i)] = (data_name, data_info)
        
        for future in tqdm(as_completed(futures), total=len(futures), desc='Processing datasets'):
            data_name, data_info = futures[future]
            file_num = future.result()
            data_info['length'] = file_num
            new_dataset[data_name] = data_info
            print(f'new_dataset={new_dataset}')
    with open(out_json_file, 'w') as out_f:
        json.dump(new_dataset, out_f, ensure_ascii=False, indent=4)
def main():
    json_file = 'dataset/r2v_sft_version3.json'
    out_json_file = 'dataset/new_r2v_sft_version3.json'
    check_train_image(json_file, out_json_file)
    return

if __name__ == '__main__':
    main()
