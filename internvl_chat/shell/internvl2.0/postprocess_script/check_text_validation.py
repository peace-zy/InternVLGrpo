import json
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import copy

def impl_check_text_validation(line):
    def check_wall(wall_list):
        valid = True
        for wall in wall_list:
            for point3d in wall['坐标']:
                if point3d['x'] < 0 or point3d['y'] < 0:
                    valid = False
                    return valid, 'wall point3d x or y < 0'
            wall_item_list = wall.get('墙体附件列表', [])
            for wall_item in wall_item_list:
                start_point = wall_item['坐标']['起始点']
                end_point = wall_item['坐标']['结束点']
                if start_point['x'] < 0 or start_point['y'] < 0 or end_point['x'] < 0 or end_point['y'] < 0:
                    valid = False
                    return valid, 'wall_attach point3d x or y < 0'
        return valid, 'success'
    
    text = json.loads(line)
   
    frame_id = text['id']
    try:
        frame_2_0 = json.loads(text['conversations'][1]['value'])
    except:
        frame_2_0 = eval(text['conversations'][1]['value'])
    valid, error = check_wall(frame_2_0['墙体列表'])
    return valid, f'frame_id={frame_id}, {error}'
    
def check_data(data_name, data_info, position):
    annotation = data_info['annotation']
    image_root = data_info['root']
    file_num = 0
    if not annotation or not image_root:
        print(f'dataset_name={data_name}, annotation={annotation}, image_root={image_root} is empty!')
        return file_num
    
    #check_num = 1000
    del_line_idx = []
    max_workers = 50
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        with open(annotation, 'r') as f:
            for idx, line in enumerate(f):
                futures[executor.submit(impl_check_text_validation, line)] = idx
        file_num = len(futures)
        for future in tqdm(as_completed(futures), total=len(futures), desc=f'Processing {data_name}', position=position):
            valid, error = future.result()
            if not valid:
                print(f'dataset_name={data_name}, annotation={annotation}, error={error}')
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

def check_text_validation(json_file, out_json_file):
    with open(json_file, 'r') as f:
        dataset = json.load(f)
    max_workers = len(dataset)
    new_dataset = copy.deepcopy(dataset)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i, (data_name, data_info) in enumerate(dataset.items()):
            # if data_name != 'real_45_case_book':
            #     continue
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
    from tqdm import tqdm
    infile = 'dataset/train.jsonl'
    with open(infile, 'r') as f:
        for line in tqdm(f.readlines(), desc='check_text_validation'):
            valid, error = impl_check_text_validation(line)
            if not valid:
                print(f'error={error}')
    return

if __name__ == '__main__':
    main()
