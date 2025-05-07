import json
import copy
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_dataset(ds_name, ds_info, new_dataset):
    try:
        annotation = ds_info['annotation']
        if not annotation:
            print(f'dataset_name={ds_name}, annotation={annotation} is empty!')
            return None

        anno_root, anno_file = os.path.split(annotation)
        anno_root = os.path.join(os.path.dirname(anno_root), 'train_data_without_space')
        os.makedirs(anno_root, exist_ok=True)
        new_anno_file = os.path.join(anno_root, anno_file)
        new_dataset[ds_name]['annotation'] = new_anno_file

        with open(annotation, 'r') as f, open(new_dataset[ds_name]['annotation'], 'w') as out_f:
            for line in tqdm(f, total=ds_info['length'], desc=f'Processing {ds_name}', leave=False):
                data = json.loads(line)
                data['conversations'][1]['value'] = data['conversations'][1]['value'].replace(' ', '')
                out_f.write(json.dumps(data, ensure_ascii=False) + '\n')
        return new_anno_file
    except Exception as e:
        print(f"Error processing {ds_name}: {e}")
    return None

def main():
    old_json_file = 'dataset/train_data_with_space.json'
    new_json_file = 'dataset/train_data_without_space.json'

    with open(old_json_file, 'r') as f:
        dataset = json.load(f)

    new_dataset = copy.deepcopy(dataset)

    max_workers = len(dataset)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_dataset, ds_name, ds_info, new_dataset): ds_name for ds_name, ds_info in dataset.items()}
        for future in tqdm(as_completed(futures), total=len(futures), desc='Overall Progress'):
            ds_name = futures[future]
            try:
                new_anno_file = future.result()
                if new_anno_file is not None:
                    new_dataset[ds_name]['annotation'] = new_anno_file
            except Exception as e:
                print(f"Error in future for {ds_name}: {e}")

    with open(new_json_file, 'w') as out_f:
        json.dump(new_dataset, out_f, ensure_ascii=False, indent=4)
if __name__ == '__main__':
    main()
