import argparse
import json
import os
import torch
import math
from tqdm import tqdm
from lmdeploy.vl import load_image
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig
import requests
from io import BytesIO
import base64
from PIL import Image

def load_eval_data(test_json_file):
    eval_data = []
    suffix_text = '。不要在一个列表里重复生成同一个墙体，并确保生成的是可解析的JSON格式。'
    suffix_text = ''
    with open(test_json_file, mode="r", encoding="utf-8") as f:
        test_dataset = json.load(f)
        for dataset_name, dataset_info in tqdm(test_dataset.items(), desc="Loading datasets"):
            test_info = dataset_info.get('test_info', {})
            if not test_info:
                print(f'{dataset_name} has no test_info')
                continue
            root = test_info["root"]
            with open(test_info['annotation'], 'r') as f1:
                for line in tqdm(f1.readlines(), desc=f'Processing {dataset_name}'):
                    data = json.loads(line)
                    data["image"] = os.path.join(root, data["image"])
                    #assert 'image' in data, "No need for field image in the current task."
                    #assert 'question' in data, "No need for field question in the current task."
                    eval_data.append(
                        {
                            'image_id': data['id'],
                            'input_image_path': data['image'],
                            'input_text': data['question'] + suffix_text,
                            'raw': data,
                        }
                    )
    return eval_data

def decode_image(b64_str):
    img_data = base64.b64decode(b64_str)
    image_stream = BytesIO(img_data)
    image = Image.open(image_stream)
    return image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--test_json_file', type=str, default='')
    parser.add_argument('--test_file', type=str, default='')
    parser.add_argument('--test_image_dir', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--tp', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--do_sample', action='store_true')
    parser.add_argument('--outpath', type=str, default='results')
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    parser.add_argument('--version', type=str, default='0')
    parser.add_argument('--spaces_between_special_tokens', action='store_true')
    parser.add_argument('--quant_policy', type=int, default=0, help='Quantization policy')

    args = parser.parse_args()

    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)
    
    if args.test_json_file == '' and args.test_file == '' and args.test_image_dir == '':
        raise ValueError('Please specify either test_json_file or test_file or test_image_dir.')
    if args.test_json_file:
        eval_data = load_eval_data(test_json_file=args.test_json_file)
    elif args.test_file:
        eval_data = []
        question = '请将该户型图转换为标准的矢量json格式，并确保其是一个有效的户型图，需要包含入户门，外轮廓完整，不能有密闭的空间，并且每个房间都有明确的名称类型，如客厅、卧室、厨房、卫生间等。不要在一个列表里重复生成同一个墙体，并确保生成的是可解析的JSON格式。'
        eval_data.append(
                {
                    'input_image_path': args.test_file,
                    'input_text': question,
                    'raw': {},
                }
        )
    elif args.test_image_dir:
        eval_data = []
        question = '请将该户型图转换为标准的矢量json格式，并确保其是一个有效的户型图，需要包含入户门，外轮廓完整，不能有密闭的空间，并且每个房间都有明确的名称类型，如客厅、卧室、厨房、卫生间等。不要在一个列表里重复生成同一个墙体，并确保生成的是可解析的JSON格式。'
        for root, dirs, files in os.walk(args.test_image_dir):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png'):
                    eval_data.append(
                            {
                                'input_image_path': os.path.join(root, file),
                                'input_text': question,
                                'raw': {},
                            }
                    )

    pipe = pipeline(
            model_path=args.checkpoint,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map='cuda',
            backend_config=TurbomindEngineConfig(tp=args.tp, quant_policy=args.quant_policy),
            chat_template_config=None,
            trust_remote_code=True
    )
    '''
    gen_config = GenerationConfig(
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1
    )
    '''

    gen_config = GenerationConfig(
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            repetition_penalty=args.repetition_penalty,
            #spaces_between_special_tokens=args.spaces_between_special_tokens
    )

    total_params = sum(p.numel() for p in pipe.vl_encoder.model.model.parameters()) / 1e9

    print(f'[test] args: {args}')
    print(f'[test] total_params: {total_params}B')
    print(f'[test] config: {pipe.vl_encoder.model.config}')
    model_name = os.path.basename(args.checkpoint)
    #output_filename= f"{model_name}_v{args.version}_tp{args.tp}_bs{args.batch_size}_mnk{args.max_new_tokens}"\
    output_filename= f"{model_name}_tp{args.tp}_bs{args.batch_size}_mnk{args.max_new_tokens}"\
                     f"_rp{args.repetition_penalty}_temperature{args.temperature}_sample{args.do_sample}_res.jsonl"
    output_file = os.path.join(args.outpath, output_filename)
    with open(output_file, "w") as f:
        for i in tqdm(range(0, len(eval_data), args.batch_size), total=math.ceil(len(eval_data) / args.batch_size), desc='eval'):
            batch = eval_data[i: i + args.batch_size]

            prompts = [(sample['input_text'], load_image(sample['input_image_path'])) for sample in batch]
            responses = pipe(prompts, gen_config=gen_config)

            for sample, response in zip(batch, responses):
                output = sample['raw']
                output['response'] = response.text.replace(' ', '')
                if not response.text:
                    print(f"response.text is None for {output['id']}")
                f.write(json.dumps(output, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    main()
