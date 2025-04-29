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

def load_eval_data(test_json_file, gpu_index, num_gpus):
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
                    eval_data.append(
                        {
                            'image_id': data['id'],
                            'input_image_path': data['image'],
                            'input_text': data['question'] + suffix_text,
                            'raw': data,
                        }
                    )
    
    # 根据GPU编号对数据进行分片
    eval_data = eval_data[gpu_index::num_gpus]
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
    parser.add_argument('--gpu_index', type=int, default=0, help='GPU index to use')
    parser.add_argument('--num_gpus', type=int, default=1, help='Total number of GPUs')
    parser.add_argument('--spaces_between_special_tokens', action='store_true')
    parser.add_argument('--quant_policy', type=int, default=0, help='Quantization policy')

    args = parser.parse_args()

    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)

    eval_data = load_eval_data(test_json_file=args.test_json_file, gpu_index=args.gpu_index, num_gpus=args.num_gpus)

    pipe = pipeline(
            model_path=args.checkpoint,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map='cuda',
            backend_config=TurbomindEngineConfig(tp=args.tp, quant_policy=args.quant_policy),
            chat_template_config=None,
            trust_remote_code=True
    )

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
    output_filename= f"{model_name}_gpu{args.gpu_index}_tp{args.tp}_bs{args.batch_size}_mnk{args.max_new_tokens}"\
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
