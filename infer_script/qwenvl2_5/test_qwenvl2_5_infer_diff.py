import os
import sys
import json
from tqdm import tqdm
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers.utils.import_utils import _is_package_available

from vllm import LLM, SamplingParams
from tqdm import tqdm

def load_eval_data(test_json_file):
    eval_data = []
    suffix_text = ''
    suffix_text = '用中文回答'
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
    return eval_data

def infer_with_transformers(
        model_path, 
        eval_data, 
        output_dir, 
        temperature=0.0, 
        max_new_tokens=256, 
        do_sample=False, 
        repetition_penalty=1.0, 
        top_p=1.0,
        top_k=1,
        dtype=torch.bfloat16,
    ):
    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        #attn_implementation="flash_attention_2",
        device_map="auto",
    )

    # The default range for the number of visual tokens per image in the model is 4-16384.
    # You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
    min_pixels = 256*28*28
    max_pixels = 1280*28*28
    processor = AutoProcessor.from_pretrained(
        model_path, 
        # min_pixels=min_pixels, 
        # max_pixels=max_pixels
    )
    generate_config = dict(
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,
        top_p=top_p,
        top_k=top_k,
    )
    with open(os.path.join(output_dir, 'qwenvl2_5_transformers_res.txt'), 'w') as f:
        for data in tqdm(eval_data, desc="Inference with Transformers"):
            image_path = data['input_image_path']
            image_id = data['image_id']
            input_text = data['input_text']
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": f'file://{image_path}',
                            "min_pixels": 256 * 28 * 28,
                            "max_pixels": 1280 * 28 * 28,
                        },
                        {
                            "type": "text", 
                            "text": "详细描述一下这张图片。",
                            #"text": f"{input_text}",
                        },
                    ]
                }
            ]

            # Preparation for inference
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)

            # Inference: Generation of the output
            generated_ids = model.generate(**inputs, **generate_config)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            #print(f'{image_id}\t{output_text}')
            f.write(f'{image_id}\t{output_text}\n')
    
def infer_with_vllm(
        model_path, 
        eval_data, 
        output_dir, 
        temperature=0.0, 
        max_new_tokens=256, 
        do_sample=False, 
        repetition_penalty=1.0, 
        top_p=1.0,
        top_k=1,
        dtype=torch.bfloat16,
    ):

    llm = LLM(
        model=model_path,
        dtype=dtype,
        task='generate',
        limit_mm_per_prompt={"image": 10, "video": 10},
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        max_tokens=max_new_tokens,
        stop_token_ids=[],
    )
    processor = AutoProcessor.from_pretrained(model_path)

    with open(os.path.join(output_dir, 'qwenvl2_5_vllm_res.txt'), 'w') as f:
        for data in tqdm(eval_data, desc="Inference with VLLM"):
            image_path = data['input_image_path']
            print(f'image_path={image_path}')
            image_id = data['image_id']
            input_text = data['input_text']
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": f'file://{image_path}',
                            "min_pixels": 256 * 28 * 28,
                            "max_pixels": 1280 * 28 * 28,
                        },
                        {
                            "type": "text", 
                            "text": "详细描述一下这张图片。",
                            #"text": f"{input_text}",
                        },
                    ]
                }
            ]

            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(messages)

            mm_data = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs
            if video_inputs is not None:
                mm_data["video"] = video_inputs

            llm_inputs = {
                "prompt": prompt,
                "do_sample": do_sample,
                "multi_modal_data": mm_data,

                # FPS will be returned in video_kwargs
                #"mm_processor_kwargs": video_kwargs,
            }

            outputs = llm.generate([llm_inputs], sampling_params=sampling_params, use_tqdm=False)
            generated_text = [outputs[0].outputs[0].text]
            #print(f'{image_id}\t{generated_text}')
            f.write(f'{image_id}\t{generated_text}\n')

def get_outputd_dir(dtype, mode, is_flash_attn_available):
    frame_str = 'ts' if mode == '0' else 'vllm'
    if dtype == torch.bfloat16:
        dtype_str = 'bf16'
    elif dtype == torch.float16:
        dtype_str = 'fp16'
    else:
        raise ValueError(f'Unsupported dtype: {dtype}')
    flash_attn_str = 'w_flash_attn' if is_flash_attn_available else 'wo_flash_attn'
    output_dir = f'outputs/{frame_str}_{dtype_str}_{flash_attn_str}'
    return output_dir

def main():
    model_path = 'open_models/Qwen/Qwen2.5-VL-3B-Instruct'
    test_json_file = 'test.json'
    
    eval_data = load_eval_data(test_json_file=test_json_file)
    #eval_data = [data for data in eval_data if '11000006396565' in data['input_image_path']]
    temperature = 0.0
    max_new_tokens = 1000
    do_sample = False
    repetition_penalty = 1.0
    top_p = 0.01
    top_k = 1
    mode = sys.argv[1]
    support_mode = {
        '0': 'transformers',
        '1': 'vllm'
    }
    dtype=torch.bfloat16
    is_flash_attn_available = _is_package_available("flash_attn")
    output_dir = get_outputd_dir(dtype, mode, is_flash_attn_available)
    os.makedirs(output_dir, exist_ok=True)
    run_batch = True
    if run_batch:
        sample_num = 100
        eval_data = eval_data[:sample_num]
        for dtype in tqdm([torch.bfloat16, torch.float16], desc='Running different dtype'):
            for mode, framework in tqdm(support_mode.items(), desc='Running different framework'):
                print(f'Running {framework} with dtype={dtype}')
                output_dir = get_outputd_dir(dtype, mode, is_flash_attn_available)
                os.makedirs(output_dir, exist_ok=True)
                func = eval(f'infer_with_{framework}')
                func(
                    model_path=model_path,
                    eval_data=eval_data,
                    output_dir=output_dir,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample, 
                    repetition_penalty=repetition_penalty,
                    top_p=top_p,
                    top_k=top_k,
                    dtype=dtype
                )
                print(f'Finished running {framework} with dtype={dtype}')


    else:
        if mode == '0':      
            infer_with_transformers(
                model_path=model_path,
                eval_data=eval_data,
                output_dir=output_dir,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample, 
                repetition_penalty=repetition_penalty,
                top_p=top_p,
                top_k=top_k,
                dtype=dtype,
            )
        else:
            infer_with_vllm(
                model_path=model_path,
                eval_data=eval_data,
                output_dir=output_dir,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample, 
                repetition_penalty=repetition_penalty,
                top_p=top_p,
                top_k=top_k,
                dtype=dtype,
            )
        
if __name__ == '__main__':
    main()
