import os
from types import new_class

from check_train_image import check_train_image
from check_text_validation import check_text_validation
from get_text_token_length import get_text_token_length
from del_long_text_token import del_long_text_token

def check(json_file, out_json_file, output_path, use_new_model=False):
    print('\033[32m[check_train_image]\033[0m start...')
    #check_train_image(json_file, out_json_file)
    print('\033[32m[check_train_image]\033[0m Done')

    print('\033[32m[check_text_validation]\033[0m start...')
    check_text_validation(json_file, out_json_file)
    print('\033[32m[check_text_validation]\033[0m Done')

    print('\033[32m[get_text_token_length]\033[0m start...')
    text_token_length_output_path = f'{output_path}/text_token_length_output'
    process_num = 30
    thread_num = 20


    if not use_new_model:
        # original model
        model_name_or_path = 'Model/OpenGVLab/InternVL2-8B'
    else:
        # new model
        model_name_or_path = 'Model/OpenGVLab/InternVL2-8B-add-token'

    get_text_token_length(
        json_file=out_json_file,
        model_name_or_path=model_name_or_path,
        process_num=process_num,
        thread_num=thread_num,
        output_path=text_token_length_output_path
    )
    print('\033[32m[get_text_token_length]\033[0m Done')

    print('\033[32m[del_long_text_token]\033[0m start...')
    long_text_token_output_path = f'{output_path}/long_text_token_output'
    len_thres = 14400

    del_long_text_token(
        json_file=out_json_file,
        token_info_path=text_token_length_output_path,
        output_path=long_text_token_output_path,
        len_thres=len_thres
    )
    print('\033[32m[del_long_text_token]\033[0m Done')
    return

def main():
    version = 'version3'
    fname = f'r2v_sft_{version}_with_space.json'
    json_file = f'dataset/{fname}'
    dirname, filename = os.path.split(json_file)
    dirname = 'Internvl2_20241230/InternVL/dataset/r2v'
    output_path = f'{dirname}/{version}'
    use_new_model = True

    os.makedirs(output_path, exist_ok=True)
    out_json_file = f'{dirname}/filtered_{filename}'
    print(f'input json file: {json_file}, output json file: {out_json_file}, output path: {output_path}')
    check(json_file, out_json_file, output_path, use_new_model=use_new_model)
    print('\033[32m[check]\033[0m Done')
    print(f'input json file: {json_file}, output json file: {out_json_file}, output path: {output_path}')

if __name__ == '__main__':
    main()
