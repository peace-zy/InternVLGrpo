import json
import logging

def check_json_str_validation(json_str):
    try:
        json_data = json.loads(json_str)
        return True, 'sucess'
    except Exception as e:
        #print(f"JSON字符串不合法。错误信息: {e}")
        #print(f"错误位置: 行 {e.lineno}, 列 {e.colno}")
        #print(f"错误字符索引: {e.pos}")
        #print(f'lenth of json_string: {len(json_str)}')
        pos = e.pos - 1
        start = pos - 10 if pos - 10 >= 0 else 0
        end = pos + 10 if pos + 10 <= len(json_str) else len(json_str)
        error = f"错误字符: '{json_str[pos]}'\n错误字符段{json_str[start:end]}"
        return False, error
                

def format_reward_func(prompts, completions, **reward_kwargs):
    format_rewards = [0] * len(completions)
    for i, completion in enumerate(completions):
        try:
            format_is_valid, format_error = check_json_str_validation(completion)
            format_reward = 1 if format_is_valid else 0
            format_rewards[i] = format_reward
            if not format_is_valid:
                logging.error(f'format_error={format_error}')
                continue
        except Exception as e:
            logging.exception(f"format_reward_func error: {e}")
    return format_rewards
