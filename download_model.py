import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import traceback
from glob import glob
from huggingface_hub import snapshot_download

model_name_list = [
    # 'THUDM/chatglm-6b-int4-qe',
    # 'THUDM/chatglm-6b-int4',
    # 'THUDM/chatglm-6b',
    # 'THUDM/glm-10b-chinese',
    #
    # 'THUDM/chatglm2-6b',
    # 'THUDM/chatglm2-6b-int4',
    #
    # 'THUDM/chatglm3-6b',
    # 'THUDM/chatglm3-6b-128k',

    'THUDM/glm-4-9b-chat-1m',

    # 'silver/chatglm-6b-slim',
    # 'silver/chatglm-6b-int4-slim',
    # 'silver/chatglm-6b-int4-qe-slim',
]

for model_name in model_name_list:
    dst_path = f'models/{model_name}'
    if glob(f'{dst_path}/*.bin') or glob(f'{dst_path}/*.pt'):
        print(f'{model_name} already downloaded')
        continue
    retry_times = 10
    while retry_times > 0:
        try:
            print(f'Downloading {model_name}')
            snapshot_download(
                repo_id=model_name,
                max_workers=2,
                # proxies={'https': 'http://127.0.0.1:7890'}
            )
            snapshot_download(
                repo_id=model_name,
                local_dir=dst_path,
                local_dir_use_symlinks=False,
            )
            break
        except:
            traceback.print_exc()
            retry_times -= 1
            print(f'Retry download {model_name}, {retry_times} times left...')
    print(f'{model_name} downloaded')
