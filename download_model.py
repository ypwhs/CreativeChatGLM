from glob import glob
from huggingface_hub import snapshot_download

model_name_list = [
    'THUDM/chatglm-6b-int4-qe',
    'THUDM/chatglm-6b-int4',

    'silver/chatglm-6b-int4-qe-slim',
    'BelleGroup/BELLE-LLAMA-7B-2M-gptq',
    'BelleGroup/BELLE-7B-gptq',

    'THUDM/chatglm-6b',
    'THUDM/glm-10b-chinese',
    'BelleGroup/BELLE-LLAMA-7B-2M',
    'BelleGroup/BELLE-7B-2M',
    'silver/chatglm-6b-slim',
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
                resume_download=True,
            )
            snapshot_download(
                repo_id=model_name,
                local_dir=dst_path,
                local_dir_use_symlinks=False,
            )
            break
        except:
            retry_times -= 1
            print(f'Retry download {model_name}, {retry_times} times left...')
    print(f'{model_name} downloaded')
