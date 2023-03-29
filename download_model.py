from glob import glob
from huggingface_hub import snapshot_download

model_name_list = [
    'THUDM/chatglm-6b',
    'THUDM/chatglm-6b-int4',
    'THUDM/chatglm-6b-int4-qe',
    'THUDM/glm-10b-chinese',
    'BelleGroup/BELLE-7B-2M',
    'BelleGroup/BELLE-7B-gptq',
    'BelleGroup/BELLE-LLAMA-7B-2M',
    'BelleGroup/BELLE-LLAMA-7B-2M-gptq',
    'silver/chatglm-6b-slim',
]

for model_name in model_name_list:
    if glob(f'{model_name}/*.bin') or glob(f'{model_name}/*.pt'):
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
                local_dir=model_name,
                local_dir_use_symlinks=False,
            )
            break
        except:
            retry_times -= 1
            print(f'Retry download {model_name}, {retry_times} times left...')
    print(f'{model_name} downloaded')
