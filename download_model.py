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
]

for model_name in model_name_list:
    if glob(f'{model_name}/*.bin'):
        print(f'{model_name} already downloaded')
        continue
    print(f'Downloading {model_name}')
    retry_times = 10
    while retry_times > 0:
        try:
            snapshot_download(
                repo_id=model_name,
                local_dir=model_name,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            break
        except:
            retry_times -= 1
            print(f'Retry download {model_name}')
    print(f'{model_name} downloaded')
