from glob import glob
from huggingface_hub import snapshot_download

model_name_list = [
    'THUDM/chatglm-6b',
    'THUDM/chatglm-6b-int4',
    'THUDM/chatglm-6b-int4-qe',
    'THUDM/glm-10b-chinese',
]

for model_name in model_name_list:
    if glob(f'{model_name}/*.bin'):
        print(f'{model_name} already downloaded')
        continue
    print(f'Downloading {model_name}')
    snapshot_download(
        repo_id=model_name,
        local_dir=model_name,
        local_dir_use_symlinks=False)
    print(f'{model_name} downloaded')
