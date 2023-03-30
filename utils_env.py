def collect_env():
    import sys
    from collections import defaultdict

    env_info = {}
    env_info['sys.platform'] = sys.platform
    env_info['Python'] = sys.version.replace('\n', '')
    env_info['Python executable'] = sys.executable

    import torch
    env_info['PyTorch'] = torch.__version__

    import gradio
    env_info['Gradio'] = gradio.__version__

    import transformers
    env_info['Transformers'] = transformers.__version__

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            devices[torch.cuda.get_device_name(k)].append(str(k))
        for name, device_ids in devices.items():
            env_info['GPU ' + ','.join(device_ids)] = name

    return env_info


if __name__ == '__main__':
    for name, val in collect_env().items():
        print(f'{name}: {val}')
