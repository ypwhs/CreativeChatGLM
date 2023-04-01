import time
import torch
import transformers
from predictors.llama import LLaMa
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, LlamaForCausalLM
from gptq.llama_inference import load_quant
from transformers.utils.hub import cached_file


class LLaMaGPTQ(LLaMa):
    def __init__(self, model_name, checkpoint_path='llama7b-2m-4bit-128g.pt', wbits=4, groupsize=128):
        print(f'Loading model {model_name}')
        start = time.perf_counter()
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, resume_download=True)
        checkpoint_path = cached_file(model_name, checkpoint_path)
        print(f'Loading model from {checkpoint_path} ...')
        model: LlamaForCausalLM = load_quant(model_name, checkpoint_path, wbits, groupsize)
        model.eval()
        self.model = model
        end = time.perf_counter()
        print(f'Successfully loaded model {model_name}, time cost: {end - start:.2f}s')


def test():
    model_name = 'BelleGroup/BELLE-LLAMA-7B-2M-gptq'
    checkpoint_path = 'llama7b-2m-4bit-128g.pt'
    wbits = 4
    groupsize = 128

    predictor = LLaMaGPTQ(model_name, checkpoint_path, wbits, groupsize)
    device = predictor.device
    tokenizer = predictor.tokenizer
    model = predictor.model
    min_length = 10
    max_length = 2048
    top_p = 0.95
    temperature = 0.8

    print("Human:")
    line = input()
    inputs = 'Human: ' + line.strip() + '\n\nAssistant:'
    input_ids = tokenizer.encode(inputs, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            do_sample=True,
            min_length=min_length,
            max_length=max_length,
            top_p=top_p,
            temperature=temperature,
        )
    print("Assistant:\n【")
    print(tokenizer.decode([el.item() for el in generated_ids[0]]))
    print("】\n-------------------------------\n")

    for x in predictor.predict_continue(
            line, '', max_length, top_p, temperature, [True], None):
        print("Assistant:\n【")
        print(x[0][-1][1])
        print("】\n-------------------------------\n")


if __name__ == '__main__':
    test()
