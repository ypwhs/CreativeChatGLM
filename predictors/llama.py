from transformers import LlamaForCausalLM, AutoTokenizer
import torch
from typing import List, Tuple
from predictors.base import BasePredictor


class LLaMa(BasePredictor):

    def __init__(self, model_name):
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, resume_download=True)
        self.model = LlamaForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            resume_download=True,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            device_map={'': self.device})
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def stream_chat_continue(self,
                             model,
                             tokenizer,
                             query: str,
                             history: List[Tuple[str, str]] = None,
                             max_new_tokens=500,
                             do_sample=True,
                             top_k=30,
                             top_p=0.85,
                             temperature=0.5,
                             repetition_penalty=1.,
                             eos_token_id=2,
                             bos_token_id=1,
                             pad_token_id=0,
                             **kwargs):
        if history is None:
            history = []
        if len(history) > 0:
            answer = history[-1][1]
        else:
            answer = ''
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "top_p": top_p,
            "top_k": top_k,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "eos_token_id": eos_token_id,
            "bos_token_id": bos_token_id,
            "pad_token_id": pad_token_id,
            **kwargs
        }
        if not history:
            prompt = f'Human: {query} \n\nAssistant:'
        else:
            prompt = ""
            for i, (old_query, response) in enumerate(history):
                if i != len(history) - 1:
                    prompt += f'Human: {old_query} \n\nAssistant:{response} \n\n'
                else:
                    prompt += f'Human: {old_query} \n\nAssistant:'
        batch_input = tokenizer([prompt], return_tensors="pt", padding=True)
        batch_input = batch_input.to(model.device)

        batch_answer = tokenizer(answer, return_tensors="pt")
        batch_answer = batch_answer.to(model.device)

        input_length = len(batch_input['input_ids'][0])
        final_input_ids = torch.cat(
            [batch_input['input_ids'], batch_answer['input_ids'][:, :-2]],
            dim=-1).cuda()
        attention_mask = torch.ones_like(final_input_ids).bool().to(
            model.device)
        attention_mask[:, input_length:] = False

        batch_input['input_ids'] = final_input_ids
        batch_input['attention_mask'] = attention_mask

        for outputs in model.stream_generate(**batch_input, **gen_kwargs):
            outputs = outputs.tolist()[0][input_length:]
            response = tokenizer.decode(outputs)
            response = model.process_response(response)
            new_history = history + [(query, response)]
            yield response, new_history


def test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = 'BelleGroup/BELLE-LLAMA-7B-2M'
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, resume_download=True)
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        resume_download=True,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        device_map={'': device})
    min_length = 10
    max_length = 2048
    top_p = 0.95
    temperature = 0.8

    print("Human:")
    line = input()
    while line:
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
        line = input()


if __name__ == '__main__':
    test()
