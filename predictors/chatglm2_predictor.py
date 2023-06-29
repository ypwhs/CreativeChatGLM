import time
from typing import List, Tuple

import torch
from transformers import AutoModel, AutoTokenizer
from transformers import LogitsProcessor, LogitsProcessorList

from predictors.base import BasePredictor, parse_codeblock


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores


class ChatGLM2(BasePredictor):

    def __init__(self, model_name):
        print(f'Loading model {model_name}')
        start = time.perf_counter()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, resume_download=True)
        if 'slim' in model_name:
            model = AutoModel.from_pretrained(
                model_name, trust_remote_code=True,
                resume_download=True)
            if self.device == 'cuda':
                model = model.half().to(self.device)
            else:
                model = model.float()
        elif 'int4' in model_name:
            model = AutoModel.from_pretrained(
                model_name, trust_remote_code=True,
                resume_download=True)
            if self.device == 'cuda':
                model = model.half().to(self.device)
            else:
                model = model.float()
        else:
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                resume_download=True,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16
                if self.device == 'cuda' else torch.float32,
                device_map={'': self.device})
            if self.device == 'cpu':
                model = model.float()
        model = model.eval()
        self.model = model
        self.model_name = model_name
        end = time.perf_counter()
        print(
            f'Successfully loaded model {model_name}, time cost: {end - start:.2f}s'
        )

    @torch.no_grad()
    def stream_chat_continue(self,
                             model,
                             tokenizer, query: str, history: List[Tuple[str, str]] = None, past_key_values=None,
                             max_length: int = 8192, do_sample=True, top_p=0.8, temperature=0.8, logits_processor=None,
                             return_past_key_values=False, **kwargs):
        if history is None:
            history = []
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        if len(history) > 0:
            answer = history[-1][1]
        else:
            answer = ''
        logits_processor.append(
            InvalidScoreLogitsProcessor())
        gen_kwargs = {
            "max_length": max_length,
            "do_sample": do_sample,
            "top_p": top_p,
            "temperature": temperature,
            "logits_processor": logits_processor,
            **kwargs
        }

        if not history:
            prompt = query
        else:
            prompt = ""
            for i, (old_query, response) in enumerate(history):
                if i != len(history) - 1:
                    prompt += "[Round {}]\n\n问：{}\n\n答：{}\n\n".format(
                        i, old_query, response)
                else:
                    prompt += "[Round {}]\n\n问：{}\n\n答：\n\n".format(i, old_query)
        batch_input = tokenizer([prompt], return_tensors="pt")
        batch_input = batch_input.to(model.device)

        batch_answer = tokenizer(answer, return_tensors="pt")
        batch_answer = batch_answer.to(model.device)

        final_input_ids = torch.cat(
            [batch_input['input_ids'], batch_answer['input_ids'][:, 3:]],
            dim=-1)
        final_input_ids = final_input_ids.to(model.device)

        final_input = {}
        final_input['input_ids'] = final_input_ids
        final_input['position_ids'] = model.get_position_ids(final_input_ids, device=final_input_ids.device)
        final_input['attention_mask'] = torch.ones(final_input_ids.shape, dtype=torch.long, device=final_input_ids.device)

        for outputs in model.stream_generate(**final_input, past_key_values=past_key_values,
                                             return_past_key_values=return_past_key_values, **gen_kwargs):
            if return_past_key_values:
                outputs, past_key_values = outputs
            outputs = outputs.tolist()[0][len(batch_input["input_ids"][0]):]
            response = tokenizer.decode(outputs)
            if response and response[-1] != "�":
                response = model.process_response(response)
                yield parse_codeblock(response)


def test():
    model_name = 'chatglm2-6b'

    predictor = ChatGLM2(model_name)
    top_p = 0.01
    max_length = 128
    temperature = 0.01

    history = []
    line = '你是谁？'
    last_message = '我是张三丰，我是武当派'
    print(line)
    for x in predictor.predict_continue(
            query=line, latest_message=last_message,
            max_length=max_length, top_p=top_p, temperature=temperature,
            allow_generate=[True], history=history, last_state=[[], None, None]):
        print(x[0][-1][1])


def test2():
    from chatglm2.modeling_chatglm import ChatGLMForConditionalGeneration
    model_name = 'chatglm2-6b'
    device = 'cuda'
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, resume_download=True)
    model = ChatGLMForConditionalGeneration.from_pretrained(
        model_name,
        trust_remote_code=True,
        resume_download=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        device_map={'': device})
    model = model.eval()

    query = '继续'
    history = [('你是谁？', '我是张三丰，')]
    max_length = 128
    top_p = 0.95
    temperature = 0.8

    for response, new_history in model.stream_chat(
            tokenizer=tokenizer,
            query=query,
            history=history,
            max_length=max_length,
            top_p=top_p,
            temperature=temperature):
        print(response, new_history)


if __name__ == '__main__':
    test()
