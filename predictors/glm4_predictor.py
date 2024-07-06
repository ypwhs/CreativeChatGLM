import time
import json
from typing import List, Dict

import torch
from transformers import AutoModel, AutoTokenizer
from transformers import LogitsProcessor, LogitsProcessorList
from transformers import BitsAndBytesConfig

from predictors.base import BasePredictor, parse_codeblock


class InvalidScoreLogitsProcessor(LogitsProcessor):

    def __call__(self, input_ids: torch.LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores


class GLM4(BasePredictor):

    def __init__(self, model_name, int4=False):
        self.predict_mode = 'dict'
        print(f'Loading model {model_name}')
        start = time.perf_counter()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True)
        if 'slim' in model_name:
            model = AutoModel.from_pretrained(
                model_name, trust_remote_code=True)
            if self.device == 'cuda':
                model = model.half().to(self.device)
            else:
                model = model.float()
        elif 'int4' in model_name:
            model = AutoModel.from_pretrained(
                model_name, trust_remote_code=True)
            if self.device == 'cuda':
                model = model.half().to(self.device)
            else:
                model = model.float()
        else:
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16
                if self.device == 'cuda' else torch.float32,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True) if int4 else None,
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

    @torch.inference_mode()
    def stream_chat_continue(self,
                             model,
                             tokenizer,
                             query: str,
                             history: List[Dict] = None,
                             role: str = "user",
                             past_key_values=None,
                             max_length: int = 8192,
                             do_sample=True,
                             top_p=0.8,
                             temperature=0.8,
                             logits_processor=None,
                             return_past_key_values=False,
                             **kwargs):
        if history is None:
            history = []
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())

        eos_token_id = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|user|>"),
            tokenizer.convert_tokens_to_ids("<|observation|>")
        ]

        gen_kwargs = {
            "max_length": max_length,
            "do_sample": do_sample,
            "top_p": top_p,
            "temperature": temperature,
            "logits_processor": logits_processor,
            **kwargs
        }

        answer = history[-1]["content"]

        input_ids = []
        for item in history[:-1]:
            content = item["content"]
            if item["role"] == "system" and "tools" in item:
                content = content + "\n" + json.dumps(
                    item["tools"], indent=4, ensure_ascii=False)
            input_ids.extend(
                tokenizer.build_single_message(item["role"],
                                               item.get("metadata", ""),
                                               content))
        batch_input = tokenizer.batch_encode_plus([input_ids],
                                                  return_tensors="pt",
                                                  is_split_into_words=True)
        batch_input = batch_input.to(model.device)

        answer_input_ids = tokenizer.build_single_message(
            "assistant", "", answer)
        batch_answer = tokenizer.batch_encode_plus([answer_input_ids],
                                                   return_tensors="pt",
                                                   is_split_into_words=True)
        batch_answer = batch_answer.to(model.device)

        final_input_ids = torch.cat(
            [batch_input['input_ids'], batch_answer['input_ids'][:, 2:]],
            dim=-1)
        final_input_ids = final_input_ids.to(model.device)

        final_input = {}
        final_input['input_ids'] = final_input_ids
        final_input['position_ids'] = model.get_position_ids(
            final_input_ids, device=final_input_ids.device)
        final_input['attention_mask'] = torch.ones(
            final_input_ids.shape,
            dtype=torch.long,
            device=final_input_ids.device)

        for outputs in model.stream_generate(
                **final_input,
                past_key_values=past_key_values,
                eos_token_id=eos_token_id,
                return_past_key_values=return_past_key_values,
                **gen_kwargs):
            if return_past_key_values:
                outputs, past_key_values = outputs
            outputs = outputs.tolist()[0][
                len(batch_input["input_ids"]
                    [0]):-1]  # Exclude the last token if it's EOS
            response = tokenizer.decode(outputs)
            if response and response[-1] != "�":
                response, new_history = model.process_response(
                    response, history)
                yield response


def test():
    model_name = 'THUDM/glm-4-9b-chat-1m'

    predictor = GLM4(model_name)
    top_p = 0.01
    max_length = 128
    temperature = 0.01

    history = []
    query = '你是谁？'
    last_message = '我是张三丰，我是武当派'

    print(query)
    for x in predictor.predict_continue_dict(
            query=query,
            latest_message=last_message,
            max_length=max_length,
            top_p=top_p,
            temperature=temperature,
            allow_generate=[True],
            history=history,
            last_state=[[], None, None]):
        print(x[0][-1])


def test2():
    from glm4.modeling_chatglm import ChatGLMForConditionalGeneration
    model_name = 'THUDM/glm-4-9b-chat-1m'
    device = 'cuda'
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True)
    model = ChatGLMForConditionalGeneration.from_pretrained(
        model_name,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16 if device == 'cuda' else torch.float32,
        device_map={'': device})
    model = model.eval()

    query = '继续'
    history = [{
        'role': 'user',
        'content': '你是谁？'
    }, {
        'role': 'assistant',
        'content': '我是张三丰，我是武当派'
    }]
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
