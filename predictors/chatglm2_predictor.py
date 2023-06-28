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

        if past_key_values is None and not return_past_key_values:
            batch_input = model.build_inputs(tokenizer, query, history=history)
        else:
            batch_input = model.build_stream_inputs(tokenizer, query, history=history)

        batch_answer = tokenizer(answer, return_tensors="pt")
        batch_answer = batch_answer.to(model.device)

        input_length = len(batch_input['input_ids'][0])
        final_input_ids = torch.cat(
            [batch_input['input_ids'], batch_answer['input_ids'][:, :-2]],
            dim=-1)
        final_input_ids = final_input_ids.to(model.device)

        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[0]
            final_input_ids.position_ids += past_length
            attention_mask = final_input_ids.attention_mask
            attention_mask = torch.cat((attention_mask.new_ones(1, past_length), attention_mask), dim=1)
            final_input_ids['attention_mask'] = attention_mask

        batch_input['input_ids'] = final_input_ids

        input_ids = final_input_ids
        # MASK, gMASK = self.model.config.bos_token_id - 4, self.model.config.bos_token_id - 3
        # mask_token = MASK if MASK in input_ids else gMASK
        # mask_positions = [seq.tolist().index(mask_token) for seq in input_ids]
        # batch_input['position_ids'] = self.model.get_position_ids(
        #     input_ids, mask_positions, device=input_ids.device)

        for outputs in model.stream_generate(**batch_input, past_key_values=past_key_values,
                                            return_past_key_values=return_past_key_values, **gen_kwargs):
            if return_past_key_values:
                outputs, past_key_values = outputs
            outputs = outputs.tolist()[0][len(batch_input["input_ids"][0]):]
            response = tokenizer.decode(outputs)
            if response and response[-1] != "�":
                response = model.process_response(response)
                new_history = history + [(query, response)]
                yield parse_codeblock(response)


def test():
    model_name = 'THUDM/chatglm2-6b'

    predictor = ChatGLM2(model_name)
    top_p = 0.95
    max_length = 128
    temperature = 0.8

    line = '你是谁？'
    last_message = '我是张三丰，'
    print(line)
    for x in predictor.predict_continue(
            query=line, latest_message=last_message,
            max_length=max_length, top_p=top_p, temperature=temperature,
            allow_generate=[True], history=None, last_state=[[], None, None]):
        print(x[0][-1][1])


if __name__ == '__main__':
    test()
