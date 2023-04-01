import time
from typing import List, Tuple

import torch
from transformers import AutoModel, AutoTokenizer
from transformers import LogitsProcessor, LogitsProcessorList

from predictors.base import BasePredictor
from chatglm.modeling_chatglm import ChatGLMForConditionalGeneration


class InvalidScoreLogitsProcessor(LogitsProcessor):

    def __call__(self, input_ids: torch.LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 20005] = 5e4
        return scores


class ChatGLM(BasePredictor):

    def __init__(self, model_name):
        print(f'Loading model {model_name}')
        start = time.perf_counter()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, resume_download=True)
        if 'int4' not in model_name:
            model = ChatGLMForConditionalGeneration.from_pretrained(
                model_name,
                trust_remote_code=True,
                resume_download=True,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                device_map={'': self.device}
            )
        else:
            model = ChatGLMForConditionalGeneration.from_pretrained(
                model_name,
                trust_remote_code=True,
                resume_download=True
            ).half().to(self.device)
        model = model.eval()
        self.model = model
        end = time.perf_counter()
        print(f'Successfully loaded model {model_name}, time cost: {end - start:.2f}s')

    @torch.no_grad()
    def stream_chat_continue(self,
                             model,
                             tokenizer,
                             query: str,
                             history: List[Tuple[str, str]] = None,
                             max_length: int = 2048,
                             do_sample=True,
                             top_p=0.7,
                             temperature=0.95,
                             logits_processor=None,
                             **kwargs):
        if history is None:
            history = []
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        if len(history) > 0:
            answer = history[-1][1]
        else:
            answer = ''
        logits_processor.append(InvalidScoreLogitsProcessor())
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
                    prompt += "[Round {}]\n问：{}\n答：{}\n".format(
                        i, old_query, response)
                else:
                    prompt += "[Round {}]\n问：{}\n答：".format(i, old_query)
        batch_input = tokenizer([prompt], return_tensors="pt", padding=True)
        batch_input = batch_input.to(model.device)

        batch_answer = tokenizer(answer, return_tensors="pt")
        batch_answer = batch_answer.to(model.device)

        input_length = len(batch_input['input_ids'][0])
        final_input_ids = torch.cat(
            [batch_input['input_ids'], batch_answer['input_ids'][:, :-2]],
            dim=-1).cuda()
        attention_mask = torch.ones_like(final_input_ids).bool().to(model.device)
        attention_mask[:, input_length:] = False

        batch_input['input_ids'] = final_input_ids
        batch_input['attention_mask'] = attention_mask

        for outputs in model.stream_generate(**batch_input, **gen_kwargs):
            outputs = outputs.tolist()[0][input_length:]
            response = tokenizer.decode(outputs)
            response = model.process_response(response)
            yield response
