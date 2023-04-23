import copy
import time
import warnings
from typing import List, Tuple, Optional, Callable

import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, AutoTokenizer
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList, GenerationConfig
from transformers.utils import logging

from predictors.base import BasePredictor

logger = logging.get_logger(__name__)


@torch.no_grad()
def stream_generate(
        self,
        input_ids,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        **kwargs,
):
    batch_size, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]

    if generation_config is None:
        generation_config = self.generation_config
    generation_config = copy.deepcopy(generation_config)
    model_kwargs = generation_config.update(**kwargs)
    bos_token_id, eos_token_id = generation_config.bos_token_id, generation_config.eos_token_id

    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]

    has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
    if has_default_max_length and generation_config.max_new_tokens is None:
        warnings.warn(
            f"Using `max_length`'s default ({generation_config.max_length}) to control the generation length. "
            "This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we"
            " recommend using `max_new_tokens` to control the maximum length of the generation.",
            UserWarning,
        )
    elif generation_config.max_new_tokens is not None:
        generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length
        if not has_default_max_length:
            logger.warn(
                f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                "Please refer to the documentation for more information. "
                "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)",
                UserWarning,
            )

    if input_ids_seq_length >= generation_config.max_length:
        input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
        logger.warning(
            f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
            f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
            " increasing `max_new_tokens`."
        )

    # 2. Set generation parameters if not already defined
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    logits_processor = self._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_seq_length,
        encoder_input_ids=input_ids,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
    )

    stopping_criteria = self._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria
    )
    logits_warper = self._get_logits_warper(generation_config)

    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
    scores = None
    while True:
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        # forward pass to get next token
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )

        next_token_logits = outputs.logits[:, -1, :]

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)
        next_token_scores = logits_warper(input_ids, next_token_scores)

        # sample
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        if generation_config.do_sample:
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(probs, dim=-1)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
        unfinished_sequences = unfinished_sequences.mul((sum(next_tokens != i for i in eos_token_id)).long())

        # stop when each sentence is finished, or if we exceed the maximum length
        if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
            break
        yield input_ids


class LLaMa(BasePredictor):

    def __init__(self, model_name):
        print(f'Loading model {model_name}')
        start = time.perf_counter()
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
        self.model.eval()
        end = time.perf_counter()
        print(f'Successfully loaded model {model_name}, time cost: {end - start:.2f}s')

    @torch.no_grad()
    def stream_chat_continue(self,
                             model,
                             tokenizer,
                             query: str,
                             history: List[Tuple[str, str]] = None,
                             max_length=500,
                             do_sample=True,
                             top_p=0.85,
                             temperature=0.5,
                             **kwargs):
        if history is None:
            history = []
        if len(history) > 0:
            answer = history[-1][1]
        else:
            answer = ''
        gen_kwargs = {
            "max_length": max_length,
            "do_sample": do_sample,
            "top_p": top_p,
            "temperature": temperature,
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
        batch_input = tokenizer([prompt], return_tensors="pt")
        batch_input = batch_input.to(model.device)

        batch_answer = tokenizer(answer, return_tensors="pt")
        batch_answer = batch_answer.to(model.device)

        input_length = len(batch_input['input_ids'][0])
        final_input_ids = torch.cat(
            [batch_input['input_ids'], batch_answer['input_ids'][:, :-2]],
            dim=-1)
        final_input_ids = final_input_ids.to(model.device)
        attention_mask = torch.ones_like(final_input_ids).bool().to(
            model.device)
        attention_mask[:, input_length:] = False

        batch_input['input_ids'] = final_input_ids
        batch_input['attention_mask'] = attention_mask

        for outputs in stream_generate(model, **batch_input, **gen_kwargs):
            outputs = outputs.tolist()[0][input_length:]
            response = tokenizer.decode(outputs)
            yield response


def test():
    model_name = 'BelleGroup/BELLE-LLAMA-7B-2M'

    predictor = LLaMa(model_name)
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
