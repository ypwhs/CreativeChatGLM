import torch
from typing import List, Tuple
from transformers import LogitsProcessor, LogitsProcessorList


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 20005] = 5e4
        return scores


@torch.no_grad()
def stream_chat_continue(
        self, tokenizer, query: str, history: List[Tuple[str, str]] = None,
        max_length: int = 2048, do_sample=True, top_p=0.7, temperature=0.95, logits_processor=None, **kwargs):
    if history is None:
        history = []
    if logits_processor is None:
        logits_processor = LogitsProcessorList()
    if len(history) > 0:
        answer = history[-1][1]
    else:
        answer = ''
    logits_processor.append(InvalidScoreLogitsProcessor())
    gen_kwargs = {"max_length": max_length, "do_sample": do_sample, "top_p": top_p,
                  "temperature": temperature, "logits_processor": logits_processor, **kwargs}
    if not history:
        prompt = query
    else:
        prompt = ""
        for i, (old_query, response) in enumerate(history):
            if i != len(history) - 1:
                prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
            else:
                prompt += "[Round {}]\n问：{}\n答：".format(i, old_query)
    batch_input = tokenizer([prompt], return_tensors="pt", padding=True)
    batch_input = batch_input.to(self.device)

    batch_answer = tokenizer(answer, return_tensors="pt")
    batch_answer = batch_answer.to(self.device)

    input_length = len(batch_input['input_ids'][0])
    final_input_ids = torch.cat([batch_input['input_ids'], batch_answer['input_ids'][:, :-2]], dim=-1).cuda()
    attention_mask = torch.ones_like(final_input_ids).bool().cuda()
    attention_mask[:, input_length:] = False

    batch_input['input_ids'] = final_input_ids
    batch_input['attention_mask'] = attention_mask

    for outputs in self.stream_generate(**batch_input, **gen_kwargs):
        outputs = outputs.tolist()[0][input_length:]
        response = tokenizer.decode(outputs)
        response = self.process_response(response)
        new_history = history + [(query, response)]
        yield response, new_history
