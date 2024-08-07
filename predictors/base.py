import copy
from abc import ABC, abstractmethod


def parse_codeblock(text):
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if "```" in line:
            if line != "```":
                lines[i] = f'<pre><code class="{lines[i][3:]}">'
            else:
                lines[i] = '</code></pre>'
        else:
            if i > 0:
                lines[i] = "<br/>" + line.replace("<", "&lt;").replace(
                    ">", "&gt;")
    return "".join(lines)


class BasePredictor(ABC):

    @abstractmethod
    def __init__(self, model_name, predict_mode='tuple'):
        self.model = None
        self.tokenizer = None
        self.model_name = model_name
        self.predict_mode = predict_mode

    @abstractmethod
    def stream_chat_continue(self, *args, **kwargs):
        raise NotImplementedError

    def predict_continue(self, *args, **kwargs):
        if self.predict_mode == 'tuple':
            yield from self.predict_continue_tuple(*args, **kwargs)
        else:
            yield from self.predict_continue_dict(*args, **kwargs)

    def predict_continue_tuple(self, query, latest_message, max_length, top_p,
                               temperature, allow_generate, history,
                               last_state, *args, **kwargs):
        last_state[0] = copy.deepcopy(history)
        last_state[1] = query
        last_state[2] = latest_message
        if history is None:
            history = []
        allow_generate[0] = True
        history.append((query, latest_message))
        for response in self.stream_chat_continue(
                self.model,
                self.tokenizer,
                query=query,
                history=history,
                max_length=max_length,
                top_p=top_p,
                temperature=temperature):
            history[-1] = (history[-1][0], response)
            history_colorful = copy.deepcopy(history)
            colorful_response = f'<span style="color:red">{latest_message}</span>{response[len(latest_message):]}'
            history_colorful[-1] = (history_colorful[-1][0], colorful_response)
            yield history_colorful, '', ''
            if not allow_generate[0]:
                break

    def predict_continue_dict(self, query, latest_message, max_length, top_p,
                              temperature, allow_generate, history, last_state,
                              *args, **kwargs):
        last_state[0] = copy.deepcopy(history)
        last_state[1] = query
        last_state[2] = latest_message
        if history is None:
            history = []
        allow_generate[0] = True
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": latest_message})
        for response in self.stream_chat_continue(
                self.model,
                self.tokenizer,
                query=query,
                history=history,
                max_length=max_length,
                top_p=top_p,
                temperature=temperature):
            history[-1]["content"] = response
            history_colorful = copy.deepcopy(history)
            colorful_response = f'<span style="color:red">{latest_message}</span>{response[len(latest_message):]}'
            history_colorful[-1]["content"] = colorful_response
            history_tuple = []
            for i in range(0, len(history_colorful), 2):
                history_tuple.append((history_colorful[i]["content"],
                                      history_colorful[i + 1]["content"]))
            yield history_tuple, '', ''
            if not allow_generate[0]:
                break
