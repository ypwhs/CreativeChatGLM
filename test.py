url = "http://localhost:8000/stream"

params = {"query": "Hello" ,'blank_input':"哇你会说","allow_generate": [True],
          'history':[('你好啊','你在和我套近乎吗?'),("美女别走啊","我不喜欢不会说英语的人"),('我会说英语哦','那如果你会说的话 我可能会喜欢你哦')]}

import requests
from requests.exceptions import RequestException


def event_source_response_iterator(response):
    buf = []
    for chunk in response.iter_content(None):
        if not chunk:
            break
        buf.extend(chunk.split(b"\n"))
        while buf:
            line = buf.pop(0).strip()
            if line:
                try:
                    event, data = line.split(b":", 1)
                    if event.startswith(b"id"):
                        continue
                    if event.strip() == b"data":
                        yield data.strip()
                except ValueError:
                    pass

try:
    response = requests.post(
        url,
        json=params
    )
    response.raise_for_status()
    for data in event_source_response_iterator(response):
        print(data.decode())
except RequestException as e:
    print(e)
