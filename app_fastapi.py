from utils_env import collect_env
from fastapi import FastAPI
from sse_starlette.sse import ServerSentEvent, EventSourceResponse
# from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import argparse
import logging
import os
import json
import sys

# 加载模型
# model_name = 'THUDM/chatglm-6b'
model_name = 'THUDM/chatglm-6b-int4'

if 'chatglm' in model_name.lower():
    from predictors.chatglm_predictor import ChatGLM
    predictor = ChatGLM(model_name)
elif 'gptq' in model_name.lower():
    from predictors.llama_gptq import LLaMaGPTQ
    predictor = LLaMaGPTQ(model_name)
elif 'llama' in model_name.lower():
    from predictors.llama import LLaMa
    predictor = LLaMa(model_name)
elif 'debug' in model_name.lower():
    from predictors.debug import Debug
    predictor = Debug(model_name)
else:
    from predictors.chatglm_predictor import ChatGLM
    predictor = ChatGLM(model_name)


# 接入log
def getLogger(name, file_name, use_formatter=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s    %(message)s')
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    if file_name:
        handler = logging.FileHandler(file_name, encoding='utf8')
        handler.setLevel(logging.INFO)
        if use_formatter:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(message)s')
            handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = getLogger('ChatGLM', 'chatlog.log')

# 超参数 用于控制模型回复时 上文的长度
MAX_HISTORY = 5


# 接入FastAPI
def start_server(quantize_level, http_address: str, port: int, gpu_id: str):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

    bot = predictor

    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"])

    @app.get("/")
    def index():
        return {'message': 'started', 'success': True}

    @app.post("/stream")
    def continue_question_stream(arg_dict: dict):

        def decorate(generator):
            for item in generator:
                yield ServerSentEvent(
                    json.dumps(item, ensure_ascii=False), event='delta')

        # inputs = [query, creative_factor, max_length, top_p, temperature, allow_generate, history]
        try:
            query = arg_dict["query"]
            creative_factor = arg_dict.get("creative_factor", "")
            max_length = arg_dict.get("max_length", 256)
            top_p = arg_dict.get("top_p", 0.7)
            temperature = arg_dict.get("temperature", 1.0)
            allow_generate = arg_dict.get("allow_generate", [True])
            history = arg_dict.get("history", [])
            logger.info("Query - {}".format(query))
            if len(history) > 0:
                logger.info("History - {}".format(history))
            history = history[-MAX_HISTORY:]
            history = [tuple(h) for h in history]
            inputs = [
                query, creative_factor, max_length, top_p, temperature,
                allow_generate, history
            ]
            return EventSourceResponse(decorate(bot.predict_continue(*inputs)))
            # return EventSourceResponse(bot.predict_continue(*inputs))
        except Exception as e:
            logger.error(f"error: {e}")
            return EventSourceResponse(
                decorate(bot.predict_continue(None, None)))

    logger.info("starting server...")
    # uvicorn.run(app=app, host=http_address, port=port, debug=False)
    uvicorn.run(app=app, host=http_address, port=port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Stream API Service for ChatGLM-6B')
    parser.add_argument(
        '--device',
        '-d',
        help='device，-1 means cpu, other means gpu ids',
        default='0')
    parser.add_argument(
        '--quantize',
        '-q',
        help='level of quantize, option：16, 8 or 4',
        default=16)
    parser.add_argument(
        '--host', '-H', help='host to listen', default='0.0.0.0')
    parser.add_argument(
        '--port', '-P', help='port of this service', default=8000)
    args = parser.parse_args()
    start_server(args.quantize, args.host, int(args.port), args.device)
    # print(f"Server started at {args.host}:{args.port}")
