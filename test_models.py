def test_model(model_name):
    if 'chatglm2' in model_name.lower():
        from predictors.chatglm2_predictor import ChatGLM2
        predictor = ChatGLM2(model_name)
    elif 'chatglm' in model_name.lower():
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


def main():
    model_list = [
        'THUDM/chatglm2-6b',
        'THUDM/chatglm2-6b-int4',
        'THUDM/chatglm3-6b',
    ]
    for model_name in model_list:
        print(f'Testing {model_name}')
        test_model(model_name)


if __name__ == '__main__':
    main()
