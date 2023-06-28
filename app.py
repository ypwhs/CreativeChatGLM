import gradio as gr
from utils_env import collect_env

# 收集环境信息
print('Collect environment info'.center(64, '-'))
for name, val in collect_env().items():
    print(f'{name}: {val}')
print('Done'.center(64, '-'))

# 加载模型
# model_name = 'THUDM/chatglm-6b'
# model_name = 'silver/chatglm-6b-int4-slim'
model_name = 'THUDM/chatglm2-6b-int4'

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


def revise(history, latest_message):
    history[-1] = (history[-1][0], latest_message)
    return history, ''


def revoke(history, last_state):
    if len(history) >= 1:
        history.pop()
    last_state[0] = history
    last_state[1] = ''
    last_state[2] = ''
    return history


def interrupt(allow_generate):
    allow_generate[0] = False


def regenerate(last_state, max_length, top_p, temperature, allow_generate):
    history, query, continue_message = last_state
    if len(query) == 0:
        print("Please input a query first.")
        return
    for x in predictor.predict_continue(query, continue_message, max_length, top_p,
                                        temperature, allow_generate, history, last_state):
        yield x


# 搭建 UI 界面
with gr.Blocks(css=""".message {
    width: inherit !important;
    padding-left: 20px !important;
}""") as demo:
    gr.Markdown(
        f"""
# 💡Creative ChatGLM WebUI

👋 欢迎来到 ChatGLM 创意世界！[https://github.com/ypwhs/CreativeChatGLM](https://github.com/ypwhs/CreativeChatGLM)

当前模型：{model_name}

* 📖 你可以使用“续写”按钮帮 ChatGLM 想一个开头，并让它继续生成更多的内容。
* 📝 你可以使用“修订”按钮修改最后一句 ChatGLM 的回复。
""")
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(elem_id="chat-box", show_label=False).style(height=850)
        with gr.Column(scale=1):
            with gr.Row():
                max_length = gr.Slider(32, 4096, value=2048, step=1.0, label="Maximum length", interactive=True)
                top_p = gr.Slider(0.01, 1, value=0.7, step=0.01, label="Top P", interactive=True)
                temperature = gr.Slider(0.01, 5, value=0.95, step=0.01, label="Temperature", interactive=True)
            with gr.Row():
                query = gr.Textbox(show_label=False, placeholder="Prompts", lines=4).style(container=False)
                generate_button = gr.Button("生成")
            with gr.Row():
                continue_message = gr.Textbox(
                    show_label=False, placeholder="Continue message", lines=2).style(container=False)
                continue_btn = gr.Button("续写")
                revise_message = gr.Textbox(
                    show_label=False, placeholder="Revise message", lines=2).style(container=False)
                revise_btn = gr.Button("修订")
                revoke_btn = gr.Button("撤回")
                regenerate_btn = gr.Button("重新生成")
                interrupt_btn = gr.Button("终止生成")

    history = gr.State([])
    allow_generate = gr.State([True])
    blank_input = gr.State("")
    last_state = gr.State([[], '', ''])  # history, query, continue_message
    generate_button.click(
        predictor.predict_continue,
        inputs=[query, blank_input, max_length, top_p, temperature, allow_generate, history, last_state],
        outputs=[chatbot, query])
    revise_btn.click(revise, inputs=[history, revise_message], outputs=[chatbot, revise_message])
    revoke_btn.click(revoke, inputs=[history, last_state], outputs=[chatbot])
    continue_btn.click(
        predictor.predict_continue,
        inputs=[query, continue_message, max_length, top_p, temperature, allow_generate, history, last_state],
        outputs=[chatbot, query, continue_message])
    regenerate_btn.click(regenerate, inputs=[last_state, max_length, top_p, temperature, allow_generate],
                         outputs=[chatbot, query, continue_message])
    interrupt_btn.click(interrupt, inputs=[allow_generate])

demo.queue(concurrency_count=4).launch(server_name='0.0.0.0', server_port=7860, share=False, inbrowser=False)
demo.close()
