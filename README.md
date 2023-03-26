# 💡Creative ChatGLM WebUI

👋 欢迎来到 ChatGLM 创意世界！你可以使用修订和续写的功能来生成创意内容！

* 📖 你可以使用“续写”按钮帮 ChatGLM 想一个开头，并让它继续生成更多的内容。
* 📝 你可以使用“修订”按钮修改最后一句 ChatGLM 的回复。

# 环境配置

环境配置参考官方链接：[https://github.com/THUDM/ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)

配置好之后，直接运行 `web_demo.py` 即可：

```bash
python web_demo.py
```

# 用法介绍

## 续写

### 原始对话

如果你直接问 ChatGLM：“你几岁了？”

它只会回答：“作为一个人工智能语言模型，我没有年龄，我只是一个正在不断学习和进化的程序。”

<img width="388" alt="image" src="https://user-images.githubusercontent.com/10473170/227778266-e7f2b55a-59de-4eee-bfa2-f28f911ec018.png">

### 续写对话

而如果你给它起个头：“我今年”

它就会回答：“我今年21岁。”

<img width="388" alt="image" src="https://user-images.githubusercontent.com/10473170/227778334-d459ad8d-7c16-466d-851c-5af174216773.png">

### 使用视频

![ChatGLM2](https://user-images.githubusercontent.com/10473170/227778636-a8fcd650-eeeb-44e5-8f24-9260b27cce5d.gif)

## 修订

### 原始对话

如果你直接跟 ChatGLM 说：“你是谁？”

它会回答：“我是一个名为 ChatGLM-6B 的人工智能助手，是基于清华大学 KEG 实验室和智谱 AI 公司于 2023 年共同训练的语言模型开发的。我的任务是针对用户的问题和要求提供适当的答复和支持。”

你再问它：“你几岁了？”

它只会说：“作为一个人工智能助手，我没有年龄，因为我只是一个程序，没有实际的肉体或生命。我只是一个在计算机上运行的程序，专门设计为回答用户的问题和提供相关的帮助。”

![image](https://user-images.githubusercontent.com/10473170/227777039-75b9dfb6-9b83-45af-8555-c3a27808c683.png)

### 修改对话

你可以改变它的角色，比如你通过“修订”功能，将它的回复改成：“我是杨开心。”

然后你再问它：“你几岁了？”

它就会回答：“我今年15岁。”

![image](https://user-images.githubusercontent.com/10473170/227777136-e2a176f8-6742-41a9-abaf-72a9540b834d.png)

### 使用视频

![未命名项目](https://user-images.githubusercontent.com/10473170/227777930-6aa5981a-0695-40c7-b083-b76bb063c481.gif)

# 实现原理

这个方法并没有训练，没有修改官方发布的权重，而只是对推理的函数做了修改。

续写的原理是，将用户的输入直接设置为 `history[-1][1]`，模拟模型自己的部分输出，然后继续走之后的推理函数 `stream_chat_continue` [code](https://github.com/ypwhs/CreativeChatGLM/blob/a5c6dd1/chatglm/modeling_chatglm.py#L1158)。

修订的原理是，将用户的输入直接设置为 `history[-1][1]`，模拟模型自己的完整输出，但是不走推理函数。

