# 💡Creative ChatGLM WebUI

👋 欢迎来到 ChatGLM 创意世界！你可以使用修订和续写的功能来生成创意内容！

* 📖 你可以使用“续写”按钮帮 ChatGLM 想一个开头，并让它继续生成更多的内容。
* 📝 你可以使用“修订”按钮修改最后一句 ChatGLM 的回复。

# 环境配置

## 离线包

此安装方法适合：

* 非开发人员，不需要写代码
* 没有Python经验，不会搭建环境
* 网络环境不好，配置环境、下载模型速度慢

| 名称 | 文件名 | 大小 | 百度网盘 | 备注 |
| ---- | ---- | ---- | ---- | ---- |
| **小显存离线包** | CreativeChatGLM_<br/>py310_torch200_<br/>chatglm_6b_int4.zip | 5.55 GB | [点击下载](https://pan.baidu.com/s/1I_KqxWyjO1ksL-wzXZ0JCA?pwd=cglm) | 使用 ChatGLM-6B-int4 权重的离线包，<br/>显存需求 8GB |
| 大显存离线包 | CreativeChatGLM_<br/>py310_torch200_<br/>chatglm_6b.zip | 14.3 GB | [点击下载](https://pan.baidu.com/s/1wPldt4CY783-JOF80eF7bg?pwd=cglm) | 使用 ChatGLM-6B 权重的离线包，<br/>显存需求 16GB |
| 环境离线包 | CreativeChatGLM_<br/>py310_torch200_<br/>chatglm_6b.zip | 2.56 GB | [点击下载](https://pan.baidu.com/s/1v3RWNFCIBQGe0yPBfYTjxQ?pwd=cglm) | 不带权重的离线包，启动之后可以<br/>自动下载模型，默认自动下载 <br/>ChatGLM-6B 权重。 |
| ChatGLM-6B | chatglm-6b.zip | 11.7 GB | [点击下载](https://pan.baidu.com/s/1gqG9a_nnglRouA8yF9KPgA?pwd=cglm) | [THUDM/chatglm-6b](https://huggingface.co/THUDM/chatglm-6b) |
| ChatGLM-6B-int4 | chatglm-6b-int4.zip | 2.99 GB | [点击下载](https://pan.baidu.com/s/1TXlGUKxKr7EM4oaKWiVSqw?pwd=cglm) | [THUDM/chatglm-6b-int4](https://huggingface.co/THUDM/chatglm-6b-int4) |

如果上面没有你想要的，可以在这里看看：

* 百度网盘链接：[https://pan.baidu.com/s/1pnIEj66scZOswHm8oivXmw?pwd=cglm](https://pan.baidu.com/s/1pnIEj66scZOswHm8oivXmw?pwd=cglm)

<details><summary>点击查看详细步骤</summary>

下载好环境包之后，解压，然后运行 `start_offline.bat` 脚本，即可启动服务：

![image](https://user-images.githubusercontent.com/10473170/227984407-cdf8ba61-e4e0-4838-a34f-07de18cb37fc.png)

</details>

## 虚拟环境

此安装方法适合已经安装了 Python，但是希望环境与系统已安装的 Python 环境隔离的用户。

<details><summary>点击查看详细步骤</summary>

首先启动 `setup_venv.bat` 脚本，安装环境：

![image](https://user-images.githubusercontent.com/10473170/227982667-a8090ffa-f836-4ebc-93a1-91ab39d9259b.png)

然后使用 `start_venv.bat` 脚本启动服务：

![image](https://user-images.githubusercontent.com/10473170/227983154-27ed9751-b9c3-44ec-9583-31f192955b11.png)

</details>

## Python 开发环境

此项配置方法适合代码开发人员，使用的是自己系统里安装的 Python。

环境配置参考官方链接：[https://github.com/THUDM/ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)

配置好之后，运行 `app.py` 开始使用，或者使用 IDE 开始开发。

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

# 离线包制作方法

## 准备 Python

首先去 Python 官网下载：[https://www.python.org/downloads/](https://www.python.org/downloads/)

![image](https://user-images.githubusercontent.com/10473170/229679144-86d96c5c-58e0-4a54-9657-ccfe37943c6e.png)

注意要下载 `Windows embeddable package (64-bit)` 离线包，我选择的是 [python-3.10.10-embed-amd64.zip](https://www.python.org/ftp/python/3.10.10/python-3.10.10-embed-amd64.zip)。

![image](https://user-images.githubusercontent.com/10473170/229679189-1f8b2032-c92c-47ee-ba25-147f4acbf90f.png)

解压到 `./system/python` 目录下。

![image](https://user-images.githubusercontent.com/10473170/229679264-b3633920-757f-4ab8-b9f8-e79a21036146.png)

## 准备 get-pip.py

去官网下载：[https://bootstrap.pypa.io/get-pip.py](https://bootstrap.pypa.io/get-pip.py)

保存到 `./system/python` 目录下。

## 安装依赖

运行 [setup_offline.bat](setup_offline.bat) 脚本，安装依赖。

# 协议

本仓库的代码依照 [Apache-2.0](LICENSE) 协议开源，ChatGLM-6B 模型的权重的使用则需要遵循 [Model License](MODEL_LICENSE)。
