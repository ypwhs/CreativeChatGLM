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

| 名称 | 大小 | 百度网盘 | 备注 |
| ---- | ---- | ---- | ---- |
| **小显存离线包** | 5.4 GB | [点击下载](https://pan.baidu.com/s/1s7XPtDBoyOV5VOD4Onf3Ug?pwd=cglm) | 使用 ChatGLM-6B-int4 权重的离线包，显存需求 8GB |
| 大显存离线包 | 14 GB | [点击下载](https://pan.baidu.com/s/1gL4GwWp6iNqh0CGwx6NGUA?pwd=cglm) | 使用 ChatGLM-6B 权重的离线包，显存需求 16GB |
| 环境离线包 | 2.6 GB | [点击下载](https://pan.baidu.com/s/1gTKi8mKGWM5xhmAz724qSA?pwd=cglm) | 不带权重的离线包，启动之后可以自动下载模型，默认自动下载 ChatGLM-6B 权重。 |

除了这些离线一键环境之外，你还可以在下面下载一些模型的权重，包括 `THUDM/chatglm-6b` 系列、`silver/chatglm-6b-slim` 系列、`BelleGroup/BELLE` 系列。

* 百度网盘链接：[https://pan.baidu.com/s/1pnIEj66scZOswHm8oivXmw?pwd=cglm](https://pan.baidu.com/s/1pnIEj66scZOswHm8oivXmw?pwd=cglm)

下载好环境包之后，解压，然后运行 `start_offline.bat` 脚本，即可启动服务：

<img width="734" alt="企业微信截图_16822982234979" src="https://user-images.githubusercontent.com/10473170/229680404-0b28dfd4-382e-4cfc-9392-997f134c0242.png">

如果你想使用 API 的形式来调用，可以运行 `start_offline_api.bat` 启动 API 服务：

<img width="734" alt="企业微信截图_16822982234979" src="https://user-images.githubusercontent.com/10473170/233877877-1a0a1daf-2cf1-41d1-9cd8-7f2ad8cb2427.png">

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

关于本项目中的离线包制作方法，可以查看下面的详细步骤。

<details><summary>点击查看详细步骤</summary>

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

## ⚠️必做

解压之后，记得删除 pth 文件，以解决安装依赖的问题。

比如我删除的文件路径是 `./system/python/python310._pth`

![image](https://user-images.githubusercontent.com/10473170/229679450-7acc005d-8203-4dd6-8be9-fa546aeaa2bf.png)

## 安装依赖

运行 [setup_offline.bat](setup_offline.bat) 脚本，安装依赖。

![image](https://user-images.githubusercontent.com/10473170/229679544-162b8db1-851f-47f0-af54-675c6a710b42.png)

## 下载离线模型

你可以使用 [download_model.py](download_model.py) 脚本下载模型，如果你的网络环境不好，这个过程可能会很长。下载的模型会存在 `~/.cache` 一份，存在 `./models` 一份。

当你之后使用 `AutoModel.from_pretrained` 加载模型时，可以从 `~/.cache` 缓存目录加载模型，避免二次下载。

![image](https://user-images.githubusercontent.com/10473170/229679938-44486557-dbc7-4e0b-9793-acfb6c46459e.png)

下载好的模型，你需要从 `./models` 文件夹移出到项目目录下，这样就可以离线加载了。

![image](https://user-images.githubusercontent.com/10473170/229680125-6af06b25-3d26-49cc-969b-4f6154c522de.png)

下载完模型之后，你需要修改 [app.py](app.py) 里的 `model_name`，改成你想加载的模型名称。

## 测试

使用 [start_offline.bat](start_offline.bat) 启动服务：

![image](https://user-images.githubusercontent.com/10473170/229680404-0b28dfd4-382e-4cfc-9392-997f134c0242.png)

可以看到，服务正常启动。

</details>

# 协议

本仓库的代码依照 [Apache-2.0](LICENSE) 协议开源，ChatGLM-6B 模型的权重的使用则需要遵循 [Model License](MODEL_LICENSE)。
