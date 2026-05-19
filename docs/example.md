# 项目部署说明

本项目是一个本地文档问答 Agent。

它会读取 docs 文件夹里的 Markdown 和 TXT 文档，然后根据用户的问题搜索相关片段，最后调用大模型生成回答。

启动步骤：

1. 安装依赖：py -m pip install -r requirements.txt
2. 配置 .env 文件，填入 DEEPSEEK_API_KEY
3. 运行命令：py agent.py

如果没有配置 DEEPSEEK_API_KEY，程序无法调用模型。
