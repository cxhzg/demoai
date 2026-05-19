# API Key 配置说明

本项目使用 DeepSeek API。

需要在项目根目录创建 .env 文件，并填写：

DEEPSEEK_API_KEY=你的真实 API Key

程序启动时会读取 .env 文件中的 DEEPSEEK_API_KEY。
如果没有配置，程序会提示需要配置 API Key。