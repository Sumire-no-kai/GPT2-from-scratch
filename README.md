# GPT-2 from Scratch

个人学习项目：使用 PyTorch 逐步实现 GPT-2，并记录实现和实验过程。

**标签：** `gpt-2` · `pytorch` · `transformer` · `from-scratch` · `language-model`

## 环境准备

需要与当前 PyTorch 版本兼容的 Python 3 和 pip。建议在虚拟环境中安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__)"
```

Windows 命令提示符用户可用 `.venv\Scripts\activate` 激活虚拟环境。需要 CUDA 或 ROCm 时，请参考 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 选择与设备匹配的安装命令。

## 参考资料

本项目的学习和实现参考了 [OpenAI 官方 GPT-2 仓库](https://github.com/openai/gpt-2)及其论文 [*Language Models are Unsupervised Multitask Learners*](https://cdn.openai.com/better-language-models/language-models.pdf)。本仓库是独立的个人学习项目，并非 OpenAI 官方项目。

## 许可证

本仓库使用 [MIT License](LICENSE)。
