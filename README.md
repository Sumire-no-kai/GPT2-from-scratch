# GPT-2 from Scratch

个人学习项目：使用 PyTorch 逐步实现 GPT-2，并记录实现和实验过程。

**标签：** `gpt-2` · `pytorch` · `transformer` · `from-scratch` · `language-model`

## 环境准备

两端统一使用 Python 3.13；直接依赖固定为 PyTorch 2.14.1 和 NumPy 2.5.3。Mac 使用 MPS，Windows NVIDIA 显卡使用 CUDA，训练代码需要将模型和数据放到对应设备上。后续实现按 `cuda` → `mps` → `cpu` 选择设备。

每台机器分别创建自己的 `.venv`，不要跨系统复制虚拟环境。PyCharm 已创建 `.venv` 时，跳过创建步骤，并把项目解释器设为其中的 Python。

### Mac（Apple Silicon）

确认 `python3 --version` 为 3.13，再运行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__); assert torch.backends.mps.is_available(), 'MPS unavailable'; print('MPS ready')"
```

M 系列 GPU 使用 PyTorch 的 MPS 后端，无需安装 CUDA。本机 Python 3.13.5 / PyTorch 2.14.1 的 CPU、MPS 运算和自动求导已验证。

### Windows（NVIDIA GPU，CUDA 12.6）

以下组合面向 x64 Windows 和 RTX 3060。先安装 Python 3.13（64 位）及适合显卡的最新 NVIDIA 驱动，并运行 `nvidia-smi` 确认显卡和驱动可识别。在仓库根目录的 PowerShell 中运行：

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install "torch==2.14.1+cu126" --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0)); x = torch.ones(3, device='cuda', requires_grad=True); (x * x).sum().backward(); torch.cuda.synchronize(); print('CUDA forward/backward ready')"
```

`+cu126` 指 CUDA 12.6 构建；它满足共享清单中的 `torch==2.14.1`。先安装指定的 CUDA 构建，再安装共享依赖，避免选错 GPU 版本。已核对官方索引中存在 Python 3.13 / Windows x64 的此版本安装包，实际 CUDA 运算仍需在 Windows 机器上执行上述命令验证。

预编译 PyTorch 包自带所需的 CUDA 运行时依赖，普通训练通常不必另装 CUDA Toolkit 或 cuDNN；NVIDIA 驱动仍需兼容。参见 [PyTorch 安装说明](https://pytorch.org/get-started/locally/)、[官方 CUDA 包索引](https://download.pytorch.org/whl/cu126/torch/)及[运行时依赖说明](https://discuss.pytorch.org/t/should-i-install-the-extra-cudatoolkit-and-cudnn/194528/2)。

Mac 用于学习、调试和小实验，Windows 作为主要训练机器。先测实际训练速度和显存占用，再决定是否需要租用远端 GPU。

## 文档

- [前置知识大纲](docs/prerequisites.md)：各阶段需要掌握的知识与自查标准。
- [学习笔记](notes/)：学习过程中写的技术笔记。
- [开发日志](docs/dev-log.md)：项目中各项决策的记录。

## 参考资料

本项目的学习和实现参考了 [OpenAI 官方 GPT-2 仓库](https://github.com/openai/gpt-2)及其论文 [*Language Models are Unsupervised Multitask Learners*](https://cdn.openai.com/better-language-models/language-models.pdf)。本仓库是独立的个人学习项目，并非 OpenAI 官方项目。

## 许可证

本仓库使用 [MIT License](LICENSE)。
