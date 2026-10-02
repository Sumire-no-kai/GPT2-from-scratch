# 开发日志

这里记录仓库里的决策：定了什么、为什么这样定、放弃了哪些方案，以及还有哪些待办。本文件由 AI 助手维护，按时间顺序追加，最新的记录在最后。

面向读者的学习笔记放在 [`notes/`](../notes/)，不写在这里。个人网站的 `/blog` 栏目也叫"开发日志"，但那是对外发布的学习笔记，对应的是 `notes/`。本文件只是内部的决策记录，不对外发布。

## 当前状态

- 前置知识大纲已整理至 [`prerequisites.md`](prerequisites.md)，可按学习时机和掌握标准准备。
- 还没正式开始学习，等用户通知。起点定在阶段 0（bigram 热身）。
- 阶段 0 的初步设计见 2026-09-26 的记录，正式开始时再确认。
- 本地使用 PyCharm 创建的 `.venv`（Python 3.13.5），已安装 PyTorch 2.14.1 和 NumPy 2.5.3；CPU / MPS 的基础运算与自动求导验证通过。
- 跨平台基线固定为 Python 3.13、PyTorch 2.14.1、NumPy 2.5.3；Mac 使用 MPS，Windows RTX 3060 使用 CUDA 12.6 构建，待 Windows 实机验证。
- 同步范围已确认：项目文档、协作约定和依赖配置纳入版本管理，`.idea/` 与 `.venv/` 留在本地。

## 2026-09-26 项目启动

### 仓库初始化

- 提交了 README、AGENTS.md、.gitignore 和 requirements.txt（`2b91d47`）。目前依赖只有 `torch`。
- 本机是 Apple M5（24 GB 内存），PyTorch 可以用 MPS。另有一台 RTX 3060 笔记本，可用来跑耗时较长的训练。

### 目标：以理解为主，不追求训练规模

- 仓库以学习为主，不追求复现 GPT-2 的训练结果。
- 模型实现是否正确，不靠训练来验证，而是加载 OpenAI 官方 GPT-2 124M 权重，和参考实现对比 logits。
- 训练只要求跑通流程：小模型跑几百步，loss 正常下降即可。需要长时间训练时，放到 3060 笔记本上跑。
- 因此代码要自动选择设备（`cuda` → `mps` → `cpu`），换机器时不用改代码。
- 放弃的方案：从零预训练 GPT-2 124M。在 M5 和 3060 笔记本上都不现实，原始规模的复现用了 8 张 A100 和约 100 亿 token。

### 学习路线

| 阶段 | 内容 | 验证方式 |
|---|---|---|
| 0 | 字符级 bigram 热身：数据分批、embedding、交叉熵、训练循环、生成 | loss 能降下来 |
| 1 | Tokenizer：自己实现 BPE，再弄懂 GPT-2 的 byte-level BPE | 编码结果和 `tiktoken` 的 `gpt2` 编码一致 |
| 2 | 模型结构：`wte`/`wpe`、多头因果自注意力、MLP（GELU）、Pre-LayerNorm、残差、`ln_f`、与 `wte` 共享权重的 `lm_head` | 参数量约 124M，各层 tensor 形状有标注 |
| 3 | 加载官方 124M 权重 | 与参考实现的 logits 一致 |
| 4 | 生成：temperature、top-k、top-p、KV cache | 生成通顺的英文 |
| 5 | 训练：AdamW、warmup + cosine 学习率、梯度裁剪、梯度累积、验证集、checkpoint | 小模型跑通，loss 正常下降 |
| 6 | 进阶（可选）：`scaled_dot_product_attention`、混合精度、`torch.compile`、微调、HellaSwag 评测 | 视兴趣而定 |

主要参考资料：OpenAI 的 GPT-2 仓库和论文，以及 Andrej Karpathy 的 *Let's reproduce GPT-2 (124M)* 视频和 `nanoGPT` / `build-nanogpt` 仓库。

### 分工：用户写实现，AI 引导

- AI 负责讲原理，提供骨架（函数签名、标注 tensor 形状的 docstring、TODO 提示）和测试。用户写完实现后由 AI review。
- 除非用户要求，AI 不直接写实现。
- 测试交给用户前，要先用一份不提交进仓库的参考实现验证，避免测试本身有错。

### 两份日志

- `docs/dev-log.md`（本文件）：AI 维护的决策日志。
- `notes/`：用户的学习笔记。主要内容由用户写，AI 负责润色，之后发布到个人博客，也可能同步到公众号。写作约定见 [`notes/README.md`](../notes/README.md)。

### 阶段 0 的初步设计（待正式开始时确认）

- `warmup/bigram.py` 包含以下部分：
  - `CharTokenizer`：字符按排序后的顺序编号。`set` 的遍历顺序每次运行都可能不同，所以要排序。
  - `get_batch(data, block_size, batch_size)`：返回 `x` 和 `y`，形状都是 `(B, T)`，`y` 是 `x` 右移一位。
  - `BigramLanguageModel`：`forward` 始终返回形状为 `(B, T, V)` 的 logits；`generate` 接收整条序列作为输入，这样换成 GPT-2 后几乎不用改。
  - `estimate_loss`：返回 Python float。
  - `main`：训练流程。
- 超参数参考 Karpathy 的 `ng-video-lecture`：`block_size` 8、`batch_size` 32、学习率 1e-2、训练 3000 步。
- `tests/test_bigram.py` 覆盖以下内容：
  - tokenizer 编码后能解码回原文，且编号按字符顺序分配。
  - `get_batch` 的输出形状、右移一位，以及 `data` 长度恰好为 `block_size + 1` 时的起点边界。
  - logits 形状，以及 loss 与 logits 一致。
  - bigram 只看当前 token。
  - `generate` 保留输入的前缀。
  - 在 `"abcabc…"` 上训练后，能确定地续写下去。
- 数据用 tiny Shakespeare（约 1 MB，放在 `data/`，不进版本库），下载命令由用户自己运行。
- 新增 `pytest` 依赖，并在 `pyproject.toml` 里设置 `pythonpath = ["."]`，这样直接运行 `pytest` 也能导入仓库里的模块。
- 预期结果：初始 loss 约为 ln(65) + 0.5 ≈ 4.7（embedding 用 N(0, 1) 随机初始化，所以高于 ln(65)），训练后降到 2.5 左右。

### 学习笔记的发布目标

- `notes/` 里的笔记发布到个人网站 [xiaonan.dev](https://xiaonan.dev) 的 `/blog` 栏目，源码在 `Sumire-no-kai/Personal_Website`，也可能同步到公众号。
- 网站上"从零实现 GPT-2"的现有内容只是草稿，以后会全部替换：包括 4 个阶段的开发计划、那篇 Tokenizer 预览记录，以及 `app/data/journal.ts` 里的记录结构。所以以本仓库的笔记和路线为准，网站跟着改，现在不用去对齐。
- 待定：
  - 网站目前支持中、英、日三种语言，而笔记只写中文。同步时英文和日文版本怎么处理，到时再定。
  - 网站现在每篇记录由固定的几段组成（遇到了什么、要解决的问题、目前的判断、准备怎么验证、结论、下一步），装不下大段讲解、公式和代码。长篇笔记怎么在网站上呈现，等第一篇写出来再定。

## 2026-10-03 PyCharm 环境检查

- 复用 PyCharm 已创建的项目 `.venv`：Python 3.13.5、arm64，隔离系统包，IDE 模块配置指向该环境；`.venv` 已被 Git 忽略。无需重建环境或使用全局解释器。
- 按 `requirements.txt` 安装 PyTorch，并加入 NumPy。首次导入 PyTorch 出现缺少 NumPy 的初始化警告，因此补齐依赖以支持张量与 NumPy 数组互转；未选择保留警告的环境。
- 当前安装版本为 PyTorch 2.14.1、NumPy 2.5.3。CPU / MPS 上的张量运算、反向传播，以及 NumPy 互转均通过检查。沙箱内 MPS 探测为不可用，沙箱外验证可用。
- 延续使用 PyTorch 手写模型结构的路线：张量运算和自动求导由框架提供，模型实现由用户完成；手写底层计算和自动求导引擎不作为当前阶段目标。
- Git 检查：远端 `master` 仍为 `ca16a28`，本地初始化提交 `2b91d47` 尚未推送。现有文档改动适合提交；建议 PyCharm 本机配置留在本地。本次未调整已有暂存内容，也未提交或推送。
- 待办：确认提交范围后同步项目文档与依赖；开始学习阶段 0 时再补测试依赖。

## 2026-10-03 固定跨平台依赖版本

- 在 `requirements.txt` 固定 PyTorch 2.14.1 和 NumPy 2.5.3，沿用 Mac 已验证的版本；建议两端使用 Python 3.13。仅固定直接依赖，尚不是所有传递依赖的锁文件。
- Windows 训练环境按此前记录的 RTX 3060 选择官方 CUDA 12.6 构建 `torch==2.14.1+cu126`。已直接查询官方包索引，确认存在 `cp313-cp313-win_amd64` 安装包。README 说明先装 CUDA 构建，再安装共享依赖，并提供实际 GPU 前向／反向验证命令。
- 采用同一 PyTorch 发布版本、按平台选择构建的方式；不继续使用无版本限制的依赖，也不把 CUDA 构建强加给 Mac。CUDA 12.6 用于既有 RTX 3060，暂不切换到 CUDA 13 系列。
- 按用户计划，Mac 负责学习、调试和小实验，Windows 为主要训练设备。先测吞吐与显存，确有需要时才考虑远端 GPU。
- 待办：Windows 端确认实际显卡、驱动、Python 版本，安装后验证 CUDA 运算；目前没有 Windows 实机测试结果。此处包版本一致不代表两种 GPU 后端结果逐位一致。

## 2026-10-03 同步仓库配置与文档

- 用户确认提交所有需要同步的内容：README、协作约定、开发日志、学习笔记说明、固定版本的依赖清单及忽略规则；连同已有初始化提交一起推送到远端 `master`。
- 将 `.idea/` 加入忽略规则，并仅从 Git 暂存区移除其中的文件，保留本机 PyCharm 配置。未采用提交整个 IDE 配置目录的方案，避免把本机解释器设置带到 Windows。
- 待办：Windows 端拉取后按 README 创建独立环境并验证 CUDA；随后按学习计划开始阶段 0。

## 2026-10-03 保存前置知识大纲

- 将对话中的八部分知识大纲保存为 `docs/prerequisites.md`，包含学习时机、具体知识点和掌握标准，并在 README 添加入口。
- 作为项目学习指南放在 `docs/`，不放入由用户撰写的 `notes/` 学习笔记，便于区分学习要求和个人学习记录。
- 待办：按大纲检查基础掌握情况，从字符级 bigram 开始，后续知识随实现逐步补齐。
