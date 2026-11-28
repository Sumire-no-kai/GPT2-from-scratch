---
title: 阶段 3：加载预训练权重
summary: 把 OpenAI 训练好的 GPT-2 124M 参数放进自己写的模型，并用同一份参数的参考实现核对输出
date: 2026-10-10
tags: [GPT-2]
status: draft
---

# 阶段 3：加载预训练权重

## 导读

阶段 2 的 `GPT2()` 已经搭好了网络，但里面 1.2 亿多个参数还都是随机数，输出的分数没有意义。OpenAI 当年训练好 GPT-2 之后，把这些参数（也叫**权重**）公开了。这一阶段要做的，就是把这些训练好的数字一个不漏地放回我们模型里对应的位置。[阶段 2 模型前向](02-model.md#完整模型如何输出-logits)

为什么要这样做？因为这是检验阶段 2 有没有写对的最直接的办法。如果我们的结构和 GPT-2 有哪怕一处不同，比如 GELU 用错了公式、某个矩阵少转置一次、LayerNorm 放错了位置，同一份权重算出来的分数就会和参考实现对不上。举个实测的例子：只把 tanh 近似的 GELU 换成精确公式，logits 就会差出 `8e-2`；而两份正确的实现之间，只差 `1e-11` 这个量级的舍入误差（见 [Q2.2](#q22-为什么-float64-对照的误差是-15e-11-这样的数)）。这比训练一个模型再看 loss 快得多，也严格得多。

本阶段分两步：第 1 节把文件里的参数名和矩阵方向换成我们模型的写法，第 2 节用同一份权重跑参考实现，逐个核对输出分数。

## 0 进度与成果

| 任务 | 产出 |
|---|---|
| 加载权重 | `load_gpt2_weights(model, checkpoint_path)`，检查参数名和形状，转换投影布局并保留权重共享。 |
| 验收完整模型 | 5 组真实权重对照全部通过。环境、逐组误差和测试清单见[阶段 3 验收报告](#阶段-3-验收报告)。 |

## 1 把权重放进我们的模型

### 接口与约束

- 输入是已按 checkpoint 配置创建的 `GPT2` 实例，以及本地 Safetensors 文件路径。公开的 124M 文件对应 `GPT2()` 的默认配置。
- 允许文件带旧版的注意力遮罩 `h.i.attn.bias`。本项目会按输入长度现场生成遮罩，所以载入时跳过这个缓冲区。
- 先检查需要的键及形状，并报告缺少或多出的键。校验通过后把所有权重复制进传入的模型，返回值为 `None`。

OpenAI 最初发布的是 TensorFlow 格式的文件，直接读要装 TensorFlow。Hugging Face 把同一份数字转存成了 PyTorch 能直接读的 Safetensors 格式，放在 `openai-community/gpt2` 仓库里，我们用的就是这份转换。为了以后复现时拿到完全相同的文件，下载时固定到版本 `607a30d783dfa663caf39e06633721c8d4cfcd7e`。下面的名字和形状都来自这个版本的 `model.safetensors`。[权重文件](https://huggingface.co/openai-community/gpt2/blob/607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors) · [124M 模型配置](https://huggingface.co/openai-community/gpt2/blob/607a30d783dfa663caf39e06633721c8d4cfcd7e/config.json)

### 参数名与形状怎么对应

| 文件中的参数 | 本项目里的参数 | 124M 的源形状 → 目标形状 |
|---|---|---|
| `wte.weight` | `embeddings.wte.weight`、`lm_head.weight` | `(50257, 768) → (50257, 768)` |
| `wpe.weight` | `embeddings.wpe.weight` | `(1024, 768) → (1024, 768)` |
| `ln_f.weight`、`ln_f.bias` | 同名 | `(768,) → (768,)` |
| `h.i.ln_1.*`、`h.i.ln_2.*` | `blocks.i.ln_1.*`、`blocks.i.ln_2.*` | `(768,) → (768,)` |
| `h.i.attn.c_attn.weight` | `blocks.i.attn.c_attn.weight` | `(768, 2304) → (2304, 768)` |
| `h.i.attn.c_proj.weight` | `blocks.i.attn.c_proj.weight` | `(768, 768) → (768, 768)`，仍需转置 |
| `h.i.mlp.c_fc.weight` | `blocks.i.mlp.c_fc.weight` | `(768, 3072) → (3072, 768)` |
| `h.i.mlp.c_proj.weight` | `blocks.i.mlp.c_proj.weight` | `(3072, 768) → (768, 3072)` |
| 上面四个投影层的 `bias` | `h.i.` 换成 `blocks.i.` | 原样复制 |

表里的 `i` 是 block 编号，124M 模型从 0 数到 11。大部分参数只是换个名字：文件里叫 `h.3.ln_1.weight`，我们模型里叫 `blocks.3.ln_1.weight`。真正要动脑的是那四种投影权重，它们的行和列需要对调（转置）。

原因是两边存矩阵的方向不一样。OpenAI 的原始代码把线性层写成一个叫 `conv1d` 的函数（名字来自卷积，算的其实就是普通的线性变换），Hugging Face 转换时沿用了这个名字，叫 `Conv1D`。它的权重按「输入宽度 × 输出宽度」存，计算是 `x @ W + b`。PyTorch 的 `nn.Linear` 正好反过来，按「输出宽度 × 输入宽度」存，计算是 `x @ W.T + b`。以输入 2 个数、输出 3 个数的层为例：

| 写法 | 权重形状 | 计算 |
|---|---|---|
| 文件里的 `Conv1D` | `(2, 3)` | `x @ W + b` |
| 我们的 `nn.Linear` | `(3, 2)` | `x @ W.T + b` |

两种算法得到的结果一样，只是 `W` 互为转置，所以加载时把文件里的矩阵转置一次就行。要小心的是 `attn.c_proj.weight`：它是 `768 × 768` 的方阵，转置前后形状一样，光看形状发现不了漏转置，必须按「它属于哪种层」来判断。四个投影层的 `bias` 都是一维的，每个输出项加一个数，不需要转置。[Transformers 的 Conv1D 实现](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/pytorch_utils.py#L90)

### 加载共享的 token 表

阶段 2 让 `wte` 和 `lm_head` 共用同一张表：两个名字指向同一个参数对象，改一处另一处也跟着变。文件里只存了一份 `wte.weight`，加载时有两个坑要避开：

- **键要给全**。`state_dict()` 是按模块路径列参数的，所以会同时列出 `embeddings.wte.weight` 和 `lm_head.weight` 两个名字，虽然它们背后是同一份数字。`load_state_dict(..., strict=True)` 会严格核对名字，少给一个就报错。所以加载器把同一个源张量填给这两个名字。
- **共享不能被拆开**。`load_state_dict` 默认是把新数值复制进模型已有的参数里（`assign=False`），而不是换成一个新的参数对象，所以两个名字仍然指向同一个对象。

加载完以后检查 `model.lm_head.weight is model.embeddings.wte.weight` 是否为 `True`，就能确认共享还在。[阶段 2 的权重共享](02-model.md#初始化如何装好完整模型)

### 加载器代码

```python
# 将文件中的名字和矩阵布局转换成 GPT2 的 state_dict，再复制到已有参数中。
def load_gpt2_weights(model: GPT2, checkpoint_path: str | Path) -> None:
    """加载 openai-community/gpt2 的 model.safetensors 到已初始化的模型。

    模型配置须与 checkpoint 对应。词表和位置表分别是 (V, C) 和 (P, C)。
    文件中的 Conv1D 权重为 (输入宽度, 输出宽度)，nn.Linear 使用相反的轴顺序。
    键和形状在写入前检查。复制参数保留 wte 与 lm_head 的权重共享。
    """
    source_names = {
        "embeddings.wte.weight": "wte.weight",
        "embeddings.wpe.weight": "wpe.weight",
        "ln_f.weight": "ln_f.weight",
        "ln_f.bias": "ln_f.bias",
    }
    mask_names = set()
    for index, block in enumerate(model.blocks):
        # block 内的子层沿用 GPT-2 的命名，两边的名字只差 blocks.i. 和 h.i. 前缀。
        for suffix in block.state_dict():
            source_names[f"blocks.{index}.{suffix}"] = f"h.{index}.{suffix}"
        # 公开文件存了这个旧版因果遮罩，本项目的注意力会按输入长度生成。
        mask_names.add(f"h.{index}.attn.bias")

    conv_weights = (
        ".attn.c_attn.weight",
        ".attn.c_proj.weight",
        ".mlp.c_fc.weight",
        ".mlp.c_proj.weight",
    )
    target_state = model.state_dict()
    with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
        expected_names = set(source_names.values())
        file_names = set(checkpoint.keys())
        missing = expected_names - file_names
        unexpected = file_names - expected_names - mask_names
        if missing or unexpected:
            raise ValueError(
                f"Invalid GPT-2 checkpoint keys: missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )

        mapped = {}
        for target_name, source_name in source_names.items():
            tensor = checkpoint.get_tensor(source_name)
            needs_transpose = source_name.endswith(conv_weights)
            expected_shape = target_state[target_name].shape
            if needs_transpose:
                expected_shape = expected_shape[::-1]
            if tensor.shape != expected_shape:
                raise ValueError(
                    f"{source_name}: expected shape {tuple(expected_shape)}, "
                    f"got {tuple(tensor.shape)}"
                )
            mapped[target_name] = tensor.T if needs_transpose else tensor

        # strict=True 需要两个键。默认的复制行为保留模型里的同一个 Parameter。
        mapped["lm_head.weight"] = mapped["embeddings.wte.weight"]
        model.load_state_dict(mapped, strict=True)
```

1. `load_gpt2_weights` 接收已经建好的模型和本地文件路径。`source_names` 用来登记映射关系：键是我们模型里的参数名，值是文件里的名字。其中 `wte` 和 `wpe` 挂在 `embeddings` 下面，`ln_f` 两边同名。
2. `enumerate(model.blocks)` 逐层遍历每个 block。因为阶段 2 给 block 里的子层起名时特意沿用了 GPT-2 的命名（`ln_1`、`attn.c_attn`、`mlp.c_fc` 等），两边只差 `blocks.i.` 和 `h.i.` 前缀，直接拼上去就能全部对上。同时把每层可以忽略的旧版因果遮罩 `h.i.attn.bias` 放进 `mask_names`。
3. `conv_weights` 列出四类需要转置的投影权重后缀；`target_state = model.state_dict()` 用来提前查每个目标参数应有的形状。
4. 用 `safe_open(..., framework="pt", device="cpu")` 打开文件，整理出期望读到的键 `expected_names` 和文件里实际存在的键 `file_names`。
5. 通过集合做差一次性检查缺键和多键：`expected_names - file_names` 找出缺了什么，`file_names - expected_names - mask_names` 找出除旧遮罩之外多了什么。只要任意一边非空，就直接报错并列出具体参数名，避免静默漏加载。
6. 逐个用 `checkpoint.get_tensor(source_name)` 读出张量，并通过 `endswith(conv_weights)` 判断它是不是需要转置的投影权重。
7. 核对形状时，如果该参数需要转置，就把目标形状用 `[::-1]` 颠倒过来再比对（比如目标是 `(2304, 768)`，文件里就该是 `(768, 2304)`）。校验放在写入模型之前，一旦发现尺寸不匹配就立刻报错，不会把模型改到一半。
8. 校验通过后把结果放进 `mapped`（需要转置的取 `tensor.T`，其余原样放入）。最后补上 `lm_head.weight` 指向同一份 `wte` 张量，再调用 `model.load_state_dict(mapped, strict=True)` 一次性把数值复制进模型，同时保留权重共享。

单元测试用两个小 block 验收四种投影的转置、归一化、嵌入表和共享参数，并检查缺键、多键及错形状的报错。方形投影权重刻意用非对称的数值，漏转置时即使形状相同也会被发现。[加载器源码](../../gpt2/weights.py) · [单元测试](../../tests/test_gpt2_weights.py)

## 2 用官方权重对照输出

### 准备本地文件

在仓库根目录的 Python 环境里装好 `requirements.txt`。下面两条下载命令可以在 Mac 终端执行，`data/` 由 Git 忽略。用于参考对照的 `config.json` 与权重文件来自同一版本。

```bash
python -m pip install -r requirements.txt
mkdir -p data/gpt2
revision=607a30d783dfa663caf39e06633721c8d4cfcd7e
curl -fLo data/gpt2/model.safetensors "https://huggingface.co/openai-community/gpt2/resolve/$revision/model.safetensors"
curl -fLo data/gpt2/config.json "https://huggingface.co/openai-community/gpt2/resolve/$revision/config.json"
```

下载完可以用 `shasum -a 256 data/gpt2/model.safetensors` 算出文件的 SHA-256 指纹，确认文件完整、没被替换。实测文件大小为 548,105,171 字节，SHA-256 与发布页一致，为 `248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707`。文本用[阶段 1 的 tokenizer](01-tokenizer.md)转成 ID，因此文本对照示例会读同目录下已有的 `encoder.json` 和 `vocab.bpe`。

### 同一批 token 怎么比

参考答案来自 Hugging Face 的 `transformers` 库：它的 `GPT2LMHeadModel` 是公认可靠的 GPT-2 实现，能读同一份权重文件。它只用来对答案，所以没有写进 `requirements.txt`。想复现时，另建一个虚拟环境，先装好项目依赖，再执行 `python -m pip install transformers==4.57.6`。平时用 `GPT2()` 和 `load_gpt2_weights(...)` 加载权重，不需要它。

脚本里有几行是为了让对比干净：
- `torch.set_num_threads(4)` 固定 CPU 线程数，每次运行条件一样，计时也能比较。
- `torch.set_float32_matmul_precision("highest")` 要求矩阵乘法用完整的 float32 精度，不允许为了提速降低精度。
- 参考模型调用 `eval()` 关闭 dropout；`torch.inference_mode()` 告诉 PyTorch 不用记录梯度，省时间也省内存。
- `attn_implementation="eager"` 让参考模型用最直白的注意力写法，和我们手写的版本一致；`use_cache=False` 关闭生成时用来提速的 KV cache（阶段 6 再讲），让参考模型每次都完整地算一遍。

```python
from pathlib import Path
from time import perf_counter

import torch
from transformers import GPT2LMHeadModel

from gpt2.model import GPT2
from gpt2.tokenizer import GPT2Tokenizer
from gpt2.weights import load_gpt2_weights

root = Path("data/gpt2").resolve()
torch.set_num_threads(4)
torch.set_float32_matmul_precision("highest")
start = perf_counter()
model = GPT2().eval()
load_gpt2_weights(model, root / "model.safetensors")
assert model.embeddings.wte.weight is model.lm_head.weight

reference = GPT2LMHeadModel.from_pretrained(
    root,
    use_safetensors=True,
    local_files_only=True,
    dtype=torch.float32,
    attn_implementation="eager",
).eval()
tokenizer = GPT2Tokenizer(root / "encoder.json", root / "vocab.bpe")
cases = {
    "single-token": [[50256]],
    "English": [tokenizer.encode("The capital of France is")],
    "mixed Unicode": [tokenizer.encode("你好，GPT-2!")],
    "two sequences": [[50256, 18435, 11, 995, 0], [15496, 11, 616, 1438, 318]],
    "64-token sequence": [[15496, 11, 616, 1438, 318, 262, 1049, 13] * 8],
}
with torch.inference_mode():
    for label, values in cases.items():
        ids = torch.tensor(values, dtype=torch.long, device="cpu")
        actual = model(ids)
        expected = reference(ids, use_cache=False).logits
        assert actual.dtype == expected.dtype == torch.float32
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=0)
        top1 = actual.argmax(dim=-1) == expected.argmax(dim=-1)
        assert top1.all()
        error = (actual - expected).abs()
        print(label, tuple(actual.shape), error.max().item(), error.mean().item())
        print(f"Top-1: {top1.sum().item()}/{top1.numel()}, logits: {actual.numel()}")
print(f"构造、加载和对照用时 {perf_counter() - start:.2f} 秒")
```

每一组比较的是整个 `(B, T, V)` 张量，也就是每个位置对词表里全部 50,257 个候选的分数。`atol=1e-4, rtol=0` 要求每一对分数的绝对差都在 `0.0001` 以内；这个容差是本项目自己定的，不是 GPT-2 的官方标准，它的局限见 [Q2.3](#q23-atol1e-4-是-gpt-2-规定的标准吗)。`argmax(dim=-1)` 另外确认两边在每个位置给最高分的是同一个 token。

### 阶段 3 验收报告

**结论：通过。** 5 组输入共比较 4,623,644 个 logits，最大绝对误差为 `3.43322754e-5`，低于 `1e-4`。两边在全部 92 个位置上给最高分的 token ID 一致。

| 检查项 | 设置与实测结果 |
|---|---|
| 权重来源 | [同一版本的公开 Safetensors 文件及配置](#准备本地文件)，权重文件 548,105,171 字节，SHA-256 与发布页一致。 |
| 参数加载 | 文件含 160 个张量：跳过 12 个旧遮罩，用其余 148 个参数填充模型的 149 个状态键（多出 1 个键是与 `wte` 共享的 `lm_head.weight`）；12 个 block 共转置 48 个投影矩阵。 |
| 共享关系 | `model.embeddings.wte.weight is model.lm_head.weight` 为 `True`。共享表只计一次，共 124,439,808 个可训练参数。 |
| 对照环境 | Apple Silicon Mac CPU，float32，4 个 PyTorch 线程。Python 3.13.5、PyTorch 2.14.1、safetensors 0.7.0、Transformers 4.57.6。参考模型用 eager attention，并关闭 dropout 和 KV cache。 |
| 通过条件 | 两边使用相同 token ID 和权重。对比每个位置的整个词表分数，`atol=1e-4, rtol=0`。另检查两边的最高分 token ID。 |
| 计时 | 从构造模型到完成 5 组比较，初次约 2.3 秒，复跑为 2.48 秒。两次都包含从本地文件加载权重。 |

| 实测输入 | logits 形状 | 对比分数数目 | 最大绝对误差 | 平均绝对误差 | 最高分 token 一致的位置 |
|---|---|---:|---:|---:|---:|
| 单 token | `(1, 1, 50257)` | 50,257 | `3.43322754e-5` | `1.96672063e-5` | 1 / 1 |
| 英文 | `(1, 5, 50257)` | 251,285 | `0` | `0` | 5 / 5 |
| 中英混合 | `(1, 12, 50257)` | 603,084 | `0` | `0` | 12 / 12 |
| 两段组成的批次 | `(2, 5, 50257)` | 502,570 | `0` | `0` | 10 / 10 |
| 64 个 token | `(1, 64, 50257)` | 3,216,448 | `0` | `0` | 64 / 64 |

本次实测有 4 组逐项差值为 0，单 token 输入存在小幅浮点差异。验收依据是设定的绝对容差。上述数值对应本机和这 5 组输入。

**回归测试共 73 项通过。** 其中 4 项是本阶段新增的加载测试，已用一份独立写的逐参数复制实现复核。另外还做过一次性的检查（不在测试集里）：用 Transformers 生成一个两层的小 checkpoint，加载到本项目后核对完整 logits。

| 本阶段测试 | 验收结果 |
|---|---|
| 参数映射 | 两个 block 的四类投影都正确转置，嵌入、偏置和归一化参数正确复制。测试包含形状保持不变的方形权重。 |
| 权重共享 | 文件省略旧遮罩时仍可加载，`wte` 和 `lm_head` 保持对原有同一个参数的引用。 |
| 键检查 | 文件同时缺少一个需要的键且多出一个未知键时，报错列出两者。 |
| 形状检查 | 文件中靠后的投影尺寸出错时，在复制权重前报错，原有 `wte` 的值保持不变。 |

[加载回归测试](../../tests/test_gpt2_weights.py)

### 相关问答

#### Q2.1 为什么用本机 CPU，Colab 的 L4 或 T4 能跑吗？

**A：** 两张卡都能做前向。这里的主要工作是对照参数布局和输出，计算只涉及几组短输入。把两边先固定在同一台机器、同一设备和 float32，可以减少后端与精度差异带来的干扰。后续训练要反复做前向和反向传播，GPU 节省的时间更明显。

#### Q2.2 为什么 float64 对照的误差是 1.5e-11 这样的数？

**A：** 这个数字本身没有特别含义，它只是舍入误差累积出来的大小，要看的是数量级。复核时，我们在仓库外另写了一份不用 `model.py` 和 `weights.py` 的实现，直接按文件里 `Conv1D` 的 `x @ W + b` 计算。两边都换成 float64（每个数用 64 位存，比默认的 float32 精确得多）以后，1024 个 token 的输入最多只差 `1.5e-11`，8 个 token 的输入只差 `2.6e-13`。

float64 在 1 附近相邻两个可表示值的间隔约为 `2.2e-16`。一次前向经过 12 层，每次矩阵乘法包含很多次乘加。运算顺序、数值大小和累加过程都会影响最终误差。这里的 `1.5e-11` 是对这组输入直接测得的最大差值。

这次也观察到，两份 float32 实现与各自 float64 结果的最大差约为 `7e-3`，与浮点舍入误差会随精度而变化的现象一致。作为对照，只把 tanh 近似的 GELU 换成精确公式，同一次 float64 实验中的差值就达到 `8e-2`。这两组数值只描述实测设置，可用来判断这次对照中的差异大小。

#### Q2.3 atol=1e-4 是 GPT-2 规定的标准吗？

**A：** 不是。GPT-2 本身没有规定容差，`atol≈1e-4` 是本项目在 2026-10-10 定路线时自己选的初始值，当时就说好等实际对比后再调整。

用 float32 对比时，两份都正确的实现能不能控制在 `1e-4` 以内，取决于两件事：
- **两边的运算顺序是否几乎一样**。`transformers` 和我们的模型执行的几乎是同一串运算，所以上面 5 组里有 4 组差值正好是 0。Q2.2 那份独立实现的运算顺序不同，8 个 token 的输入就差了 `1.4e-4`，会被判为不通过，尽管它也是对的。
- **输入有多长、分数有多大**。本阶段最长的输入只有 64 个 token；到 1024 个 token 时，正常的舍入误差就到了 `2e-3`。

这五组输入都在设定的容差内通过，也都选出了相同的最高分 token。其他输入或运算顺序可能产生更大误差，测试范围以外的行为仍需验证。可以用 float64 做更严格的独立对照，为 float32 另定容差，并补上一组 1024 个 token 的测试。

## 3 回顾

本阶段先把文件里的名字、形状和矩阵方向对上，再用同一份权重和参考实现做前向对照，证明阶段 2 的结构和 GPT-2 一致。下一阶段会用这个加载好权重的模型，从最后一个位置的 logits 里选出一个 token，反复接到提示词后面，让它续写文本。[阶段 4 文本生成](04-generation.md)

## 附录 术语表

| 术语 | 含义 |
|---|---|
| 预训练 checkpoint | 模型训练后保存的一组参数。这里用它恢复 GPT-2 已经学到的数值。 |
| Safetensors | 存放张量名字、类型、形状和数值的权重文件格式。 |
| state_dict | PyTorch 模块按名字列出的参数和持久缓冲区。 |
| 转置 | 交换矩阵的行和列。GPT-2 文件里四种投影的权重需要这一步。 |
| 绝对误差 | 两个数相减后取绝对值。此处用所有 logits 中最大的差值记录结果。 |
| 容差 | 判断两个数「算一样」时允许的最大差距。`atol` 是绝对容差，`rtol` 是按数值大小换算的相对容差。 |
| Conv1D | OpenAI 原始代码和 Hugging Face 转换里对线性层的叫法。权重按「输入宽度 × 输出宽度」存，加载到 `nn.Linear` 时要转置。 |
| float32 / float64 | 用 32 位或 64 位存储浮点数。在 1 附近，float64 的表示间隔比 float32 小约 5 亿倍。 |
| 机器精度 | 浮点格式中，从 1 到下一个更大的可表示值的距离。float64 约为 `2.2e-16`，float32 约为 `1.2e-7`。 |
| 舍入误差 | 小数只能存有限位，每步运算都要舍掉一点；不同的运算顺序舍掉的位置不同，结果会有极小的差别。 |
