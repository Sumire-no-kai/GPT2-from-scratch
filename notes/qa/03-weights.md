---
title: 阶段 3：加载预训练权重问答
summary: 参数映射、权重布局与参考输出对照中的问题与回答
date: 2026-10-10
tags: [GPT-2]
status: draft
---

# 阶段 3：加载预训练权重问答

## 导读

阶段 2 的 `GPT2()` 建好了网络，每个参数还是随机初始值。这一阶段把 GPT-2 训练好的数值放回对应的位置。两个模型如果使用同一组参数，输入相同的 token ID，每个位置给词表的打分也应在容差内对齐。[阶段 2 模型前向](02-model.md#完整模型如何输出-logits)

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

权重源是 `openai-community/gpt2` 的公开 Transformers 格式转换，固定到版本 `607a30d783dfa663caf39e06633721c8d4cfcd7e`。下面的名字和形状来自该版本的 `model.safetensors`。[权重文件](https://huggingface.co/openai-community/gpt2/blob/607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors) · [124M 模型配置](https://huggingface.co/openai-community/gpt2/blob/607a30d783dfa663caf39e06633721c8d4cfcd7e/config.json)

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

`i` 是 block 编号，124M 模型从 0 数到 11。源文件的 `Conv1D` 投影用 `x @ W + b`，`W` 按输入、输出的顺序存。`nn.Linear` 用 `x @ W.T + b`，因此要先将源权重的两条轴交换。方阵转置前后形状相同，所以 `attn.c_proj.weight` 也得按层的类型判断，光看形状会漏掉。线性层的 `bias` 就是一维的输出偏置，每个输出项加一个数。[Transformers 的 Conv1D 实现](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/pytorch_utils.py#L90)

### 加载共享的 token 表

阶段 2 的 `wte` 和 `lm_head` 引用同一个 `Parameter`。`state_dict()` 按模块路径记录权重，因此会列出 `embeddings.wte.weight` 和 `lm_head.weight` 两个键，两者共用同一份张量存储。当前加载器给这两个目标键填入同一个源张量，让 `strict=True` 的键检查通过。默认的 `assign=False` 把数值复制到已有参数中，保留共享。验收时检查 `model.lm_head.weight is model.embeddings.wte.weight` 为 `True`。[阶段 2 的权重共享](02-model.md#初始化如何装好完整模型)

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

1. 函数接收模型和路径。`source_names` 的键是我们要填的位置，值是文件里的名字。`wte` 和 `wpe` 在项目里属于 `embeddings`，最终归一化 `ln_f` 的名字两边一样。
2. `enumerate(model.blocks)` 同时取到每个 block 和编号。`block.state_dict()` 提供诸如 `attn.c_attn.weight` 的名字，给同一个名字分别加上 `blocks.i.` 和 `h.i.`，就能连接两份结构。`mask_names` 则记下每一层可跳过的旧遮罩。
3. `conv_weights` 列出所有要转置的权重后缀。`target_state = model.state_dict()` 让后面能读目标参数的形状。
4. `safe_open(..., framework="pt", device="cpu")` 打开本地文件，为后续取出 CPU 上的 PyTorch 张量做准备。`expected_names` 是所需的源键，`file_names` 是实际出现的键。
5. `expected_names - file_names` 找出缺少的键。反向相减找出多余的键，再去掉已知的旧遮罩。两边的差集都有可能为空。存在差异时，错误会列出具体的名字。
6. `checkpoint.get_tensor(source_name)` 按源键读张量。`endswith(conv_weights)` 判断该键是否属于四组投影权重。
7. `target_state[target_name].shape` 是目标形状。如果需要转置，`[::-1]` 将目标的两个维度倒过来，得到源文件该有的形状。例如目标是 `(2304, 768)`，文件里就该是 `(768, 2304)`。形状不符时报出参数名以及期望和实际值。
8. `mapped[target_name] = ...` 准备最终要加载的字典。需要转置的张量取 `tensor.T`，其余直接用原张量。这时模型还保留原参数值。
9. 在 `mapped` 里补上共享的输出层键，然后调用 `load_state_dict(..., strict=True)`。PyTorch 把这份字典里的数值复制进模型。`strict=True` 要求映射后的目标键与模型匹配。

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

实测文件大小为 548,105,171 字节，SHA-256 与发布页一致，为 `248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707`。文本用[阶段 1 的 tokenizer](01-tokenizer.md)转成 ID，因此文本对照示例会读同目录下已有的 `encoder.json` 和 `vocab.bpe`。

### 同一批 token 怎么比

下面的对照代码需要在能导入 `transformers==4.57.6` 的验证环境中运行。平时用 `GPT2()` 和 `load_gpt2_weights(...)` 加载时，只用项目的常规依赖。对照脚本从本地读取同一份权重，参考模型使用 `eval()` 关闭 dropout，`inference_mode()` 省去梯度记录。

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

`use_cache=False` 让参考模型按完整输入做一次前向。每一组比较的是整个 `(B, T, V)` 张量。`atol=1e-4, rtol=0` 要求每对分数的绝对差都在 `0.0001` 内。`argmax(dim=-1)` 额外确认两边在每个位置给最高分的 token ID 一样。

### 阶段 3 验收报告

**结论：通过。** 5 组输入共比较 4,623,644 个 logits，最大绝对误差为 `3.43322754e-5`，低于 `1e-4`。两边在全部 92 个位置上给最高分的 token ID 一致。

| 检查项 | 设置与实测结果 |
|---|---|
| 权重来源 | [同一版本的公开 Safetensors 文件及配置](#准备本地文件)，权重文件 548,105,171 字节，SHA-256 与发布页一致。 |
| 参数加载 | 文件含 160 个张量。跳过 12 个旧遮罩，以 148 个源参数张量填充模型的 149 个状态键。12 个 block 共转置 48 个投影矩阵。 |
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

**回归测试共 73 项通过。** 其中 4 项是本阶段新增的加载测试，并用独立的逐参数复制实现复核。另用 Transformers 生成一个两层小 checkpoint，加载到本项目后核对完整 logits。

| 本阶段测试 | 验收结果 |
|---|---|
| 参数映射 | 两个 block 的四类投影都正确转置，嵌入、偏置和归一化参数正确复制。测试包含形状保持不变的方形权重。 |
| 权重共享 | 文件省略旧掩码时仍可加载，`wte` 和 `lm_head` 保持对原有同一个参数的引用。 |
| 键检查 | 文件同时缺少一个需要的键且多出一个未知键时，报错列出两者。 |
| 形状检查 | 文件中靠后的投影尺寸出错时，在复制权重前报错，原有 `wte` 的值保持不变。 |

[加载回归测试](../../tests/test_gpt2_weights.py)

### 相关问答

#### Q2.1 为什么用本机 CPU，Colab 的 L4 或 T4 能跑吗？

**A：** 两张卡都能做前向。这里的主要工作是对照参数布局和输出，计算只涉及几组短输入。把两边先固定在同一台机器、同一设备和 float32，可以减少后端与精度差异带来的干扰。后续训练要反复做前向和反向传播，GPU 节省的时间更明显。

## 回顾

本节先确认名字、形状与矩阵布局，再用同一份权重做前向对照。下一阶段会使用这些已经加载好权重的模型，从最后一个位置的 logits 中选出一个 token，反复追加到提示词后面。[阶段 4 文本生成](04-generation.md)

## 延伸问答

## 附录 术语表

| 术语 | 含义 |
|---|---|
| 预训练 checkpoint | 模型训练后保存的一组参数。这里用它恢复 GPT-2 已经学到的数值。 |
| Safetensors | 存放张量名字、类型、形状和数值的权重文件格式。 |
| state_dict | PyTorch 模块按名字列出的参数和持久缓冲区。 |
| 转置 | 交换矩阵的行和列。GPT-2 文件里四种投影的权重需要这一步。 |
| 绝对误差 | 两个数相减后取绝对值。此处用所有 logits 中最大的差值记录结果。 |
