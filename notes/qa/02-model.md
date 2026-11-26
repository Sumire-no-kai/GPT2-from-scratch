---
title: 阶段 2：GPT-2 模型结构问答
summary: Embedding、注意力、Transformer block 与完整模型中的问题与回答
date: 2026-10-10
tags: [GPT-2]
status: draft
---

# 阶段 2：GPT-2 模型结构问答

## 导读

[阶段 1 的 tokenizer](01-tokenizer.md#1-tokenizer-的整体结构)把文本变成一串 token ID。模型从这些编号出发，先查表得到向量，再让 Transformer 处理整段序列。

## 0 进度与成果

| 小节 | 产出 | 状态 |
|---|---|---|
| 输入嵌入 | `GPT2Embeddings`，用 `wte` 和 `wpe` 合成每个位置的输入向量 | 已完成 |
| 因果自注意力 | 让当前位置按规则读取序列中已有的信息 | 待开始 |
| Transformer block 和完整模型 | 把注意力等组件接起来，输出下一个 token 的分数 | 待开始 |

## 1 模型的输入怎么走到输出

每段文本先由 tokenizer 编码。把多段等长的 ID 序列放进一个张量，形状是 `(B, T)`，其中 `B` 是一次处理的序列数，`T` 是每段序列的 token 数。输入嵌入为每个 ID 生成一条有 `C` 个数的向量，把张量变成 `(B, T, C)`。Transformer 后续处理这些向量，最后为下一个 token 的候选项打分。

## 2 输入嵌入

### 目标与接口

`GPT2Embeddings(vocab_size, max_seq_len, embedding_dim)` 建立两张表。把输入 ID 张量传给这个对象，就能得到每个位置的输入向量。[gpt2/model.py](../../gpt2/model.py)

| 名称 | 形状 | 含义 |
|---|---|---|
| `input_ids` | `(B, T)` | `B` 段序列，每段有 `T` 个整数 token ID。 |
| `wte.weight` | `(V, C)` | 词表大小 `V` 决定行数，每个 token ID 对应一行。 |
| `wpe.weight` | `(P, C)` | 最大序列长度 `P` 决定行数，位置 `0` 到 `P - 1` 一行一个。 |
| 输出 | `(B, T, C)` | 每个输入位置对应一条由两个向量相加得到的向量。 |

### 初始化两张查找表

```python
class GPT2Embeddings(nn.Module):
    """查出 token 向量和绝对位置向量，并将它们逐项相加。"""

    # 建立两张可训练的查找表。每个 token 一行，每个序列位置也有一行。
    def __init__(
        self, vocab_size: int, max_seq_len: int, embedding_dim: int
    ) -> None:
        """用词表大小 V、最大序列长度 P 和向量维度 C 初始化 embedding。

        wte.weight 的形状是 (V, C)，wpe.weight 的形状是 (P, C)。
        """
        super().__init__()
        self.wte = nn.Embedding(vocab_size, embedding_dim)
        self.wpe = nn.Embedding(max_seq_len, embedding_dim)

        # 沿用原版 GPT-2 的随机初始化，两个表分别使用不同的标准差。
        nn.init.normal_(self.wte.weight, std=0.02)
        nn.init.normal_(self.wpe.weight, std=0.01)
```

1. `class GPT2Embeddings(nn.Module)` 把两张表组织成一个 PyTorch 模块。传入词表大小、最大序列长度和向量维度，就可以创建不同大小的两张表。
2. `super().__init__()` 先初始化 `nn.Module`。后面把两张表赋到 `self` 上时，PyTorch 会登记它们，训练时可以找到这些参数。
3. `nn.Embedding(vocab_size, embedding_dim)` 建出 token 表 `wte`。例如词表有四个 token，向量长三个数字，实际权重 `wte.weight` 就有四行三列。
4. `nn.Embedding(max_seq_len, embedding_dim)` 建出位置表 `wpe`。行号用的是序列位置，列数仍然是 `embedding_dim`，所以两张表查出的向量能逐项相加。
5. 两行 `nn.init.normal_` 直接填充权重张量，给训练一个随机起点。`std` 控制初始数字的分散程度。这里沿用 GPT-2 的初始化，token 表使用 `0.02`，位置表使用 `0.01`。[GPT-2 官方实现](https://github.com/openai/gpt-2/blob/master/src/model.py)

### 前向计算怎么查表和相加

```python
    # 同一批序列共用位置编号 0 到 T - 1，每个位置把查出的两个向量相加。
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """输入 (B, T) 的整数 token ID，返回 (B, T, C) 的输入向量。

        B 为批量大小，T 为序列长度，C 为向量维度。
        ID 必须在词表范围内，T 不能超过位置表的行数。
        input_ids 和两张表应在同一设备上。
        """
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch_size, sequence_length)")
        sequence_length = input_ids.shape[1]
        if sequence_length > self.wpe.num_embeddings:
            raise ValueError("sequence length exceeds max_seq_len")

        # 新建的位置编号跟随输入设备，模型搬到 MPS 或 CUDA 时也能直接查表。
        position_ids = torch.arange(sequence_length, device=input_ids.device)
        token_embeddings = self.wte(input_ids)  # (B, T, C)
        position_embeddings = self.wpe(position_ids)  # (T, C)

        # (T, C) 沿批量维广播：每段序列的第 t 个 token 都加同一个位置向量。
        return token_embeddings + position_embeddings
```

1. 写 `embeddings(input_ids)` 时，PyTorch 会调用这里的 `forward`。它期望两个维度，例如 `[[1, 1]]` 表示一段有两个 token 的序列。`input_ids.ndim != 2` 在输入形状不符时给出明确报错。
2. `input_ids.shape[1]` 取第二维的大小，也就是这次实际处理的序列长度 `T`。`self.wpe.num_embeddings` 是位置表的行数。若 `T` 超过这个数，就没有足够的位置向量，函数在查表前报错。
3. `torch.arange(sequence_length, ...)` 生成 `0` 到 `T - 1` 的位置 ID。例如两个 token 对应 `[0, 1]`。`device=input_ids.device` 让新建的位置张量和输入在同一设备上。
4. `self.wte(input_ids)` 把每个 token ID 换成对应的一行。每个位置原本只有一个整数，现在有 `C` 个数，所以结果是 `(B, T, C)`。
5. `self.wpe(position_ids)` 把每个位置 ID 换成对应的一行，得到 `(T, C)`。例如所有序列的第 0 个 token 都用位置表的第 0 行。
6. `token_embeddings + position_embeddings` 为每段序列逐位置相加。PyTorch 把位置张量看作 `(1, T, C)`，沿批量这一维使用同一套位置向量。这种自动对齐叫广播，结果保持 `(B, T, C)`。

### 验收

测试用固定的小表验证两张表的形状、多个序列的逐项相加，以及计算梯度时重复 token 对同一行贡献的累加。形状不对和超过位置表范围的输入也有单独校验。[输入嵌入测试](../../tests/test_gpt2_embeddings.py)

用本地 GPT-2 tokenizer 把 `" Hello!"` 编成 `[18435, 0]`。包成一段序列后，输入形状是 `(1, 2)`。验收时把向量长度设为 `3`，输出得到 `(1, 2, 3)`，每个位置的值都等于对应的 token 行加位置行。

### 相关问答

#### Q2.1 Token embedding（`wte`）和位置 embedding（`wpe`）是什么？

**A：** 它们是两张查表用的数字表。每一行都是一个向量，也就是一串数字。`wte` 用 token ID 查行，告诉模型当前是什么 token。`wpe` 用序列位置查行，给这次出现加上位置信息。

假设 tokenizer 输出 `[18435, 18435]`，表示同一个 token 连续出现两次。这个手算例子里，每行只有三个数字。

| 出现位置 | 查 `wte` 得到的 token 向量 | 查 `wpe` 得到的位置向量 | 两个向量逐项相加 |
|---|---|---|---|
| 0 | `wte[18435] = [2, 1, 4]` | `wpe[0] = [1, 0, 2]` | `[3, 1, 6]` |
| 1 | `wte[18435] = [2, 1, 4]` | `wpe[1] = [0, 3, 1]` | `[2, 4, 5]` |

两次出现都查 `wte` 的同一行，位置却分别查 `wpe` 的第 0 行和第 1 行。相加后得到两个带位置信息的输入向量，然后交给 Transformer。表里的数字是模型参数，训练时会根据预测下一个 token 的误差更新。

#### Q2.2 不同模型的 token embedding 和位置 embedding 一样吗？

**A：** 把 token 变成向量是常见做法，但每个模型的词表、向量长度和参数都可能不同。比如两个模型收到同一个 ID `18435`，要先分别查它们的词表，才能知道这个 ID 各自代表什么。即使两者用同一套词表，同一个 token 查到的向量也可能不同，因为表里的数字可以在训练中学成不同的值。

模型利用位置的方式也会变。GPT-2 在输入端用位置查 `wpe`，再把查到的向量加到 token 向量上。LLaMA 用 RoPE（旋转位置编码），在注意力计算时根据位置旋转用于比较的向量，让这一步带上位置信息。[GPT-2 官方实现](https://github.com/openai/gpt-2/blob/master/src/model.py) · [LLaMA 官方实现](https://github.com/meta-llama/llama/blob/main/llama/model.py)

## 延伸问答

## 附录 术语表

| 术语 | 含义 |
|---|---|
| 向量 | 一串按顺序排列的数字，例如 `[2, 1, 4]`。 |
| token embedding | 按 token ID 查到的向量。GPT-2 的这张表叫 `wte`。 |
| 位置 embedding | 按当前序列位置查到的向量。GPT-2 的这张表叫 `wpe`。 |
| 位置编码 | 把序列里的位置信息带入模型计算的方法，例如查位置表或使用 RoPE。 |
| 广播 | PyTorch 对齐形状后执行逐项运算的规则。这里让一张位置向量表与批量里每段序列分别相加。 |

