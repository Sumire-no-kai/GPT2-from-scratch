---
title: 阶段 4：文本生成
summary: 从贪心解码到 temperature、top-k、top-p 采样，让加载好权重的 GPT-2 自己续写英文
date: 2026-10-10
tags: [GPT-2]
status: draft
---

# 阶段 4：文本生成

## 导读

[阶段 3](03-weights.md#阶段-3-验收报告)已经证明，我们的 GPT-2 装上官方 124M 权重后，五组测试输入的分数与参考实现的差值都在设定容差内。这一阶段要把这个打分能力真正用起来：给模型一个开头，让它自己往下续写。

模型一次前向只能回答「下一个 token 可能是谁」。要写出一整段话，办法很直白：每轮选出一个 token 拼到已有文字后面，再把更长的序列送回模型继续猜下一个——这就是**自回归生成**。这一阶段先写通默认的贪心生成，再加入 `temperature`、`top-k`、`top-p` 三种采样控制。

## 0 进度与成果

| 小节 | 产出 | 状态 |
|---|---|---|
| 贪心生成 | `GPT2.generate()`，每轮追加得分最高的 token | 已完成，124M 权重下 60 / 60 个新 token 与参考一致 |
| 采样 | `do_sample=True` 启用 temperature、top-k 和 top-p | 已完成，单项与组合采样共 60 / 60 个新 token 与参考一致 |

## 1 生成循环

### 生成循环怎么做

`generate(input_ids, max_new_tokens)` 接收形状为 `(B, T)` 的整数提示词张量（`B` 段输入，每段 `T` 个 token）。比如把 `Once upon a time` 编成 4 个 token 送进 `forward()`，模型会给这 4 个位置都打分；但生成时我们只看**最后一个位置**的那组分数，因为它才对应「整句提示词后面接什么」。

```text
当前序列 (B, S)
    → 截取最近 P 个 token 作为窗口     (B, min(S, P))
    → GPT2.forward                    (B, min(S, P), V)
    → 只取最后一个位置的分数            (B, V)
    → 贪心或采样，各选 1 个 token ID   (B, 1)
    → 拼回当前序列末尾                  (B, S + 1)
```

其中 `S` 是本轮开始前的总长度（初始等于 `T`），`P` 是位置表容量（124M 默认为 1,024），`V` 是词表大小（50,257）。每追加一个新 ID，`S` 就加 1，进入下一轮循环。[阶段 2 模型前向](02-model.md#完整模型如何输出-logits)

接口有几条明确约定：

- 输入批次和提示词都不能为空（否则没有「最后一个位置」），数据类型须为 `torch.int32` 或 `torch.int64`，且与模型在同一设备上；同一批次里各段等长。
- `max_new_tokens` 须为非负整数；返回 `(B, T + max_new_tokens)`，原样保留开头的提示词和整数类型。传 `0` 时直接返回原输入，不做前向计算。
- 默认 `do_sample=False` 走贪心模式，每轮挑分数最高的那个词；只有显式传入 `do_sample=True` 时，才按关键字参数使用 `temperature`、`top_k`、`top_p`。
- 当总长度超过位置表上限 `P` 时，前向只截取最近的 `P` 个 token 来算（滑动上下文窗口），但返回值依然保留完整的提示词和全部新生成的 ID。

### 代码与逐行解释

代码位于 [gpt2/model.py](../../gpt2/model.py)：

```python
    # 每轮用最近的上下文预测一个 token，选出编号后追加到完整序列。
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        *,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """为每段输入续写 max_new_tokens 个 token，默认每轮贪心选择。

        input_ids 是形状 (B, T) 的非空整数张量，与模型在同一设备上。
        返回 (B, T + max_new_tokens)，保留原提示词和编号类型。
        超出位置表容量时，前向只读取最近的 token，返回值仍保留完整序列。
        do_sample=True 时缩放分数并按 top_k、top_p 的顺序限制抽样候选。
        temperature 须为有限正数，top_k 为正整数或 None，top_p 须在 (0, 1]。
        """
        if input_ids.ndim != 2 or input_ids.numel() == 0:
            raise ValueError("input_ids must have nonempty shape (batch_size, sequence_length)")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must have dtype torch.int32 or torch.int64")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if do_sample:
            if not math.isfinite(temperature) or temperature <= 0:
                raise ValueError("temperature must be finite and positive")
            if top_k is not None and (type(top_k) is not int or top_k <= 0):
                raise ValueError("top_k must be a positive integer or None")
            if not 0 < top_p <= 1:
                raise ValueError("top_p must be in (0, 1]")

        max_context = self.embeddings.wpe.num_embeddings
        for _ in range(max_new_tokens):
            # 只裁剪本轮送入模型的窗口，保留需要返回的完整 input_ids。
            context = input_ids[:, -max_context:]  # (B, min(当前长度, max_context))
            logits = self(context)  # (B, 窗口长度, V)
            last_logits = logits[:, -1, :]  # (B, V)
            if do_sample:
                next_ids = self._sample_next_token(last_logits, temperature, top_k, top_p)  # (B, 1)
            else:
                next_ids = last_logits.argmax(dim=-1, keepdim=True)  # (B, 1)
            input_ids = torch.cat((input_ids, next_ids.to(input_ids.dtype)), dim=1)  # (B, 当前长度 + 1)

        return input_ids  # (B, T + max_new_tokens)
```

1. `@torch.no_grad()` 关闭梯度记录：生成只做前向预测，不用反向传播，这样能省下大量显存和计算。至于是否切换到 `model.eval()`，留给调用方决定（下文对照实测都调用了 `model.eval()`）。
2. 函数签名里的 `*` 表示后面的参数必须按名字传（避免把 `temperature` 和 `top_p` 传混）。入口先检查输入维度、非空、整数类型和步数；只有开启 `do_sample=True` 时才校验三个采样参数（这里用 `type(top_k) is not int` 是为了挡掉 Python 里属于 `int` 子类的 `bool` 值，比如 `top_k=True`）。
3. `max_context = self.embeddings.wpe.num_embeddings` 直接从位置表读取最大长度，不管是 124M 模型还是测试用的小模型都能自动适配，不用把 `1024` 写死。
4. `context = input_ids[:, -max_context:]` 沿序列长度维截取最后 `max_context` 个 token。比如位置表只有 4 行而当前序列已有 5 个 token `[0, 2, 3, 4, 5]`，这一轮就只把 `[2, 3, 4, 5]` 送进模型，但 `input_ids` 自己保持完整。
5. `self(context)` 得到 `(B, 窗口长度, V)` 的分数；`logits[:, -1, :]` 取出每段序列最后一个位置对全部 `V` 个候选的打分，形状为 `(B, V)`。
6. 贪心分支用 `argmax(dim=-1, keepdim=True)` 直接挑出每行最高分的编号（如果最高分并列，取编号最小的那个）；采样分支把分数交给 [`_sample_next_token`](#2-采样)。两条分支都返回 `(B, 1)`。
7. `next_ids.to(input_ids.dtype)` 保持整数类型与输入一致，再通过 `torch.cat(..., dim=1)` 把新列拼到 `input_ids` 末尾，进入下一轮。循环结束后返回完整 ID 序列，交给 tokenizer 的 `decode` 就能读成文本。

### 和阶段 0 的 Bigram 续写怎么接上

主干循环一模一样：都是「取最后一个位置的打分 → 选 1 个 ID → 拼到末尾 → 重复」。区别在于，[Bigram](../../warmup/bigram.py) 只看最后一个字符、而且只做最基础的 `softmax + multinomial` 抽样；GPT-2 会综合整个可见窗口里的上文来打分，并且支持贪心和精细可控的采样。比如面对分数 `[0.2, 1.5, 0.8]`，贪心会确定地选出 ID `1`。

## 2 采样

### 为什么不能一直用贪心

贪心解码每一步都死选概率最大的那个词，听起来最稳妥，但写长了很容易**呆板、重复、甚至陷入复读机**。看下文实测里 `Once upon a time` 的贪心输出就能感受到：模型刚写完 `the world was a place of great beauty and great danger.`，紧接着又绕回了 `The world was`。

想让文字更自然、更有变化，就需要按概率**采样**（抽签）。但直接按原始概率抽签也有风险：词表里 5 万个词都有一点点非零概率，几万个长尾词加起来，偶尔就会抽中一个毫无关系的怪词，把整句话带偏。于是就有了三个经典旋钮来控制抽签的分布：

| 参数 | 怎么改分布 | 默认值与范围 |
|---|---|---|
| `temperature`（温度） | 把 logits 除以温度再做 softmax。低温（如 `0.7`）拉大分数差距，让头部候选更集中；高温（如 `1.5`）缩小差距，分布更平、更有随机性。 | 默认 `1.0`（不变），须为有限正数。 |
| `top_k` | 只留下得分最高的 `k` 个候选，其余全部屏蔽，再在这 `k` 个里抽。 | 默认 `None`（不限制），设置时为正整数。 |
| `top_p`（核采样） | 按概率从高到低累加，只保留累计概率达到或超过 `p` 的最短前缀。 | 默认 `1.0`（不限制），范围 `(0, 1]`。 |

**为什么有了 `top_k` 还要 `top_p`？** 因为模型在不同位置上的「把握程度」差得很远。如果模型非常确定下一个词（比如 `The capital of France is` 后面 `Paris` 独占 95%），固定的 `top_k=40` 会把后面 39 个不太合理的词也留进抽签池；反过来，如果开头很开放（比如 `Once upon a time`），几十个词都通顺，过小的 `top_k` 又会把好词切掉。`top_p` 能随当前分布自动伸缩候选数量：分布越尖留得越少，分布越平留得越多。[Nucleus sampling 论文](https://arxiv.org/abs/1904.09751)

**手算看看三个参数的效果：**

- **temperature**：logits 为 `[2, 1, 0]` 时，温度 `1.0` 下 softmax 概率约为 `[0.665, 0.245, 0.090]`；降到 `0.5` 时最高项飙到 `0.867`；升到 `2.0` 时最高项降到 `0.506`（只要温度是正数，候选的先后排名不会改变）。
- **top-k 与 top-p**：假设排序后的概率是 `[0.55, 0.30, 0.10, 0.05]`。`top_k=2` 留前两项；`top_p=0.8` 也留前两项（第 1 项 `0.55` 还不到 `0.8`，加上第 2 项变成 `0.85`，首次达到或超过门槛）。但如果分布变得很集中，变成 `[0.81, 0.10, 0.06, 0.03]`，同样的 `top_p=0.8` 只留第 1 项就够了。过滤后剩余候选重新归一化——例如保留前两项 `[0.55, 0.30]` 时，实际抽签概率变成 `[0.647, 0.353, 0, 0]`。

三个参数同时开启时，执行顺序固定为 **`temperature → top_k → top_p → 抽样`**（`top_p` 是在温度缩放、且经 `top_k` 截断之后的分布上做累计）。例如 `model.generate(ids, 12, do_sample=True, temperature=0.8, top_k=40, top_p=0.9)` 会在每一步生成时重新走一遍这套流程。

### 采样函数与逐行解释

```python
    # 先调分数，再屏蔽候选，最后从每行的归一化概率中抽一个编号。
    @staticmethod
    def _sample_next_token(
        logits: torch.Tensor, temperature: float, top_k: int | None, top_p: float
    ) -> torch.Tensor:
        """从 (B, V) 的分数按温度、top-k、top-p 抽样，返回 (B, 1) 的编号。"""
        if logits.dtype in (torch.float16, torch.bfloat16):
            logits = logits.float()  # (B, V)，用 float32 算概率和累计概率。
        if temperature != 1.0:
            # 同行分数减去最大值后再缩放，防止低温时最大分数溢出。
            centered = logits - logits.amax(dim=-1, keepdim=True)  # (B, V)
            # 保留最大分数对应的 0，即使温度小到当前精度下的除法溢出。
            logits = (centered / temperature).masked_fill(centered == 0, 0.0)  # (B, V)

        if top_k is not None and top_k < logits.size(-1):
            keep_logits, keep_ids = logits.topk(top_k, dim=-1)  # (B, K)，(B, K)
            logits = torch.full_like(logits, float("-inf")).scatter_(  # (B, V)
                dim=-1, index=keep_ids, src=keep_logits
            )

        if top_p < 1.0:
            sorted_logits, sorted_ids = logits.sort(dim=-1, descending=True)  # (B, V)，(B, V)
            cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)  # (B, V)
            # 前面的候选已达到阈值时才屏蔽当前位置，保留让总和达标的那个 token。
            remove = torch.cat(
                (torch.zeros_like(cumulative[:, :1], dtype=torch.bool), cumulative[:, :-1] >= top_p),
                dim=-1,
            )  # (B, V)
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))  # (B, V)
            logits = torch.empty_like(logits).scatter_(  # (B, V)，恢复词表原顺序。
                dim=-1, index=sorted_ids, src=sorted_logits
            )

        probs = logits.softmax(dim=-1)  # (B, V)，被屏蔽的位置概率为 0。
        return torch.multinomial(probs, num_samples=1)  # (B, 1)
```

1. 输入 `logits` 形状是 `(B, V)`。如果模型跑在 `float16` 或 `bfloat16` 下，先转成 `float32`，避免后面做 `softmax` 和 `cumsum` 累计概率时精度不够。
2. **温度缩放**：不是直接拿 `logits / temperature`，而是先减去每行的最大值 `amax`。数学上 softmax 对整行同减一个常数完全等价，但这样最大值变成了 `0`、其余全是负数，即使温度设得极小（比如 `1e-30`），最大项也不会因为除法溢出成 `inf` / `nan`；再用 `masked_fill(centered == 0, 0.0)` 兜底，保证至少有一个有限分数可以归一化。
3. **top-k 过滤**：当 `top_k` 小于词表大小时，用 `topk` 找出前 `K` 个分数和它们原来的词表下标，先建一张全是 `-inf` 的 `(B, V)` 表，再用 `scatter_` 只把前 `K` 个分数填回对应位置，其余候选自动被屏蔽。
4. **top-p 过滤**：先把每行按分数从高到低排序，算出排序后的概率并用 `cumsum` 求前缀累计和（如 `[0.55, 0.30, 0.10, 0.05] → [0.55, 0.85, 0.95, 1.00]`）。
5. 构造屏蔽掩码 `remove` 时有个巧思：第 0 列固定为 `False`（最高分候选永远保留），其余每一列看的是**前一位**的累计概率是否已经 `>= top_p`。以 `p = 0.8` 为例，第 2 项看的是第 1 项的 `0.55 < 0.8`，所以第 2 项保留；第 3 项看的是第 2 项的 `0.85 >= 0.8`，于是从第 3 项起全部屏蔽。这样刚好留住「让总和跨过门槛的那一个词」。
6. 把被屏蔽的位置填成 `-inf`，再按 `sorted_ids` 用 `scatter_` 把分数放回原来的词表顺序。
7. 最后做一次 `softmax`（所有 `-inf` 变成概率 `0`，剩余候选重新归一化到和为 `1`），交给 `torch.multinomial(..., num_samples=1)` 按概率抽 1 个 ID，返回 `(B, 1)`。

## 3 测试检查了什么与参考对照

### 单元测试

[生成测试](../../tests/test_gpt2_generate.py) 一共 16 项测试，全部用仓库外的纯 Python 列表循环独立复核通过（至此全量 **89 项测试通过**）：

- **贪心生成（7 项）**：覆盖批量生成、超长上下文滑动窗口、原提示词不被原地修改、生成 0 个 token 跳过前向、`int32`/`int64` 类型保留以及非法输入报错。
- **采样控制（9 项）**：覆盖温度缩放与极端低温不溢出、top-k 候选数及并列边界、top-p 刚好达标边界、三者组合顺序、固定种子可复现、默认仍为贪心以及非法参数报错。

### 用官方 124M 权重与参考模型逐个对照

用和阶段 3 相同的本地 124M 权重，与 Transformers 4.57.6 的 `generate()` 逐个比对。每次采样前两边都把随机种子重设为 `7`（权重与参考环境准备见[阶段 3 笔记](03-weights.md#2-用官方权重对照输出)）：

```python
from pathlib import Path

import torch
from transformers import GenerationConfig, GPT2LMHeadModel

from gpt2.model import GPT2
from gpt2.tokenizer import GPT2Tokenizer
from gpt2.weights import load_gpt2_weights

torch.set_num_threads(4)
torch.set_float32_matmul_precision("highest")
root = Path("data/gpt2").resolve()
tokenizer = GPT2Tokenizer(root / "encoder.json", root / "vocab.bpe")
model = GPT2().eval()
load_gpt2_weights(model, root / "model.safetensors")
reference = GPT2LMHeadModel.from_pretrained(
    root,
    local_files_only=True,
    use_safetensors=True,
    dtype=torch.float32,
    attn_implementation="eager",
).eval()

cases = [
    (["Once upon a time"], 16, {}),
    (["The capital of France is"], 8, {}),
    (["你好，GPT-2!"], 12, {}),
    (["Once upon a time", "The first step is"], 12, {}),
    (["Once upon a time"], 12, {"do_sample": True, "temperature": 0.7}),
    (["Once upon a time"], 12, {"do_sample": True, "top_k": 20}),
    (["Once upon a time"], 12, {"do_sample": True, "top_p": 0.85}),
    (
        ["Once upon a time", "The first step is"],
        12,
        {"do_sample": True, "temperature": 0.8, "top_k": 40, "top_p": 0.9},
    ),
]
for prompts, count, options in cases:
    input_ids = torch.tensor([tokenizer.encode(prompt) for prompt in prompts], dtype=torch.long)
    torch.manual_seed(7)
    actual = model.generate(input_ids, max_new_tokens=count, **options)
    # Transformers 用 top_k=0 关闭 top-k。options 里的显式值会覆盖它。
    reference_options = {"top_k": 0, **options} if options else {}
    torch.manual_seed(7)
    expected = reference.generate(
        input_ids,
        attention_mask=torch.ones_like(input_ids),
        generation_config=GenerationConfig(
            max_new_tokens=count,
            use_cache=False,
            eos_token_id=None,
            pad_token_id=50256,
            **reference_options,
        ),
        use_model_defaults=False,
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    new_ids = actual[:, input_ids.size(1):]
    print(prompts, options, new_ids.tolist(), [tokenizer.decode(row) for row in new_ids.tolist()])
```

脚本里把 `attention_mask` 设成全 `1`（各行等长无填充），并用 `eos_token_id=None`、`use_model_defaults=False` 和 `use_cache=False` 让参考实现也按固定长度、无 KV cache 运行；`atol=0, rtol=0` 要求生成的每一个 token ID 逐位相等。[Transformers 的采样实现](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/generation/logits_process.py)

| 模式 | 输入 | 形状变化 | 一致的新 token |
|---|---|---|---:|
| 贪心 | `Once upon a time` | `(1, 4) → (1, 20)` | 16 / 16 |
| 贪心 | `The capital of France is` | `(1, 5) → (1, 13)` | 8 / 8 |
| 贪心 | `你好，GPT-2!` | `(1, 12) → (1, 24)` | 12 / 12 |
| 贪心 | `Once upon a time` 和 `The first step is` | `(2, 4) → (2, 16)` | 24 / 24 |
| temperature `0.7` | `Once upon a time` | `(1, 4) → (1, 16)` | 12 / 12 |
| top-k `20` | `Once upon a time` | `(1, 4) → (1, 16)` | 12 / 12 |
| top-p `0.85` | `Once upon a time` | `(1, 4) → (1, 16)` | 12 / 12 |
| temperature `0.8`、top-k `40`、top-p `0.9` | `Once upon a time` 和 `The first step is` | `(2, 4) → (2, 16)` | 24 / 24 |

Mac CPU float32 下，**贪心 60 / 60、采样 60 / 60 全部逐个一致**。比如 `Once upon a time` 的贪心续写是 `, the world was a place of great beauty and great danger. The world was`，组合采样续写是 `, we were a small band, and we didn't know`。在 Mac MPS 上同样验证了 16 个贪心 token 与 CPU 参考完全一致，以及相同种子下 24 个采样 token 两次运行完全复现。

## 4 回顾

自回归生成就是不断循环「把最近窗口送进前向 → 取最后一位的 logits → 选一个 token 拼回末尾」。贪心直接选最高分，确定但容易重复；采样依次经过 `temperature → top_k → top_p` 塑造分布后再抽签，既保留多样性，又挡住离谱的长尾词。至此，本机上的模型搭建、官方权重对齐与文本生成已全部打通，接下来进入训练阶段。

## 5 延伸问答

### Q5.1 阶段 5 是训练吗？

**A：** 是的。阶段 0 从零训练了字符级 Bigram，阶段 3、4 用官方 124M 权重验证了 GPT-2 的前向和生成。阶段 5 会先在 tiny Shakespeare 上从零训练一个约 1,080 万参数的字符级小 GPT，把数据分批、交叉熵损失、梯度累积与裁剪、验证评估和断点续训全部走通并验对，为阶段 6 的训练加速与 124M 完整预训练打基础。[阶段 5 学习笔记](05-training.md) · [项目路线图](../../docs/dev-log.md)

## 附录 术语表

| 术语 | 含义 |
|---|---|
| 自回归生成 | 每次只预测 1 个新 token，把它拼回序列末尾后再预测下一个。 |
| 贪心解码 | 每一步都直接选当前得分最高的那个 token，结果完全确定。 |
| 前缀 | 生成下一个 token 时前面已经确定的全部内容（包括提示词和已生成的词）。 |
| 上下文窗口 | 本轮前向实际送进模型的那段最近序列，长度不超过位置表行数 `P`。 |
| logits | 模型输出层直接给出的原始候选分数，还未做 softmax。 |
| temperature | 采样前用来缩放分数差距的正数：越小越保守，越大越发散。 |
| top-k | 按排名截断，只保留得分最高的 `k` 个候选。 |
| top-p | 按累计概率截断，只保留按概率从高到低累加、总和达到或超过 `p` 的最短前缀。 |
| KV cache | 生成时把前面 token 算好的注意力 K 和 V 存下来，避免每步重复算整个前缀（阶段 6 可选）。 |
