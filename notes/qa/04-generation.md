---
title: 阶段 4：文本生成问答
summary: 自回归生成、temperature、top-k 与 top-p 中的问题与回答
date: 2026-10-10
tags: [GPT-2]
status: draft
---

# 阶段 4：文本生成问答

## 导读

阶段 3 用公开的 124M 权重验收了模型的前向计算。现在把这个能力用来续写。模型一次只能给「下一个 token」打分。要写一段话，就每次选一个 token、接到已有文本后面，再预测下一次。[阶段 3 权重验收](03-weights.md#阶段-3-验收报告)

## 0 进度与成果

| 小节 | 产出 | 状态 |
|---|---|---|
| 贪心生成 | `GPT2.generate()`，每轮追加得分最高的 token | 已实现。124M 权重与参考模型逐个 token 对照通过 |
| 采样 | `do_sample=True` 启用 temperature、top-k 和 top-p | 已实现。单项和组合采样均完成对照 |

## 1 生成循环

### 这一小节要做什么

`generate(input_ids, max_new_tokens)` 接收形状为 `(B, T)` 的整数张量。`B` 是输入段数，`T` 是每段提示词的 token 数。以 `Once upon a time` 为例，tokenizer 先把这句话变成 4 个 ID。`forward()` 给这 4 个位置都算分，但我们只需要最后一组分数，因为它才是在问「整句话之后接什么」。

```text
当前序列 (B, S)
    → 取最近 P 个 token 作为窗口      (B, min(S, P))
    → GPT2.forward                   (B, min(S, P), V)
    → 取最后一个位置的分数            (B, V)
    → 贪心或采样，各选一个 token ID    (B, 1)
    → 追加到当前序列                  (B, S + 1)
```

这里 `S` 是本轮生成前的完整序列长度，开始时等于 `T`。`P` 是位置表容量，默认是 1,024。`V` 是词表大小，默认是 50,257。选出一个 ID 后，`S` 加 1，再用更新后的序列开始下一轮。[阶段 2 模型结构](02-model.md#完整模型如何输出-logits)

- 批次和每段提示词至少有一个元素。编号张量使用 `torch.int32` 或 `torch.int64`，与模型位于同一设备。同批各段等长，每个位置都作为上下文。
- `max_new_tokens` 是非负整数。返回形状是 `(B, T + max_new_tokens)`，包含原提示词，编号类型与输入一致。传 0 时直接得到输入，跳过模型前向。
- 默认 `do_sample=False`，每轮取最高分。设为 `True` 后才处理 temperature、top-k 和 top-p。三个采样参数都以关键字传入。
- 这一版按指定长度生成。每次只把位置表能容纳的最近一段送进前向，返回值仍保留原提示词和所有生成的新 ID。

### 代码与逐行解释

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

1. `@torch.no_grad()` 让 PyTorch 在这次调用中跳过梯度记录。生成只需要模型的前向分数。模型的训练或评估模式由调用方设置，下面实测使用 `model.eval()`。
2. 签名中的 `*` 表示后面的参数须按名字传入。函数入口检查张量形状、编号类型和生成长度。`numel() == 0` 同时覆盖空批次和空提示词。空提示词没有最后一个位置，无法执行 `logits[:, -1, :]`。启用采样时再检查三个采样参数。
3. `self.embeddings.wpe.num_embeddings` 直接读位置表的行数。小模型和完整 124M 模型用各自的位置表，不必在这里再写一个固定的 1,024。
4. `range(max_new_tokens)` 控制循环轮数，每一轮为每个批次成员生成一个 token。
5. `input_ids[:, -max_context:]` 在长度方向取最近一段。假设位置表有 4 行，当前序列是 `[0, 2, 3, 4, 5]`，下一轮只拿 `[2, 3, 4, 5]` 去算，位置编号在这个窗口里从 0 开始。原序列继续保存在 `input_ids` 中。
6. `self(context)` 调用模型前向。窗口中每个位置都得到一组词表分数，形状是 `(B, 窗口长度, V)`。
7. `logits[:, -1, :]` 保留整个批次，只取每一行最后一个位置，然后保留它的全部候选分数，形状是 `(B, V)`。
8. 贪心分支用 `argmax(dim=-1, keepdim=True)` 沿词表找出最大分数的编号。分数相同取第一个最大值，等同于取并列候选中编号最小的 token。采样分支将 `(B, V)` 的分数交给 `_sample_next_token`。两条分支都返回 `(B, 1)`，可以沿长度方向追加一列。[采样分支的实现](#2-采样)
9. `next_ids.to(input_ids.dtype)` 让新编号与原编号使用同一种整数类型。`torch.cat(..., dim=1)` 得到更长的张量。它保留了完整提示词，新张量会作为下一轮的输入。
10. 循环结束后，`return input_ids` 返回完整的 ID 序列。把要看的部分交给 tokenizer 的 `decode`，就能读到文本。[生成源码](../../gpt2/model.py)

### 和 Bigram 的续写怎么接上

两个版本都做「取最后一个位置的预测 → 选一个 ID → 追加 → 重复」。Bigram 的分数只依赖末尾 token，用 `softmax` 和 `multinomial` 按概率抽样。GPT-2 根据整个可见窗口计算分数，可以贪心选择，也可以调整分布后抽样。比如当前分数是 `[0.2, 1.5, 0.8]`，贪心会选 ID `1`。[Bigram 的生成函数](../../warmup/bigram.py)

## 2 采样

### 三个参数控制什么

| 参数 | 作用 | 默认值与范围 |
|---|---|---|
| temperature | 将 logits 除以温度，再交给 softmax。低温放大分数差距，概率集中到前几名。高温缩小差距，分布更平。 | `1.0` 保持分数比例，支持有限正数。 |
| top-k | 只保留得分最高的 `k` 个候选，再在这个范围里抽。 | `None` 表示使用全部候选。设置时为正整数，达到词表大小也使用全部候选。 |
| top-p | 按概率从高到低累计，保留总和首次达到 `p` 的最短前缀。候选数量随这一步的分布变化。 | `1.0` 表示使用全部候选，范围是 `(0, 1]`。 |

**手算 temperature。** 对 logits `[2, 1, 0]` 做 softmax，温度为 `1` 时概率约为 `[0.665, 0.245, 0.090]`。降到 `0.5` 后，最高项约占 `0.867`。升到 `2` 后，最高项约占 `0.506`。温度改变抽中的概率，正数温度保留分数的排序。

**手算 top-k 和 top-p。** 假设候选按概率排成 `[0.55, 0.30, 0.10, 0.05]`。`top_k=2` 直接留前两项。`top_p=0.8` 同样留前两项，因为单项 `0.55` 还未达标，两项合计 `0.85` 已达标。若概率变成 `[0.81, 0.10, 0.06, 0.03]`，同样的 `top_p=0.8` 只需留第一项。过滤后再归一化。第一种分布保留两项时，实际抽样概率约为 `[0.647, 0.353, 0, 0]`。[Nucleus sampling 论文](https://arxiv.org/abs/1904.09751)

三个参数一起用时，顺序为 **temperature → top-k → top-p → 抽样**。top-p 累计的是已经过温度调整、且经过 top-k 保留下来的候选的归一化概率。候选 ID 的顺序最终恢复为词表顺序，抽到的下标就是下一 token 的原始 ID。

例如 `model.generate(ids, 12, do_sample=True, temperature=0.8, top_k=40, top_p=0.9)` 会对每个生成位置重新计算这一轮的候选和概率。`top_k=1` 可将采样限制到一个最高分候选。贪心模式的并列行为由 `argmax` 决定，top-k 边界并列时由 `topk` 在并列候选中选出需要的数量。

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

1. 传入 `logits` 形状为 `(B, V)`，每行对应一段提示词的下一个位置。`float16` 和 `bfloat16` 的分数升为 `float32`，用更高精度算概率与累计概率。
2. 对温度非 `1` 的情况，`amax(..., keepdim=True)` 取每行最大分数，结果为 `(B, 1)`。同行所有分数减去这个数后最大值变为 `0`，softmax 的比例不变。再除以温度。`masked_fill(centered == 0, 0.0)` 让最高项保持为 `0`，极低温时也至少留一个可归一化的分数。
3. `topk(...)` 返回保留的分数和值所在的词表下标，形状都是 `(B, K)`。`full_like(..., -inf)` 创建形状为 `(B, V)` 的空候选表。`scatter_` 按词表 ID 写回保留的分数，剩余分数为负无穷。
4. `sort(..., descending=True)` 分别保存排序后的分数和各项原来的 ID。排序后做 `softmax`，再用 `cumsum` 沿候选维得到累计概率 `(B, V)`。例如 `[0.55, 0.30, 0.10, 0.05]` 的累计值是 `[0.55, 0.85, 0.95, 1.00]`。
5. `remove` 的第一列是 `False`。其余位置看的是**前一个位置**的累计值。以 `p=0.8` 为例，第二项前的和为 `0.55`，第二项仍被保留。第三项前的和已经是 `0.85`，第三项起才被屏蔽。这一步也保留恰好让累计概率等于阈值的候选。
6. `masked_fill(remove, -inf)` 屏蔽排在末尾的候选。再用 `scatter_` 按 `sorted_ids` 把分数放回原词表位置，形状仍是 `(B, V)`。
7. 最后的 `softmax` 让每行概率重新加起来等于 `1`，负无穷对应概率 `0`。`torch.multinomial(..., num_samples=1)` 按每行的概率抽一个 ID，返回 `(B, 1)`，交给外层生成循环追加。

## 3 验收

### 单元测试

- 贪心部分的 7 项测试覆盖批量生成、上下文窗口、保留原提示词、生成 0 个 token、整数类型和非法输入。
- 新增 9 项测试覆盖默认概率、温度缩放与极端温度、top-k 的候选数量、top-p 的阈值边界、组合顺序、固定种子的采样复现、默认贪心和无效参数。
- 用仓库外的 Python 列表版循环复核了这 16 项。全量 **89 项测试通过**。[生成测试](../../tests/test_gpt2_generate.py)

### 用官方 124M 权重与参考模型逐个对照

两边使用同一份本地权重、同一组输入 ID 和同样的生成长度。参考用 Transformers 4.57.6 的 `generate()`。采样时也传入同样的设置，两边每次调用前都把随机种子重设为 `7`。样例都在模型上下文容量内。超长窗口行为由前面的单元测试和独立循环复核。[权重与参考环境准备](03-weights.md#2-用官方权重对照输出)

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

`attention_mask` 全为 1，因为这几组输入的每一行都等长且没有填充。`eos_token_id=None` 和 `use_model_defaults=False` 让参考模型也按固定长度生成，`use_cache=False` 关闭参考模型的 KV cache。断言比较整个输出张量，生成部分每一列的 ID 都必须完全相等。切片 `actual[:, input_ids.size(1):]` 只取续写部分，便于读结果。[Transformers 的采样实现](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/generation/logits_process.py)

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

Mac CPU float32 上，**贪心 60 / 60，采样 60 / 60**。例如 `Once upon a time` 贪心续写为 `, the world was a place of great beauty and great danger. The world was`，组合采样续写为 `, we were a small band, and we didn't know`。

Mac MPS 用 124M 权重贪心生成的 16 个 ID 与 CPU 参考结果一致。组合采样用同一随机种子在 MPS 上运行两次，24 个新 ID 完全复现。以上采样结果对应各自的设备、随机种子和运行设置。

## 4 回顾

训练好的模型每轮根据当前前缀给下一个 token 打分。贪心直接选最高分。采样先用 temperature 改变分布，用 top-k 限制候选数量，再用 top-p 确定累计概率范围，最后抽一个 token 接到原序列。阶段 5 将用小模型走通训练流程。

## 5 延伸问答

### Q5.1 阶段 5 是训练吗？

**A：** 是。阶段 5 会先在 tiny Shakespeare 上训练一个小型 GPT，走通数据分批、交叉熵损失、反向传播、参数更新、验证和断点续训。用小模型确认训练流程后，阶段 6 再做 124M 的从零预训练和加速实验。[阶段 5 学习笔记](05-training.md) · [项目路线图](../../docs/dev-log.md)

## 附录 术语表

| 术语 | 含义 |
|---|---|
| 自回归生成 | 每次预测一个新 token，生成结果变成下一次预测时已知的输入。 |
| 贪心解码 | 每轮选当前分数最高的 token。 |
| 前缀 | 模型生成下一个 token 时，前面已经确定的部分。包括原提示词和已生成的内容。 |
| 上下文窗口 | 本轮前向能看到的那段 token。长度受位置表容量限制。 |
| logits | 给词表中各候选 token 的原始分数。 |
| temperature | 采样前缩放分数差距的正数。 |
| top-k | 按候选数量截断，只留得分最高的 k 项。 |
| top-p | 按累计概率截断，保留累计概率达到 p 所需的候选。 |
| KV cache | 保存之前各层注意力的键和值，让后续生成可以复用这些计算结果。 |

