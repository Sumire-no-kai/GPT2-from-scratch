---
title: 阶段 1：Tokenizer
summary: 从相邻 token 的统计与合并开始，逐步实现简单 BPE 和 GPT-2 的字节级 BPE
date: 2026-10-10
tags: [GPT-2]
status: published
---

# 阶段 1：Tokenizer

## 导读

[阶段 0 的 CharTokenizer](00-bigram.md#2-chartokenizer-把文字变成编号)给每种字符分配一个固定编号。BPE（Byte Pair Encoding）能把经常相邻出现的片段合成一个新 token。两个字符可以合并，新片段也能参与下一轮合并。合并规则记录了哪对编号对应哪个新编号，因此可以据此把新 token 展开，恢复原来的内容。

## 0 进度与成果

| 小节 | 产出 | 状态 |
|---|---|---|
| 统计与合并 | `count_pairs`、`merge_pair` | 已完成，5 项测试通过 |
| 简单 BPE | 从语料学习合并顺序，并实现编码、解码 | 已完成，文本往返验收通过 |
| GPT-2 tokenizer | 字节映射、官方词表与合并规则、参考编号对齐 | 已完成，214 个输入与参考结果一致 |

## 1 Tokenizer 的整体结构

Tokenizer 把文本变成模型使用的一串整数，也能把整数还原成文本。**词表**规定每个 token 对应哪个整数，**合并规则**规定哪些相邻单位可以拼成一个更长的 token。这个文件先用字符级 BPE 练习怎么学规则，再用官方 GPT-2 词表和合并规则处理 UTF-8 字节。

例如原文 `" Hello!"` 在 GPT-2 里会得到两个 token ID `[18435, 0]`。第一个 ID 对应的词表符号是 `"ĠHello"`，其中的 `Ġ` 表示原文开头的空格。`32` 是这个空格的字节值，`18435` 则是整个 `"ĠHello"` 的 token ID。一个是文本中字节的值，一个是词表里片段的编号。

- **字符级练习**。先用阶段 0 的 `CharTokenizer` 把文本变成字符 ID。`train_bpe` 从训练 ID 学到合并规则。`encode_bpe` 用这些规则缩短新序列，`decode_bpe` 把它拆回字符 ID。最后可交给 `CharTokenizer` 还原原文。[阶段 0 字符编码](00-bigram.md#2-chartokenizer-把文字变成编号)
- **GPT-2 字节级 tokenizer**。初始化时读入官方词表、合并顺序和字节映射。编码时把文本切成片段，每段转为字节、映射成符号、做 BPE 合并，然后查词表拿到 ID。解码时按 ID 查回符号，拼成字节，再还原成文本。

| 所在部分 | 函数 | 负责什么 |
|---|---|---|
| 字符级练习 | [`count_pairs`](#count_pairs-怎样数全相邻对) | 给当前 ID 序列里的相邻对计数。 |
| 字符级练习 | [`merge_pair`](#merge_pair-怎样决定下一步走几格) | 执行一条合并规则，把选定的相邻对替换成一个新 ID。 |
| 字符级练习 | [`train_bpe`](#3-学习-bpe-合并规则) | 反复计数和合并，从训练序列学出有顺序的规则表。 |
| 字符级练习 | [`encode_bpe`](#4-用学到的规则编码) | 按学习顺序应用规则，把基础 ID 合并成更短的序列。 |
| 字符级练习 | [`decode_bpe`](#5-把合并后的-id-拆回去) | 倒着应用规则，把合并出的 ID 逐层拆开。 |
| GPT-2 | [`bytes_to_unicode`](#6-gpt-2-的字节映射) | 为 256 个字节值分配可逆的单字符表示。 |
| GPT-2 | [`GPT2Tokenizer.__init__`](#读取官方文件并建表) | 读取官方文件，建立词表、字节映射和合并优先级。 |
| GPT-2 | [`GPT2Tokenizer._bpe`](#_bpe-合并单个片段) | 在一个片段里选出当前最高优先级的相邻对并合并，直到没有下一对。 |
| GPT-2 | [`GPT2Tokenizer.encode`](#encode-从文本到-token-id) | 把文本切片，逐段转字节、合并并查词表。 |
| GPT-2 | [`GPT2Tokenizer.decode`](#decode-从-token-id-到文本) | 按 ID 找回符号和字节，恢复完整的 UTF-8 文本。 |

下面每节的代码均来自同一个模块。[gpt2/tokenizer.py](../../gpt2/tokenizer.py)

## 2 统计和合并相邻对

### 概念

假设当前的 token 编号序列是 `[1, 1, 1]`。位置 0 和 1 组成一对，位置 1 和 2 又组成一对，所以 `(1, 1)` 出现了两次。执行合并时，从左边先把前两个 `1` 换成新编号 `2`。中间位置已经用过，剩下最后一个 `1`，结果为 `[2, 1]`。统计看的是真实存在的每个相邻位置，合并则决定哪些位置能一起替换。

### 练习接口

两个函数都接收一条 token 编号序列。

| 函数 | 行为 |
|---|---|
| `count_pairs` | 统计每种有序相邻对的次数。`(1, 2)` 和 `(2, 1)` 分开统计，重叠位置也计数。空序列和单个编号返回 `{}`。 |
| `merge_pair` | 从左到右，用调用方提供的新编号替换每个匹配且未被消耗的相邻对。保留其余编号的顺序，始终返回新列表，输入列表保持原样。 |

对 `[1, 2, 1, 2, 3]` 统计得到 `{(1, 2): 2, (2, 1): 1, (2, 3): 1}`。当选定 `(1, 2)` 并分配新编号 `4` 时，合并结果是 `[4, 4, 3]`。[训练函数](#3-学习-bpe-合并规则)会根据统计结果选出要合并的相邻对。

### 沿着实现走一遍

用同一条输入 `[2, 1, 1, 1]` 走一遍，可以看到两个循环的步长为什么不同。[gpt2/tokenizer.py](../../gpt2/tokenizer.py)

#### `count_pairs` 怎样数全相邻对？

```python
# 为 BPE 挑选要合并的相邻对提供频次。将原列表与从第二项开始的切片逐项配对，
# 再用每一对 ID 作字典键累计次数。
def count_pairs(ids: list[int]) -> dict[tuple[int, int], int]:
    """统计一维编号序列里每种有序相邻对的出现次数。

    每个相邻位置都参与统计，允许相邻对共享位置。
    少于两个编号时返回空字典。
    """
    counts: dict[tuple[int, int], int] = {}
    # 右侧序列错开一格，相邻两次配对可以共用中间的编号。
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts
```

1. 函数签名里的 `ids: list[int]` 表示输入是一串整数 ID。返回的 `dict[tuple[int, int], int]` 可以用两个相邻编号查到它们的出现次数。`counts` 一开始为空，之后用每个左右编号对作键，用这个组合出现的次数作值。
2. `ids[1:]` 从第二个编号开始取，得到 `[1, 1, 1]`。`zip` 每次从原列表和这个切片各取一个值，配成元组。两边错开一格，得到的恰好是原序列的左右邻居。右边先耗尽，循环自然结束。
3. 第一轮看到 `(2, 1)`，`counts.get(pair, 0)` 取到默认值 0，加一后写入 1。接下来连续两轮都看到 `(1, 1)`，这个键的值先变成 1，再变成 2。中间那个 `1` 先当右邻居，再当左邻居，两次都应计数。
4. 最终返回 `{(2, 1): 1, (1, 1): 2}`。对空列表或单个编号，右侧切片为空，循环一次都不执行，直接返回 `{}`。

#### `merge_pair` 怎样决定下一步走几格？

假设调用方已经选中 `pair = (1, 1)`，给它分配新编号 `new_id = 3`。`merged` 从空列表开始，`i = 0` 从第一项扫描，`n` 记录这一轮的输入长度。每次进入循环时，`i` 指向第一个还没处理的输入编号，`merged` 已经装好了 `ids[:i]` 的合并结果。

```python
# 执行一条 BPE 合并规则。从左往右扫描，匹配时写入新 ID 并消耗两个位置，
# 其余时候复制当前 ID 并前进一格。
def merge_pair(
    ids: list[int], pair: tuple[int, int], new_id: int
) -> list[int]:
    """从左到右将指定的相邻对替换成新的编号。

    每个原始位置最多参与一次合并。未匹配的编号保持原顺序。
    new_id 由调用方分配。返回新列表，保留输入列表原样。
    """
    merged: list[int] = []
    i = 0
    n = len(ids)

    while i < n:
        # i 指向第一个还没处理的输入位置，merged 里的新编号不参与本轮匹配。
        if i + 1 < n and (ids[i], ids[i + 1]) == pair:
            merged.append(new_id)
            i += 2  # 两个输入位置都已用掉，重叠的匹配只能取左边一次。
        else:
            merged.append(ids[i])
            i += 1  # 只用掉当前位置，右邻居仍有机会作为下一对的起点。

    return merged
```

| 进入循环时的 `i` | 从这个位置看到了什么 | 写入结果 | 下一个 `i` |
|---|---|---|---|
| 0 | `(2, 1)` 不匹配。只处理 `2`，右边的 `1` 留给下一轮 | `[2]` | 1 |
| 1 | `(1, 1)` 匹配，两个位置一起用掉 | `[2, 3]` | 3 |
| 3 | 只剩最后一个 `1`，原样保留 | `[2, 3, 1]` | 4 |

`i + 1 < n` 检查右邻居是否存在。到最后一个位置时，这个条件为假，Python 的 `and` 会跳过右侧表达式，所以不会访问越界的 `ids[i + 1]`。当 `i` 到达输入长度，循环结束，返回新列表。整轮只从 `ids` 读取，新写入的 `3` 要等到下一轮 BPE 才有机会和邻居配对。

回看统计结果，`(1, 1)` 出现了两次，实际只合并了一次，因为这两对共享一个位置。下一轮训练要在 `[2, 3, 1]` 上重新统计，此时相邻对已经变成 `(2, 3)` 和 `(3, 1)`。选择合并哪一对、分配新编号、记录规则，都是后续训练循环的职责。

### 验收

[验收用例](../../tests/test_bpe_pairs.py)覆盖相邻对的顺序、重叠计数、从左到右合并、输入列表的保留，以及没有相邻对或没有匹配时的行为。在仓库根目录用项目环境运行这 5 项测试。

```bash
python -m unittest discover -s tests -p 'test_bpe_pairs.py' -v
```

## 3 学习 BPE 合并规则

### 目标与接口

`train_bpe` 把[相邻对统计和合并](#2-统计和合并相邻对)串成循环。每轮统计当前序列，选出最高频对，给它一个新编号，并把它在序列中能合并的位置一起替换。一轮可以生成一个后续轮次还能继续合并的 token。

| 参数或返回值 | 约定 |
|---|---|
| `ids` | 已转成 token ID 的一条训练序列。训练后原列表保留原样。 |
| `num_merges` | 最多学习几条规则。传 `0` 得到空字典，负数抛出 `ValueError`。序列没有相邻对时提前结束。 |
| `start_id` | 第一个新 token ID，应大于基础词表的所有 ID。函数会拒绝小于或等于输入最大 ID 的值。之后每轮递增一。 |
| 返回值 | `{(左 ID, 右 ID): 新 ID}`，遍历字典可得到规则学到的先后顺序。 |

### 沿着实现走一遍

实现位于 [gpt2/tokenizer.py](../../gpt2/tokenizer.py)。

```python
# 从训练序列中学出编码时可复用的合并规则。每轮选出最高频相邻对并分配新 ID，
# 在合并后的序列上重新统计，最终按学习顺序返回规则。
def train_bpe(
    ids: list[int], num_merges: int, start_id: int
) -> dict[tuple[int, int], int]:
    """在一条编号序列上学习至多 num_merges 条 BPE 合并规则。

    start_id 是第一个新编号，应大于基础词表中的所有编号。
    频次并列时，选当前序列里最先出现的相邻对。
    返回按学习顺序插入的规则字典，保留原始输入列表。
    """
    if num_merges < 0:
        raise ValueError("num_merges must be non-negative")
    if ids and start_id <= max(ids):
        raise ValueError("start_id must be greater than all input token IDs")

    rules: dict[tuple[int, int], int] = {}
    for new_id in range(start_id, start_id + num_merges):
        counts = count_pairs(ids)
        if not counts:
            break

        # count_pairs 按首次出现顺序插入，max 在频次并列时保留第一个。
        pair = max(counts, key=lambda pair: counts[pair])
        rules[pair] = new_id
        ids = merge_pair(ids, pair, new_id)

    return rules
```

1. `num_merges < 0` 会报错，因为它表示允许的合并轮数。`ids and ...` 让空输入跳过 `max(ids)`，有输入时则检查新编号必须超过当前最大 ID。由调用方传入 `start_id`，可以给整张基础词表留出编号。例如基础词表使用 `0` 到 `99`，就传 `100`，即使这份语料只用到了其中几个 ID。
2. `rules` 从空字典开始。`range(start_id, start_id + num_merges)` 依次给每轮提供一个新编号。进入一轮先用 `count_pairs(ids)` 统计当前序列。如果 `counts` 为空，说明没有相邻位置可合并，用 `break` 提前结束。
3. `max(counts, key=lambda pair: counts[pair])` 遍历字典的键，用每对的出现次数来比较。并列时取遍历中第一个。因为 `count_pairs` 从左往右扫描，并在一对首次出现时才插入字典，这个顺序就是首次出现顺序。例如 `[5, 2, 3]` 的两对各出现一次，先选 `(5, 2)`。
4. 把选中的对和新编号写入 `rules`。接着让局部变量 `ids` 指向 `merge_pair` 返回的新列表。上一轮的替换会改变相邻关系，所以进入下一轮时要在这个新列表上重新统计。
5. 返回按学习顺序写入的字典。后续编码可以按这个顺序对新文本应用规则，解码也能从规则还原新编号对应的原始片段。

用 `train_bpe([1, 2, 1, 2, 1, 2], num_merges=2, start_id=3)` 走一遍。

| 轮次 | 当前序列 | 选中的相邻对 | 记下的规则 | 合并后 |
|---|---|---|---|---|
| 1 | `[1, 2, 1, 2, 1, 2]` | `(1, 2)`，出现 3 次 | `(1, 2) → 3` | `[3, 3, 3]` |
| 2 | `[3, 3, 3]` | `(3, 3)`，出现 2 次 | `(3, 3) → 4` | `[4, 3]` |

最终返回 `{(1, 2): 3, (3, 3): 4}`。如果基础词表里 `1 = a`、`2 = b`，第一个新 token 就代表 `ab`，第二个代表 `abab`。第二条规则用到的 ID `3` 来自第一轮。

### 验收

测试覆盖每轮重新计数、规则顺序、并列频次、无可用相邻对时提前结束、零轮数和非法参数。[训练验收用例](../../tests/test_bpe_training.py)

```bash
python -m unittest discover -s tests -p 'test_bpe_training.py' -v
```

## 4 用学到的规则编码

`encode_bpe(ids, rules)` 接收基础词表中的 token ID 和 [`train_bpe` 学到的规则](#3-学习-bpe-合并规则)。训练阶段通过频次决定合并哪些对，编码阶段把已有规则用到新序列上。

### 沿着实现走一遍

`encode_bpe` 逐条应用规则，返回编码后的 ID 列表。[gpt2/tokenizer.py](../../gpt2/tokenizer.py)

```python
# 把训练时学到的 BPE 规则按顺序应用到一条新的编号序列。
# 每条规则都接着处理上一条规则的合并结果。
def encode_bpe(
    ids: list[int], rules: dict[tuple[int, int], int]
) -> list[int]:
    """按学习顺序应用合并规则，返回编码后的新列表。

    ids 是基础词表中的 token ID 序列，rules 来自 train_bpe。
    输入列表和规则字典保持原样。没有规则时返回输入列表的副本。
    """
    encoded = list(ids)
    for pair, new_id in rules.items():
        encoded = merge_pair(encoded, pair, new_id)
    return encoded
```

1. `list(ids)` 给当前结果做一份副本。即使 `rules` 为空，最后返回的也是一个独立列表。
2. `rules.items()` 按字典的插入顺序给出键和值。这里的键 `pair` 是要合并的两个 ID，值 `new_id` 是它们合并后的 ID。这个遍历顺序与训练时写入规则的顺序相同。
3. 每轮把当前 `encoded` 交给 [`merge_pair`](#2-统计和合并相邻对)，再用返回的新列表更新 `encoded`。例如第一条规则把 `[1, 2, 1, 2]` 变成 `[3, 3]`，第二条此时才能把两个 `3` 变成 `4`。
4. 循环结束后返回最终结果。输入列表和规则字典都保留原样。

### 验收

| 输入 ID | 规则 | 预期结果 |
|---|---|---|
| `[1, 2, 1, 2]` | `{(1, 2): 3, (3, 3): 4}` | 先得到 `[3, 3]`，再得到 `[4]` |
| `[1, 2]` | `{}` | `[1, 2]` 的副本 |
| `[2, 1]` | `{(1, 2): 3}` | `[2, 1]`，原顺序保留 |
| `[]` | `{(1, 2): 3}` | `[]` |
| `[1, 4, 5]` | `{(4, 5): 6, (1, 6): 7}` | 先得到 `[1, 6]`，再得到 `[7]` |

最后一行专门检查规则顺序。如果按编号把 `(1, 6)` 排到前面，应用它时序列里还没有 `6`，最终只会得到 `[1, 6]`。

## 5 把合并后的 ID 拆回去

`decode_bpe(ids, rules)` 接收[编码后的 ID](#4-用学到的规则编码)和同一套合并规则，返回展开后的基础 ID。之后可交给阶段 0 的 `CharTokenizer.decode` 还原成字符串。

训练时先学到 `(1, 2) → 3`，再学到 `(3, 3) → 4`。解码时先把 `4` 展开成两个 `3`，再把每个 `3` 展开成 `1, 2`。如果先处理 `3` 的规则，输入 `[4]` 里还没有任何 `3`。所以这一节从最后学到的规则开始，每轮扫描当前列表，把命中的 ID 用对应的两个 ID 替换，其他 ID 按原顺序保留。

### 沿着实现走一遍

`decode_bpe` 接收编码后的 ID 列表，返回原来的基础 ID 序列。[gpt2/tokenizer.py](../../gpt2/tokenizer.py)

```python
# 把合并出的 token ID 展开回基础 ID。逆着学习顺序拆开每条规则，
# 这样拆出的中间 ID 还能继续用较早的规则展开。
def decode_bpe(
    ids: list[int], rules: dict[tuple[int, int], int]
) -> list[int]:
    """将同一套规则编码后的 ID 序列还原为基础 ID 的新列表。

    rules 来自 train_bpe，基础 ID 在拆解时原样保留。
    输入列表和规则字典保持原样。没有规则时返回输入列表的副本。
    """
    decoded = list(ids)
    for pair, new_id in reversed(rules.items()):
        next_decoded: list[int] = []
        for token_id in decoded:
            if token_id == new_id:
                next_decoded.extend(pair)
            else:
                next_decoded.append(token_id)
        decoded = next_decoded
    return decoded
```

1. `list(ids)` 复制输入，空规则时也能返回独立的列表。
2. `reversed(rules.items())` 从最后写入的规则开始，依次给出原始编号对 `pair` 和合并后的 `new_id`。
3. 内层循环扫描当前列表，把这一轮的结果写进 `next_decoded`。命中时，`extend(pair)` 把左、右两个 ID 分别加入列表，保持结果为一维 ID 序列。其余位置用 `append(token_id)` 写入原来的单个编号。
4. 一轮结束后，`decoded = next_decoded` 把展开后的序列交给下一轮。等所有规则都处理完，就得到基础 ID 列表。

### 验收

下表使用规则 `{(1, 2): 3, (3, 3): 4}`，空规则的用例单独标明。

| 输入 ID | 预期结果 |
|---|---|
| `[4]` | `[1, 2, 1, 2]` |
| `[4, 2, 3]` | `[1, 2, 1, 2, 2, 1, 2]` |
| `[]` | `[]` |
| `[2, 1]`，规则为 `{}` | `[2, 1]` 的副本 |

四个解码用例通过。每次返回独立列表，输入列表和规则字典保留原样。

再用 `CharTokenizer("ababab")` 建立字符词表，得到 `a → 0`、`b → 1`。在 `ababab` 的字符 ID 上训练两轮，从 `vocab_size = 2` 开始分配新编号，得到 `(0, 1) → 2` 和 `(2, 2) → 3`。拿新文本 `abab` 跑完整流程，得到 `"abab" → [0, 1, 0, 1] → [3] → [0, 1, 0, 1] → "abab"`。中文、emoji 和空字符串的往返也通过。

## 6 GPT-2 的字节映射

前面的字符级 BPE 以 Python 字符为起点。例如 `"你"` 一开始是一个基础 token。GPT-2 先按规则切开原文，再把每段转成 UTF-8 字节。`"你"` 对应 `E4 BD A0` 三个字节，合并从这三个单位开始。前面练习的[相邻对合并](#2-统计和合并相邻对)仍然适用，GPT-2 编码时按官方合并表的优先级选择当前可合并的对。[GPT-2 官方编码器](https://github.com/openai/gpt-2/blob/master/src/encoder.py#L50-L96)

官方实现用单个 Unicode 字符表示每个字节，这样可以用字符串处理和保存合并规则，也能避开实际的空白和控制字符。例如空格字节 `32` 映射到 `Ġ`，所以 `" a"` 转为 UTF-8 后，逐字节查表得到 `"Ġa"`，随后才进入 BPE。因为每个映射字符只对应一个字节，解码时可以查反向表，找回原来的字节序列。[官方字节映射](https://github.com/openai/gpt-2/blob/master/src/encoder.py#L6-L26)

### 目标与接口

`bytes_to_unicode() -> dict[int, str]` 无参数，返回一张完整的字节映射表。[函数实现](../../gpt2/tokenizer.py)

1. 对 `33–126`、`161–172`、`174–255` 三个闭区间，字节值 `b` 直接映射为 `chr(b)`。`chr` 根据整数码点生成一个 Unicode 字符。
2. 剩下的字节按从小到大的顺序处理。第一个映射到 `chr(256)`，下一个到 `chr(257)`，依次递增。
3. 返回的字典包含 `0–255` 的每个整数键，对应 256 个互不重复的单字符字符串。

### 沿着实现走一遍

```python
# 为 GPT-2 的 256 个字节值分配可逆的单字符表示。
# 能直接保留的字节沿用同编号字符，其余按字节值顺序分配新字符。
def bytes_to_unicode() -> dict[int, str]:
    """建立与 GPT-2 官方编码器一致的字节到字符映射。

    键覆盖 0 到 255，值是互不重复的单个 Unicode 字符。
    """
    direct_ranges = (range(33, 127), range(161, 173), range(174, 256))
    mapping = {byte: chr(byte) for interval in direct_ranges for byte in interval}

    next_codepoint = 256
    for byte in range(256):
        if byte not in mapping:
            mapping[byte] = chr(next_codepoint)
            next_codepoint += 1
    return mapping
```

先用三个区间建立能直接对应的键值。推导式先遍历区间，再遍历其中的字节。`range` 的右端点不包含在范围内，所以第一段写 `range(33, 127)`。第二遍按升序扫描所有字节，每遇到一个尚未入表的值，就分配当前码点，再把码点加一。

### 验收

| 字节值 | 映射字符 | 检查点 |
|---|---|---|
| `0` | `Ā`（U+0100） | 第一个补位字符 |
| `10` | `Ċ`（U+010A） | 换行使用替代字符 |
| `32` | `Ġ`（U+0120） | 空格使用替代字符 |
| `33` | `!` | 第一段保留区间起点 |
| `160` | `ł`（U+0142） | 第二段保留区间之前 |
| `161` | `¡` | 第二段保留区间起点 |
| `173` | `Ń`（U+0143） | 最后一个补位字符 |
| `255` | `ÿ` | 第三段保留区间终点 |

最后检查整张表能否反过来查。比如从 `" a"` 的 UTF-8 字节序列 `[32, 97]` 得到 `"Ġa"`，再逐字符查反向表，应得到同样的 `[32, 97]`。

## 7 接上 GPT-2 官方词表和合并规则

`GPT2Tokenizer` 从本地的 `encoder.json` 和 `vocab.bpe` 构建。当前工作区的文件在 `data/gpt2/`。从仓库根目录运行时，可以直接把下面两个路径传给构造函数。[官方词表](https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json) · [官方合并表](https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe)

```python
from gpt2.tokenizer import GPT2Tokenizer

tokenizer = GPT2Tokenizer("data/gpt2/encoder.json", "data/gpt2/vocab.bpe")
assert tokenizer.encode(" Hello!") == [18435, 0]
assert tokenizer.decode([18435, 0]) == " Hello!"
```

### 模块里的依赖和常量

```python
import json
from pathlib import Path

import regex


# 预切分模式依据 GPT-2 官方编码器的 src/encoder.py。
_GPT2_PATTERN = regex.compile(
    r"'(?:s|t|re|ve|m|ll|d)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)
_ENDOFTEXT = "<|endoftext|>"
```

- `import json` 供构造函数把词表文件的文本读成 Python 字典。`Path` 负责打开调用方传来的本地路径。
- `import regex` 提供 GPT-2 预切分需要的 Unicode 类别。`regex.compile(...)` 在模块加载时构建一次模式，后续每次编码都复用它。模式前的 `r` 让反斜线按原样交给正则处理。
- `_ENDOFTEXT` 保存结束标记的文本形式。编码时用它找出标记的位置，并查官方词表拿到它的 ID。

预切分先在原文本上执行，得到的片段稍后才转成 UTF-8 字节。正则里的 `|` 表示从左到右尝试以下分支。[GPT-2 官方编码器](https://github.com/openai/gpt-2/blob/master/src/encoder.py#L38-L48)

| 模式片段 | 在原文本上会匹配什么 |
|---|---|
| `'(?:s\|t\|re\|ve\|m\|ll\|d)` | 英文缩写后缀，例如 `I'm` 切成 `I` 和 `'m`。 |
| ` ?\p{L}+` | 连续的 Unicode 字母，前面可以带一个普通空格。中文文字也属于字母类别。 |
| ` ?\p{N}+` | 连续的 Unicode 数字，前面可以带一个普通空格。 |
| ` ?[^\s\p{L}\p{N}]+` | 连续的标点、符号等，排除空白、字母和数字。这段也可以带前导空格。 |
| `\s+(?!\S)` | 匹配一串空白，并要求匹配结束后接着空白或已经到达结尾。比如 `"Hi  你好!"` 中，第一个空格单独成段，第二个跟着 `你好`。 |
| `\s+` | 收下其余连续空白，例如前面各分支都没接住的换行。 |

### 读取官方文件并建表

下面是类声明和完整的 `__init__`。调用 `GPT2Tokenizer(...)` 时，它会先把后续编解码要用的查找表装到实例上。

```python
class GPT2Tokenizer:
    """从官方词表和合并规则构建的 GPT-2 字节级 tokenizer。"""

    # 读取词表和合并顺序，预先建好编码与解码用的查找表。
    def __init__(
        self, encoder_path: str | Path, merges_path: str | Path
    ) -> None:
        self.encoder: dict[str, int] = json.loads(
            Path(encoder_path).read_text(encoding="utf-8")
        )
        self.decoder = {token_id: symbol for symbol, token_id in self.encoder.items()}
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {symbol: byte for byte, symbol in self.byte_encoder.items()}

        lines = Path(merges_path).read_text(encoding="utf-8").splitlines()
        merge_lines = (line for line in lines[1:] if line.strip())
        self.bpe_ranks: dict[tuple[str, str], int] = {}
        for rank, line in enumerate(merge_lines):
            left, right = line.split()
            self.bpe_ranks[left, right] = rank
```

1. 签名里的 `encoder_path`、`merges_path` 分别指词表文件和合并文件。可以传字符串路径，也可以传 `Path`。加载出的表保存在 `self` 上，后面的三个方法可以直接使用。
2. `Path(encoder_path).read_text(encoding="utf-8")` 读取词表文本。`json.loads(...)` 将它解析成 `self.encoder`，键是合并后的符号串，例如 `"ĠHello"`，值是对应的 token ID，例如 `18435`。
3. `self.decoder` 把 `self.encoder.items()` 的键和值交换。这样编码时能通过符号找 ID，解码时也能通过 ID 反查符号。
4. `self.byte_encoder = bytes_to_unicode()` 建立[字节到单字符符号的映射](#6-gpt-2-的字节映射)。下一行再反过来存成 `self.byte_decoder`。例如正向表把 `32` 查成 `Ġ`，反向表把 `Ġ` 查回 `32`。
5. 读取合并文件时，`splitlines()` 按行切开文本。`lines[1:]` 跳过文件头的版本行，`if line.strip()` 跳过空白行。剩下每一行都表示一条规则。
6. `self.bpe_ranks` 从空字典开始。`enumerate(merge_lines)` 让第一条规则的优先级序号为 `0`。`left, right = line.split()` 将一行拆成左右两个符号，最后用这两个符号组成的元组作键，把 `rank` 存进去。**rank 决定先合并哪一对，token ID 决定合并结果查词表时得到哪个号码。**

### `_bpe` 合并单个片段

这个方法收到的是**一段文本映射后的字节符号**，例如 `"ĠHello"`。它返回合并后的符号列表，交给 `encode` 查词表。

```python
    # 在单个预切分片段里反复合并优先级最高的相邻符号。
    def _bpe(self, symbols: str) -> list[str]:
        """返回片段合并后的符号列表。合并表里序号越小，优先级越高。"""
        if len(symbols) < 2:
            return list(symbols)

        word = list(symbols)
        while len(word) >= 2:
            pairs = set(zip(word, word[1:]))
            # 未收录的对排在所有有效规则之后。整轮无规则时，下面直接结束。
            bigram = min(
                pairs,
                key=lambda pair: self.bpe_ranks.get(pair, float("inf")),
            )
            if bigram not in self.bpe_ranks:
                break

            first, second = bigram
            merged: list[str] = []
            i = 0
            n = len(word)
            # 只扫描本轮的输入序列，新拼出的符号从下一轮开始参与配对。
            while i < n:
                if i + 1 < n and word[i] == first and word[i + 1] == second:
                    merged.append(first + second)
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            word = merged

        return word
```

1. `if len(symbols) < 2` 处理没有相邻对的情况。`list(symbols)` 把 `""` 变成 `[]`，把 `"x"` 变成 `["x"]`，直接返回。较长的片段也用 `word = list(symbols)` 拆成单字符符号，作为第一轮的输入。
2. `while len(word) >= 2` 表示至少还有两个符号可以检查。`zip(word, word[1:])` 把当前位置和右邻居配成一对。`set(...)` 收集不同的相邻对，比如在 `["a", "a", "a"]` 中只需检查一次 `("a", "a")` 的优先级。具体有哪些位置能合并，后面的扫描会逐个判断。
3. `min(pairs, key=...)` 根据 `key` 算出的 rank 比较各对。`self.bpe_ranks.get(pair, float("inf"))` 对已知对返回它的 rank，其余对返回正无穷。任意有效 rank 都比正无穷小，因此只要有可合并的对，就会选出 rank 最小的那一对。
4. 如果所有候选都没有规则，`min` 仍会给出其中一对。随后的 `if bigram not in self.bpe_ranks` 检查会让外层循环结束，避免拿这对去合并。正式合并表里的有效 rank 互不相同，因此合法候选的选择与集合的遍历顺序无关。
5. `first, second = bigram` 把选中的一对拆成左右符号。`merged` 是这一轮的新结果，`i = 0` 从旧序列开头扫描，`n = len(word)` 记住旧序列长度。`while i < n` 让扫描持续到每个旧位置都处理完。
6. 内层的 `if` 先用 `i + 1 < n` 检查右邻居，再按顺序比较左右符号。三个条件都成立时，`first + second` 把两个字符串拼成一个符号，`merged.append(...)` 将它写入结果。`i += 2` 跳过正好被消耗的两个位置。
7. 否则，`merged.append(word[i])` 原样保留当前位置，`i += 1` 只前进一步。右邻居仍可作为下一对的起点。最后一个符号没有右邻居，也会走这个分支。
8. 内层扫描结束后，`word = merged` 把这一轮结果交给外层循环。下一轮会重新计算相邻对，因此刚拼出的符号可以继续合并。每次有效合并至少缩短一个位置。外层结束后，`return word` 返回最终的符号列表。

例如 `"abc"` 中，起初有 `("a", "b")` 和 `("b", "c")`。假设这两对的 rank 分别为 `1` 和 `0`，第一轮选后者，得到 `["a", "bc"]`。重新计算相邻对后，如果表里有 `("a", "bc")`，下一轮可以得到 `["abc"]`。对于 `"aaaaa"` 和规则 `("a", "a")`，游标依次在 `0 → 2 → 4` 处理，得到 `["aa", "aa", "a"]`。

### `encode` 从文本到 token ID

```python
    # 先按 GPT-2 的规则预切分，再逐段转字节、执行 BPE 和查词表。
    def encode(self, text: str, *, allow_endoftext: bool = False) -> list[int]:
        """编码文本。allow_endoftext 为真时，将结束标记作为特殊 token。"""
        parts = text.split(_ENDOFTEXT) if allow_endoftext else [text]
        ids: list[int] = []
        for index, part in enumerate(parts):
            if index:
                ids.append(self.encoder[_ENDOFTEXT])

            for match in _GPT2_PATTERN.finditer(part):
                chunk = match.group().encode("utf-8")
                symbols = "".join(self.byte_encoder[byte] for byte in chunk)
                ids.extend(self.encoder[token] for token in self._bpe(symbols))
        return ids
```

1. 参数 `text` 是原文，`allow_endoftext` 是一个只能用名字传入的开关。默认直接用 `[text]`，按普通文字处理结束标记。开关打开时，`text.split(_ENDOFTEXT)` 把每个标记两侧的普通文本分开，标记本身留给循环写入对应 ID。
2. `ids` 是最终输出。`enumerate(parts)` 同时拿到每段内容和它的位置。索引从 `1` 开始的每段前面都隔着一个标记，所以 `if index` 时先用 `self.encoder[_ENDOFTEXT]` 查得 `50256` 并追加。前缀、后缀或两个相邻标记之间的空片段也会保留位置，保证每个标记都得到一个 ID。
3. `_GPT2_PATTERN.finditer(part)` 依次找到这段普通文本中的预切分片段。`match.group()` 取出整个匹配结果，`.encode("utf-8")` 把它转成字节。例如 `" Hello"` 转成 `[32, 72, 101, 108, 108, 111]` 这几个字节值。
4. `for byte in chunk` 逐字节读取数值。`self.byte_encoder[byte]` 将每个字节查成对应符号，`"".join(...)` 无分隔地拼成 `symbols`，例如 `"ĠHello"`。这一步保留了片段开头的空格信息。
5. `self._bpe(symbols)` 合并当前片段的符号，每个结果用 `self.encoder[token]` 查得官方 ID。`ids.extend(...)` 按顺序把这些 ID 加进总结果。整段文本处理完后，`return ids` 返回平铺的一维编号列表。

例如 `tokenizer.encode("a<|endoftext|>b", allow_endoftext=True)` 返回 `[64, 50256, 65]`。参考工具也可以显式允许同一个特殊 token。[tiktoken 的 GPT-2 配置](https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py#L14-L27) · [tiktoken 特殊 token 选项](https://github.com/openai/tiktoken/blob/main/tiktoken/core.py#L75-L104)

### `decode` 从 token ID 到文本

```python
    # 先恢复完整字节序列，再做 UTF-8 解码，保留跨 token 的多字节字符。
    def decode(self, ids: list[int], *, errors: str = "replace") -> str:
        """把 token ID 还原成文本。errors 传给 UTF-8 解码器。"""
        symbols = "".join(self.decoder[token_id] for token_id in ids)
        raw_bytes = bytes(self.byte_decoder[symbol] for symbol in symbols)
        return raw_bytes.decode("utf-8", errors=errors)
```

1. `self.decoder[token_id]` 把每个 ID 查成词表里的符号串。首行的 `"".join(...)` 按输入顺序拼接所有符号。例如 `[18435, 0]` 查回 `"ĠHello"` 和 `"!"`，拼成 `"ĠHello!"`。词表中不存在的 ID 会在查询时抛出 `KeyError`。
2. `for symbol in symbols` 一次取一个字节映射符号。`self.byte_decoder[symbol]` 查回一个 `0–255` 的字节值，`bytes(...)` 按顺序组成完整字节序列。例如 `Ġ` 恢复成 `32`，空格因此会出现在还原的文本里。
3. `raw_bytes.decode("utf-8", errors=errors)` 将完整字节序列还原为字符串。一个中文字符的几个字节可能分布在多个 token 中，所以代码先把所有字节拼起来，再做一次 UTF-8 解码。如果调用方只传了某个多字节字符的一部分，默认 `errors="replace"` 会输出 `�`，显式传 `errors="strict"` 会抛出 `UnicodeDecodeError`。

### 验收

本地小词表单元测试覆盖字节映射、词表和 rank 加载、预切分、最小 rank 优先、合并后重新找对、重叠合并、特殊标记和多字节解码。也有直接通过真实 `_bpe` 跑编码和解码的用例。[GPT-2 tokenizer 测试](../../tests/test_gpt2_tokenizer.py)

用官方词表和本仓库的 `_bpe` 对照官方编码器及 `tiktoken 0.14.0` 的 `gpt2`。210 个普通输入和 4 个显式启用特殊标记的输入逐项得到了相同的 token ID，解码都能还原。普通输入包括空字符串、空白、英文缩写、多个语言的文字、数字和 emoji。验收用的 `tiktoken` 安装在临时目录。

## 回顾

- 字符级练习从字符 ID 学规则，编码时按学习顺序合并，解码时按反方向展开。
- GPT-2 编码从 UTF-8 字节出发。先分片，再用官方合并顺序处理每片，最后查官方词表。解码通过两张反向表把 ID 拼回字节，再一次性恢复文本。
- 整个流程可以用 `" Hello!" → ["ĠHello", "!"] → [18435, 0] → " Hello!"` 检查。每一步都把空格和标点的信息保留下来。

## 延伸问答

### Q3.1 BPE 和字节级 tokenizer 有什么区别？和现在常见的 tokenizer 又有什么区别？

**A：** BPE 描述**怎么合并**，字节级描述**从什么单位开始**。这两项可以组合，GPT-2 用的就是字节级 BPE。

| 做法 | 初始单位 | 接下来怎么处理 |
|---|---|---|
| 阶段 0 的字符编码 | Unicode 字符 | 每种字符对应一个编号 |
| 本阶段的简单 BPE | Unicode 字符 | 用学到的规则把相邻片段继续合并 |
| GPT-2 的字节级 BPE | UTF-8 字节 | 先按规则把文本切成片段，再在每个片段内对字节应用合并规则 |

例如，字符级练习从一个“你”开始，字节级方案从它的三个 UTF-8 字节 `E4 BD A0` 开始。初始字节只有 256 种，因此即使某个字符在训练语料里没出现过，也能用它的字节表示。最终“你”占几个 token，由词表和合并规则决定。

前面的[相邻对统计与合并练习](#2-统计和合并相邻对)对字符编号和字节编号都适用。学习合并规则时，`count_pairs` 提供相邻对的频次。GPT-2 编码新文本时，按已有的合并优先级执行，最后查固定词表得到 ID。这里说的“与 GPT-2 对齐”是指同一输入得到相同的 token ID 序列。[GPT-2 官方编码器](https://github.com/openai/gpt-2/blob/master/src/encoder.py)

现在的常见模型里，Meta Llama 4 和 OpenAI 的 `o200k_base` 编码仍沿用字节级 BPE。它们与 GPT-2 的差别体现在合并表、文本预切分规则、词表和特殊 token 上。例如 GPT-2 的预切分会把连续数字看作一段，而 `o200k_base` 会把数字按最多三个一组切开。这是 BPE 合并前的分段，所以即使合并方法相近，同一句话也可能切成不同数量的 token，得到的 ID 也会不同。[Meta Llama 4 tokenizer](https://github.com/meta-llama/llama-models/blob/main/models/llama4/tokenizer.py) · [OpenAI tiktoken 编码配置](https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py)

也有其他切分算法。Unigram 给候选片段分配概率，编码时根据这些概率选取切法。SentencePiece 是可以运行 BPE 和 Unigram 等算法的工具。[SentencePiece 项目说明](https://github.com/google/sentencepiece)

## 附录 术语表

| 术语 | 含义 |
|---|---|
| token | tokenizer 处理的基本单位，可以是一个字符、字节或合并出的片段。 |
| BPE | 反复合并相邻片段，从基本单位建立更长 token 的方法。 |
| 相邻对 | 编号序列中紧挨着的两个编号，顺序属于这对编号的一部分。 |
| 合并规则 | 一对相邻编号到一个新编号的映射。 |
| 合并优先级 | GPT-2 合并规则在官方文件中的顺序。序号越小，遇到该对时越优先合并。 |
| UTF-8 | 把 Unicode 文本编码成字节的方式。一个 Unicode 字符可以占 1 到 4 个字节。 |
| 字节级 BPE | 从文本的字节表示出发进行合并的 BPE。 |
| 字节映射 | GPT-2 为每个字节指定一个独立的 Unicode 字符，供后续合并和还原使用。 |
| 预切分 | 在应用合并规则前将文本划分成片段，规定哪些片段之间有边界。 |
| Unigram | 按词表中各片段的概率对可能的切法打分的分词算法。 |

