---
title: 阶段 0：Bigram 热身
summary: 从零写一个字符级 bigram 语言模型：字符编码、切分训练样本、查表打分、交叉熵、生成与训练循环，逐行讲解代码并整理学习中的问题
date: 2026-10-10
tags: [GPT-2]
status: draft
---

# 阶段 0：Bigram 热身

## 导读

这篇笔记记录用 PyTorch 从零写出一个最小语言模型的全过程：它读入莎士比亚的剧本，学会"哪个字母后面常跟哪个字母"，然后自己续写出一段"像英文"的文字。代码只有一个文件 [warmup/bigram.py](../../warmup/bigram.py)，约 200 行（不含注释）。

**需要的基础**：会一点 Python（变量、函数、类、列表、字典、for 循环）。不需要懂机器学习，也不需要懂 PyTorch，用到的东西都会在第一次出现时解释。

**怎么读**：

- 第 [1 节](#1-全景) 先看全景，知道整件事在做什么、分几步。
- 第 [2](#2-chartokenizer-把文字变成编号)–[7](#7-main-把一切串起来) 节按代码顺序，一块一块讲。每一节的结构相同：先讲概念，再贴出最终代码并逐行讲解，然后是测试检查了什么、写的时候踩过的坑，最后是学习时问过的相关问题。
- 第 [8 节](#8-回顾-编号在代码里怎么流动) 回头把数据在代码里的流动串一遍。
- 第 [9 节](#9-延伸问答) 是从这个小模型出发、聊到 GPT-2 和现在的大模型的问题。
- 遇到不懂的词，查文末的[术语表](#附录-术语表)。
- 问答按小节编号：Q3.2 表示第 3 节的第 2 个问题，下面以 A：开头的是回答。

笔记里的数字（loss、概率、生成的文字）都是实际运行得到的，不是编的。

## 0 进度与成果

| 部分 | 代码 | 状态 |
|---|---|---|
| 字符编码 | `CharTokenizer` | 已完成，7 项测试通过 |
| 切分训练样本 | `get_batch` | 已完成，5 项测试通过 |
| 模型：查表打分、算 loss | `BigramLanguageModel.forward` | 已完成，6 项测试通过 |
| 续写文本 | `BigramLanguageModel.generate` | 已完成，5 项测试通过 |
| 训练一步、评估 | `train_step`、`estimate_loss` | 已完成，5 项测试通过 |
| 串起整个流程 | `main` | 已完成，运行正常 |

最终成果：在 tiny Shakespeare 上训练 3000 步（Apple M5 的 GPU 上约 5 秒），loss 从 4.73 降到 2.46；生成的文字从完全随机的乱码，变成有剧本格式、有常见字母组合、但拼不出真词的字母串。详见[第 7 节的运行结果](#74-运行结果)。

## 1 全景

### 1.1 语言模型在做什么

语言模型只做一件事：**看到前面的文字，猜下一个是什么**。

比如看到 `hell`，猜下一个是 `o`。猜对的概率越高，模型越好。让它不断"猜下一个、接上、再猜下一个"，就能续写出一段文字。GPT-2、ChatGPT、Claude 生成回答，用的都是这个办法。

训练语言模型有一个很方便的地方：**文本本身就带着答案**。任意一段文字，每个字的"下一个字"就是原文里紧挨着它的那个，不需要人工标注。所以只要有大量文本，就能自动出无数道"猜下一个"的题。

### 1.2 为什么先做 bigram

bi 是"二"，bigram 指**两个相邻字符组成的一对**。bigram 模型**只看当前这一个字符**来猜下一个，前面的字符一概不管。

它很弱：看到 `l`，它不知道前面是 `hel` 还是 `wor`，所以分不清下一个该是 `l`、`o` 还是 `d`。但它有一个好处：模型本身极其简单（只有一张表，见[第 4 节](#4-bigramlanguagemodel-查表打分并算-loss)），可以把注意力全放在**语言模型的完整流程**上：

```text
文本 → 编号 → 训练样本 → 预测 → loss → 梯度 → 更新参数
                         ↓
                    采样并生成文字
```

这条流程在 GPT-2 里完全一样。之后换成 GPT-2，只是把"只有一张表的模型"换成"有注意力的模型"，数据怎么准备、loss 怎么算、怎么训练、怎么生成，都不用改。

### 1.3 整体流程

代码分成下面几块，每块对应一节：

```text
data/input.txt 里的莎士比亚剧本
   │  CharTokenizer.encode            第 2 节：文字 → 编号
   ▼
data：一长串编号，约 111 万个
   │  get_batch                       第 3 节：随机切出题目 x 和答案 y
   ▼
x、y：形状 (B, T)
   │  model(x, y)，即 forward          第 4 节：查表打分，对答案算 loss
   ▼
logits：形状 (B, T, V)；loss：一个数
   │  train_step                      第 6 节：反向传播，调整表里的数字
   ▼
训练好的表
   │  generate                        第 5 节：一个字一个字地续写
   ▼
新的编号 → CharTokenizer.decode → 文字

main 把以上所有步骤串起来               第 7 节
```

### 1.4 三个符号 B T V

全文反复出现三个字母，先记住它们：

| 符号 | 代码里的名字 | 意思 | 本项目的值 |
|---|---|---|---|
| B | `batch_size` | 一批有几段文字 | 32 |
| T | `block_size` | 每段有几个字符 | 8 |
| V | `vocab_size` | 词表里有几种不同的字符 | 65 |

PyTorch 里的数据都放在**张量**（tensor）里。张量可以理解成多维的数字表格，它的**形状**（shape）说明每一维有多长。比如形状 `(B, T)` 就是 B 行 T 列的表格：

```text
            第0列  第1列  第2列     ← T = 3 列（每段几个字符）
第 0 行  [    1,     0,     2  ]   ← 第 0 段
第 1 行  [    2,     3,     0  ]   ← 第 1 段
B = 2 行（一共几段）
```

一定要分清**形状**和**内容**：形状说的是"有几行几列"，格子里装的数字才是内容（这里是字符的编号）。B 是行数，不是编号。后面还会出现 `(B, T, V)` 这样的三维形状，意思是"B 行 T 列，每个格子里不是一个数，而是一串 V 个数"。

### 1.5 文件、测试和运行方式

| 文件 | 内容 |
|---|---|
| [warmup/bigram.py](../../warmup/bigram.py) | 全部代码，带详细注释 |
| [tests/test_char_tokenizer.py](../../tests/test_char_tokenizer.py) | 第 2 节的测试 |
| [tests/test_get_batch.py](../../tests/test_get_batch.py) | 第 3 节的测试 |
| [tests/test_bigram_model.py](../../tests/test_bigram_model.py) | 第 4 节的测试 |
| [tests/test_generate.py](../../tests/test_generate.py) | 第 5 节的测试 |
| [tests/test_training.py](../../tests/test_training.py) | 第 6 节的测试 |
| `data/input.txt` | 训练数据，自行下载，不进版本库（见[第 7 节](#71-数据)） |

在仓库根目录运行全部测试，以及完整的训练：

```bash
python -m unittest discover -s tests -v
python -m warmup.bigram
```

**为什么放在 `warmup/` 而不是正式的 GPT-2 目录？** bigram 本身不属于 GPT-2，它是热身练习。和正式代码分开，正式代码就不会混进练习用的东西。正式的 GPT-2 包从阶段 1 开始建，这里的接口写法到时可以照着改。

## 2 CharTokenizer 把文字变成编号

### 2.1 为什么需要编号

**模型只会对数字做运算，没法直接处理 `"h"` 这个字母。** tokenizer（分词器）负责在文字和编号之间翻译，只做两件事：

- `encode`（编码）：文字 → 编号，交给模型。
- `decode`（解码）：模型输出的编号 → 文字，给人看。

**分词就是把文本切成一个个小单位**，每个单位叫一个 **token**。切多大没有规定：

| 切法 | `"playing"` 切成 | 词表大小 |
|---|---|---|
| 按单词 | `playing` | 几十万，还会遇到没见过的词 |
| 按子词（GPT-2 用的 BPE） | `play` + `ing` | 约 5 万 |
| 按字符（这里用的） | `p l a y i n g` | 莎士比亚剧本只有 65 种 |

按字符切最简单：几行代码就能写完，也不会遇到不认识的词。缺点是序列很长、单个字母几乎不带意思，所以真正的 GPT-2 不这么切。阶段 0 的重点是训练流程，tokenizer 先用最简单的，阶段 1 再换成 BPE。在字符级 tokenizer 里，**一个字符就是一个 token**。

**编号本身没有含义**，就像储物柜的柜号：1 号柜和 2 号柜之间没什么关系，柜号只用来找到对应的柜子。编号 7 不比编号 1 "大"，也不代表两个字符更像。

**tokenizer 不学习。** 词表在创建时建好就固定了，`h` 永远对应同一个编号。学习的是模型：模型里有一张表，编号用来找到表里的某一行（[第 4 节](#4-bigramlanguagemodel-查表打分并算-loss)），训练调整的是那张表里的数字。用柜子打比方：柜号固定，柜子里的东西在训练中不断调整。

```text
Tokenizer："h" → 编号 1（固定词表，不学习）
模型：编号 1 → 取表的第 1 行 → 预测下一个字符（这一行会被训练调整）
```

还要区分**训练**和**使用**（推理）。训练时：先分词，再预测、算 loss、调整模型里的数字。使用时：输入的文字先按固定词表分词，再交给已经训练好的模型生成回答，模型里的数字保持不变。模型读取当前对话来回答，不等于在这次对话里重新训练。

### 2.2 CharTokenizer 代码逐行讲解

最终代码（省略了 docstring 和注释，完整版见源文件）：

```python
class CharTokenizer:
    def __init__(self, corpus: str) -> None:
        unique_chars = sorted(set(corpus))
        self.stoi = {char: idx for idx, char in enumerate(unique_chars)}
        self.itos = {idx: char for idx, char in enumerate(unique_chars)}
        self.vocab_size = len(unique_chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[char] for char in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[idx] for idx in ids)
```

用 `CharTokenizer("hello")` 一行一行地看。

**`__init__`：建词表。** `corpus` 是用来建词表的语料。

```python
unique_chars = sorted(set(corpus))
```

- `set("hello")` 把字符串拆成字符并**去重**，得到集合 `{'h', 'e', 'l', 'o'}`。`l` 出现了两次，但它是同一种字符，只留一个。词表记录的是"有哪几种字符"，每种只需要一个编号。
- 集合**没有规定的顺序**，Python 每次运行时遍历集合的顺序都可能不同。`sorted` 把它排成确定的列表 `['e', 'h', 'l', 'o']`，这样同一份语料每次都得到同一份编号表，与字符在原文里出现的先后无关。编号的大小本身没有意义，排序只是为了让编号稳定、可复现。（实际使用中，tokenizer 也可以直接加载事先保存好的编号表，不一定每次重新排序；关键是词表一旦确定，就始终用同一份。）

```python
self.stoi = {char: idx for idx, char in enumerate(unique_chars)}
self.itos = {idx: char for idx, char in enumerate(unique_chars)}
```

- `enumerate(['e', 'h', 'l', 'o'])` 依次给出 `(0, 'e')`、`(1, 'h')`、`(2, 'l')`、`(3, 'o')`：每个字符配上它的位置。
- `{char: idx for ...}` 是**字典推导式**，把这些配对收集成字典。
- `stoi`（string to index）：字符 → 编号，`{'e': 0, 'h': 1, 'l': 2, 'o': 3}`，encode 时用。
- `itos`（index to string）：编号 → 字符，`{0: 'e', 1: 'h', 2: 'l', 3: 'o'}`，decode 时用。两个字典互为反向映射，才能"编码再解码"还原原文。
- `self.` 开头的变量是对象的**属性**，建好以后一直保存在这个 tokenizer 里，后面的 encode、decode 都能用。

```python
self.vocab_size = len(unique_chars)
```

- 词表大小 = 字符种类数 = 4。注意不是语料长度：`"hello"` 长 5，词表大小是 4。

**`encode`：文字 → 编号。**

```python
return [self.stoi[char] for char in text]
```

这是**列表推导式**：`for char in text` 依次取出每个字符，`self.stoi[char]` 查出它的编号，方括号把结果收集成列表。

```text
encode("hello")：h → 1，e → 0，l → 2，l → 2，o → 3   得到 [1, 0, 2, 2, 3]
```

去重只发生在建词表时；encode 按原文顺序处理每个字符，两个 `l` 都保留。词表里没有的字符，字典查询会直接报 `KeyError`。

**那 `l` 出现得多这个信息会不会丢？** 不会。训练时每遇到一次 `l`，模型里和 `l` 对应的那部分参数就被调整一次；出现得越多，调整得越多。频率靠训练数据里的重复体现，不需要给 `l` 分配两个编号。

**`decode`：编号 → 文字。**

```python
return "".join(self.itos[idx] for idx in ids)
```

读这种写法时，**先看右边的 `for`，再看左边的表达式**：依次从 `ids` 取出一个编号，赋给 `idx`；再计算左边的 `self.itos[idx]`，查出对应字符。例如 `ids = [1, 0]`：先取 `idx = 1` 查出 `h`，再取 `idx = 0` 查出 `e`。

`join` 把这些字符拼成一个字符串。它前面的字符串是**片段之间的分隔符**，这里是空字符串，表示字符之间什么都不插：

```python
"".join(["h", "e", "l", "l", "o"])   # "hello"
"-".join(["h", "e", "l", "l", "o"])  # "h-e-l-l-o"
```

`join` 返回一个新的字符串，不是往某个变量后面追加；空列表 join 得到空字符串，所以 `decode([])` 返回 `""`。

**decode 为什么没写方括号？** 不带 `[]` 的写法叫**生成器表达式**，它不先把结果全部算出来存成列表，而是在 `join` 遍历时逐个产生。两种写法在这里结果相同：

```python
"".join([self.itos[idx] for idx in ids])   # 列表推导式：先得到完整的字符列表
"".join(self.itos[idx] for idx in ids)     # 生成器表达式：join 遍历时逐个查
```

| 对比 | 列表推导式 | 生成器表达式 |
|---|---|---|
| 创建时 | 立即逐项计算，保存完整列表 | 遍历时才逐项计算 |
| 取值 | 支持下标，可以多次遍历 | 只能按顺序取，不支持下标 |
| 再次遍历 | 列表还在，可以重来 | 用完就空了 |

```python
ids = [1, 0, 2, 2, 3]
chars_list = [self.itos[idx] for idx in ids]  # 立即得到 ["h", "e", "l", "l", "o"]
chars_gen = (self.itos[idx] for idx in ids)   # 此时还没开始查

"".join(chars_list)  # "hello"
"".join(chars_list)  # 再用一次，仍是 "hello"
"".join(chars_gen)   # "hello"，生成器被用完
"".join(chars_gen)   # ""，同一个生成器已经空了
```

作为函数唯一的参数时，生成器表达式可以省略自己的外层圆括号，所以写成 `"".join(... for ...)`。不要把生成器理解成"一定更快"或"完全不占内存"，`join` 最终仍要构造完整的字符串。encode 则必须用方括号，因为调用方要拿到一个可以反复使用的整数列表。

### 2.3 规则与边界

**词表与编号规则：**

1. "字符"指 Python 字符串逐个取出的 Unicode 字符（码点），不是英文单词，也不保证是屏幕上一个完整的可见符号。
2. 对字符种类去重，再排序，编号从 0 连续到 `vocab_size - 1`。
3. 去重只用于词表；encode 不对文本去重，也不打乱顺序。
4. 词表只在创建时建立。encode / decode 只查现有词表，不增加新字符、不重新编号。如果每次输入都重新编号，同一个编号就会在不同时候代表不同字符，模型里按编号存的参数就全乱了。
5. 空格、换行、标点、中文和 emoji 都按相同规则处理；保留大小写，不做去空格、转小写之类的处理。

**大小写敏感**：`A` 和 `a` 是两个不同的字符，各占一个编号。用 `"Aa"` 建表得到 `{'A': 0, 'a': 1}`。如果编码前统一转成小写，单凭编号就无法知道原来哪些字母是大写，也就无法还原原文。

**边界情况：**

| 情况 | 结果 |
|---|---|
| `encode("")` | `[]` |
| `decode([])` | `""` |
| 用空字符串建词表 | 两个字典都为空，`vocab_size` 为 0 |
| encode 遇到词表外的字符 | 抛出 `KeyError`，不跳过、不替换、不扩充词表 |
| decode 遇到词表外的编号（包括负数） | 抛出 `KeyError` |

**两个例子：**

```text
CharTokenizer("hello")
  stoi = {'e': 0, 'h': 1, 'l': 2, 'o': 3}，vocab_size = 4
  encode("hello") → [1, 0, 2, 2, 3]
  decode([1, 0, 2, 2, 3]) → "hello"
  encode("lol") → [2, 3, 2]

CharTokenizer("baba ")
  stoi = {' ': 0, 'a': 1, 'b': 2}       （空格也是字符，排在最前）
  encode("abba ") → [1, 2, 2, 1, 0]
  encode("ac") → KeyError（c 不在词表里）
  decode([3]) → KeyError（只有 0、1、2）
```

### 2.4 CharTokenizer 的测试

[tests/test_char_tokenizer.py](../../tests/test_char_tokenizer.py) 的 7 项测试检查：词表去重并排序；编码保留顺序和重复字符；中文、emoji 等 Unicode 字符能往返；词表与语料中字符的出现顺序无关；空文本；空语料；未知字符和编号报错。

```bash
python -m unittest discover -s tests -p test_char_tokenizer.py -v
```

### 2.5 CharTokenizer 踩过的坑

初版只用 `set` 去重，没有排序。编码、解码往返没问题，但编号顺序不稳定，7 项测试只通过了 5 项（失败的是"词表排序"和"按指定编号编码"）。加上 `sorted` 后全部通过。

### 2.6 CharTokenizer 相关问答

#### Q2.1 这个东西在真实的 GPT 里面有用吗？

**A：** 有用。它对应真实 GPT 中 tokenizer 的基础职责：把输入文本转换成 token 编号，并把输出编号还原成文本。区别在于切法：这里一个字符就是一个 token；GPT-2 使用 byte-level BPE，token 可以是词的一部分、整个词或其他文本片段。encode / decode 的接口会保留，阶段 1 换成更完整的切分方式。

#### Q2.2 这个属于 tokenizer 的一部分吗？会随着不同的 tokenizer 变化吗？比如 ChatGPT 和 Claude 各用各的？

**A：** 对，词表加上 encode / decode，就是一个简单的字符级 tokenizer。不同模型可以使用不同的 tokenizer，同一家公司的不同模型也可能不同。切分规则、词表、编号和特殊 token 都可能不一样。下面只是示意，并非某个产品的实际编码：

```text
同一文本：playing
tokenizer A：[play, ing]
tokenizer B：[playing]
字符级 tokenizer：[p, l, a, y, i, n, g]
```

所以同一段文字，在不同 tokenizer 下 token 数量和编号都可能不同；两个 tokenizer 即使切出相同的片段，也可能给它不同的编号。

**tokenizer 必须和模型配套。** 模型的表是按编号排的（[第 8 节](#8-回顾-编号在代码里怎么流动)会再讲），换一个词表，编号就全对不上了。OpenAI 的 [tiktoken](https://github.com/openai/tiktoken) 提供按模型选择编码的接口；Claude 的[官方 token counting 文档](https://platform.claude.com/docs/en/build-with-claude/token-counting)按指定模型计算 token，不应拿 OpenAI 的 tokenizer 代替。阶段 3 加载 GPT-2 官方权重时，也必须使用 GPT-2 自己的词表和规则。

#### Q2.3 现在写的这个部分，是所有 tokenizer 都有的吗？都一样或者差不多吗？

**A：** 对语言模型的 tokenizer 来说，词表加编码、解码是共同的基本职责，只是实现和命名不必相同：

| 这里的写法 | 语言模型 tokenizer 中的对应功能 |
|---|---|
| `stoi`：字符到编号 | token 到编号，token 可以是字符、字节或文本片段 |
| `itos`：编号到字符 | 编号到 token，再还原为文本 |
| `encode` | 切分文本，并转换成编号序列 |
| `decode` | 把编号序列还原为文本 |
| `vocab_size` | 词表中不同 token 的数量 |

这里按字符去重、排序建词表，编码时逐字符查表；GPT-2 使用现成的词表和 BPE 合并规则，编码时还要做切分与合并。接口和数据流不变，内部规则升级。

#### Q2.4 现阶段是让模型预测下一个字符，而不是 token？真正的模型预测的是下一个 token 还是下一个词？比如 ChatGPT 或 Claude。

**A：**

- 字符级 tokenizer 里一个字符就是一个 token，所以"预测下一个字符"就是"预测下一个 token"。
- 做预测的是模型，不是 tokenizer。tokenizer 只负责切分和编号：文字 → 编号 → 模型预测下一个编号 → 还原成文字。
- 真实模型预测的是下一个 **token**，不是下一个词。GPT-2、ChatGPT、Claude 都用子词级 token：常见英文词往往整个是一个 token，少见词或长词会拆成几块；词前面的空格通常和词连在一起，例如 GPT-2 中 `" hello"` 和 `"hello"` 是两个不同的 token；中文字符可能是一个 token，也可能被拆成几个字节级 token。
- 模型每次只生成一个 token，接到末尾再预测下一个，所以回答是一小块一小块冒出来的。
- 具体切法因模型而异。tiktoken 可以查看 OpenAI 模型的切分结果；Claude 的 tokenizer 细节没有公开。阶段 1 实现 GPT-2 的 BPE 后，可以亲手看到一个词怎么被切开。

## 3 get_batch 切出题目和答案

### 3.1 题目 x 和答案 y

整份语料 encode 之后，得到一长串编号 `data`（莎士比亚剧本约 111 万个）。训练时不会把它整个交给模型，而是每次**随机切出 B 段，每段 T 个字符**：

- 整条序列太长，模型一次能看的长度有上限，这个上限就是 `block_size`（T）。
- 每次随机切不同的片段，模型能见到更多样的样本。
- 一次处理 B 段，GPU 可以并行计算；把多段的误差平均起来，参数更新也更稳定。

每一段怎么变成"题目"和"答案"？关键是**往后挪一格**。用 `"hello world"`、T = 4、起点 2 举例：

```text
下标：     0  1  2  3  4  5  6  7  8  9  10
data：     h  e  l  l  o  ␣  w  o  r  l  d       （␣ 表示空格）

x = data[2 : 6]   →  l  l  o  ␣                  题目
y = data[3 : 7]   →  l  o  ␣  w                  答案：窗口往后挪一格
```

x 和 y 上下对齐，**每一列是一道题**：

```text
x：  l   l   o   ␣
     ↓   ↓   ↓   ↓
y：  l   o   ␣   w
```

y 的每个位置，就是 x 同一位置"下一个字符"的正确答案。所以**一段长 T，就包含 T 道题**；一批 B 段，就有 B × T 道题。本项目 B = 32、T = 8，每批 256 道题。

### 3.2 合法起点

每段要切 T + 1 个位置（x 和 y 合起来覆盖 T + 1 个位置），起点不能太靠后，否则最后一个答案会越界。

起点为 s 时，用到的最后一个下标是 s + T，它必须 ≤ N − 1（N 是 data 的长度），所以 **s ≤ N − T − 1**。合法起点是 0, 1, …, N − T − 1，一共 N − T 个。

检查一个极端情况：N = 5、T = 4，整条 data 正好只够切一段，起点只能是 0。代入 N − T − 1 = 0，对得上。

### 3.3 get_batch 代码逐行讲解

```python
def get_batch(data, block_size, batch_size):
    length = len(data)
    ix = torch.randint(low=0, high=length - block_size, size=(batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x, y
```

输入 `data` 是形状 `(N,)` 的一维张量（N 个编号），`block_size` 是 T，`batch_size` 是 B。输出 x、y，形状都是 `(B, T)`。

```python
length = len(data)
```

data 的长度 N。

```python
ix = torch.randint(low=0, high=length - block_size, size=(batch_size,))
```

`torch.randint` 生成随机整数，三个参数：

| 参数 | 作用 | 这里 |
|---|---|---|
| `low` | 最小值，**能取到** | 0 |
| `high` | 上限，**取不到** | N − T |
| `size` | 结果的形状 | `(B,)`，即 B 个数 |

因为 `high` 取不到，填 N − T 正好得到 0 到 N − T − 1，就是全部合法起点。结果是一个形状 `(B,)` 的张量，例如 `tensor([0, 3, 4])`，就是 B 个随机起点。

**为什么 `size` 要写成 `(batch_size,)`？** `size` 描述的是**形状**，而形状可以有多个维度（`(2, 3)` 是二维），PyTorch 统一用元组表示，一维也不例外。`(32)` 只是加了括号的整数 32，和数学里的 `(1+2)` 一样；`(32,)` 末尾的逗号才让它成为"只有一个元素的元组"，必须留着。

```python
x = torch.stack([data[i : i + block_size] for i in ix])
y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
```

- `for i in ix`：对每个起点各做一次。
- `data[i : i + block_size]`：从 data 里**切片**，取下标 i 到 i + T − 1 这一段，共 T 个编号。切片和 randint 一样**左闭右开**：`data[0:4]` 取下标 0、1、2、3，取不到 4。值就是从这里来的：切片取出的是 data 里那一段原本的数字。
- y 的起点多了 1：**加 1 的是下标，不是数值**，相当于窗口往后挪一格。
- 方括号把 B 段收集成列表，`torch.stack` 把它们**上下叠起来**，成为 `(B, T)` 的张量：有几个起点就有几行，每段长度都是 T，所以有 T 列。

用 `"hello world"` 实际运行一遍（T = 4，B = 3）：

```text
stoi：{' ': 0, 'd': 1, 'e': 2, 'h': 3, 'l': 4, 'o': 5, 'r': 6, 'w': 7}
下标：  0  1  2  3  4  5  6  7  8  9  10
data： [3, 2, 4, 4, 5, 0, 7, 5, 6, 4, 1]

randint 挑出 ix = [0, 3, 4]

i = 0：data[0:4] → [3, 2, 4, 4]  "hell"     data[1:5] → [2, 4, 4, 5]  "ello"
i = 3：data[3:7] → [4, 5, 0, 7]  "lo w"     data[4:8] → [5, 0, 7, 5]  "o wo"
i = 4：data[4:8] → [5, 0, 7, 5]  "o wo"     data[5:9] → [0, 7, 5, 6]  " wor"
        └── 这一列叠成 x ──┘                 └── 这一列叠成 y ──┘

x = [[3, 2, 4, 4],         y = [[2, 4, 4, 5],
     [4, 5, 0, 7],              [5, 0, 7, 5],
     [5, 0, 7, 5]]              [0, 7, 5, 6]]       形状都是 (3, 4)
```

两点约定：get_batch 假设 N ≥ T + 1；它不负责把数据搬到 GPU，切出来的 x、y 和 data 在同一个设备上（[第 7 节](#7-main-把一切串起来)的 main 会把 data 放到 GPU 上）。

### 3.4 get_batch 的测试

[tests/test_get_batch.py](../../tests/test_get_batch.py) 的 5 项测试检查：形状和类型；每一行都是 data 中连续的一段；y 是 x 往后挪一格；N 恰好等于 T + 1 时的边界；所有合法起点都有机会被选中。测试数据故意让"值"和"下标"不同（`arange * 3 + 1`），这样把起点下标误当成数据返回的写法也会被抓出来；后两项专门抓上限写错一位（N − T − 1 或 N − T + 1）的错误。

### 3.5 get_batch 踩过的坑

初版写成 `size=batch_size`，5 项测试都报 `TypeError`，原因就是上面说的：`size` 是形状，必须是元组。改成 `(batch_size,)` 后全部通过。上限 `length - block_size` 一开始就写对了，这是最容易出错的地方。

### 3.6 get_batch 相关问答

#### Q3.1 y 是 x + 1 吗？张量里是数字还是字符？

**A：** **不是数值加 1，是窗口往后挪一格。** 加 1 的是起点下标：

```text
data   = [3, 2, 4, 4, 5, 0, ...]，T = 4，起点 0
x      = data[0:4] = [3, 2, 4, 4]
y      = data[1:5] = [2, 4, 4, 5]   ← 取 data 里下一个位置原本的值
x + 1  = [4, 3, 5, 5]               ← 每个数值加 1，这不是 y
```

data 进入 get_batch 之前就已经编码好了，所以 x、y 里全是编号，get_batch 用不到 tokenizer。模型从头到尾只处理编号，只有人要读结果时，才用 decode 把编号还原成字符。上面例子里写的 `"hell"`、`"ello"` 是为了方便阅读额外 decode 出来的。

#### Q3.2 batch_size 是一次给出几个？一批到底是几道题？生成时有 batch_size 吗？

**A：** `batch_size` 是一次切出几段，也就是 x 有几行。但"题"的单位要分清：**一行不是一道题，而是一段文字；这段文字的每个位置都是一道"猜下一个字符"的题。**

```text
x 第 0 行：  h    e    l    l          y 第 0 行：  e    l    l    o
            题1  题2  题3  题4    答案分别是 →     e    l    l    o
x 第 1 行：  w    o    r    l          y 第 1 行：  o    r    l    d
            题5  题6  题7  题8    答案分别是 →     o    r    l    d
```

按段数是 B 段；按题数是 B × T 道。loss 按题计算：每个位置算一个，再对 B × T 个取平均（[第 4 节](#44-交叉熵-loss-怎么算)）。

`batch_size` 这个变量只在两个地方用到，都是传给 get_batch：训练循环每一步抽一批来训练，estimate_loss 每次抽一批来评估（[第 7 节](#73-main-代码逐行讲解)）。每一步的 B × T 道题取平均后，只更新一次参数。一次给多段而不是一段：只看一段就更新，方向容易被它带偏，多段平均更稳定；GPU 一次算 32 段和算 1 段花的时间差不多；代价是 B 越大越占显存。

**生成时没有 `batch_size`，但仍有 B 这一维**：B 由传给 generate 的 idx 有几行决定。main 传的是 `torch.zeros((1, 1), ...)`，B = 1；传 `torch.zeros((3, 1), ...)` 就一次生成 3 段，结果形状 `(3, 201)`。

| | 训练 / 评估 | 生成 |
|---|---|---|
| B 从哪来 | 超参数 `batch_size`，传给 get_batch | idx 有几行就是几 |
| 本项目的值 | 32 | 1 |
| B 段之间的关系 | 随机切出，loss 平均后一起更新参数 | 各写各的，互不影响 |

两边是同一个维度（张量的第一维，"几段一起算"），只是训练时它是要调的超参数，生成时只是输入的形状。forward 和 generate 对 1 段、32 段的处理方式完全一样。

#### Q3.3 一道题就是一个 token，那同一行里前面的题有什么意义？加了注意力才有意义吗？

**A：** 对 bigram 来说，每道题只用到一个 token，前面的字符确实没有意义。x 的一行 `h e l l` 对它而言是四道互不相关的题：h → e、e → l、l → l、l → o。把 B 段 × T 个位置排成表格，和随机抽 B × T 个相邻字符对没有区别；改 block_size 也不会改变它能学到什么，只会改变每批的题数。这时"前面的题"的意义只是：每个位置都是一个训练样本，数据被充分利用了。

加了注意力后，每个位置可以看自己和前面所有位置，同一行就变成由短到长的 T 道题：

```text
                        bigram 看到的    GPT 看到的
题 1：猜 h 后面的字   →  h                h
题 2：猜 e 后面的字   →  e                h e
题 3：猜 l 后面的字   →  l                h e l
题 4：猜 l 后面的字   →  l                h e l l
```

题 3 和题 4 对 bigram 来说都是"看到 l"，分不清；GPT 看到的是 `hel` 和 `hell`，能知道第一个 l 后面接 l、第二个 l 后面接 o。这种格式对 GPT 有两个好处：一次前向同时训练 1 到 T 种上下文长度，不用为每种长度单独准备数据；生成时提示词可能很短、续写后越来越长，训练时各种长度都练过，生成时都能应对。

get_batch 和 forward 的接口（`(B, T)` 进、`(B, T, V)` 出、y 往后挪一格）换成 GPT-2 后不用改，只换模型内部，前面的题就自动有了意义。

## 4 BigramLanguageModel 查表打分并算 loss

### 4.1 模型本体是一张表

bigram 模型里**只有一张 V × V 的表**，这就是它的全部"知识"。以 `"hello world"` 的 8 种字符为例，表是 8 × 8：

```text
                     候选：  ' '    d     e     h     l     o     r     w
第 0 行（当前是 ' '）       [  .     .     .     .     .     .     .     .  ]
...
第 4 行（当前是 l）  →      [-0.61  0.03 -0.49  0.25  0.44  0.11  0.64  0.44]   ← 一行
...
                                                          第 5 列：给候选 o 打分
```

- **第 i 行**：当前字符是 i 时，"下一个字符是各个候选"的分数，一共 V 个。
- **第 j 列**：给候选字符 j 打分的那组数字（[第 9 节](#9-延伸问答)会讲到，这一列就是一个"神经元"的权重）。

这些分数叫 **logits**：任意实数，可正可负，越大表示越可能。上面第 4 行是刚初始化时的随机值：模型认为 l 后面最可能是 r（0.64 最大），显然不对。**训练就是调整表里的这些数字**（[第 6 节](#6-训练-让表里的数字变好)），所以它们叫**可学习参数**。

### 4.2 `__init__` 代码逐行讲解

```python
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)
```

- `class BigramLanguageModel(nn.Module)`：继承 PyTorch 的 `nn.Module`。所有 PyTorch 模型都这么写，继承后有两个好处：PyTorch 会自动登记模型里的参数（这张表），优化器就能找到并更新它们；写 `model(...)` 时会自动调用下面的 `forward`。
- `super().__init__()`：先初始化父类 `nn.Module`，参数登记等机制才能工作。这一行必须在最前面。
- `nn.Embedding(行数, 列数)`：一张可学习的表，传入编号就取出对应的行。行数 V 对应"当前字符"有 V 种，列数 V 对应"下一个字符"有 V 个候选。表本身存在 `self.embedding.weight` 里，形状 `(V, V)`，初始值是随机数。

**PyTorch 自己有没有现成的模型？** PyTorch 提供的是零件，不是现成的 GPT-2：

| 层次 | 例子 | 本项目怎么用 |
|---|---|---|
| 张量运算、自动求导 | `torch.tensor`、`.backward()` | 直接用 |
| 积木零件（`torch.nn`） | `nn.Embedding`、`nn.Linear`、`F.cross_entropy` | 直接用，拿来拼模型 |
| 现成的大组件 | `nn.MultiheadAttention` | 不用，注意力要自己写 |
| 完整的 GPT-2 | Hugging Face `transformers` 的 `GPT2LMHeadModel` | 不用来实现，阶段 3 只当参考答案对比结果 |

`nn.Embedding` 只是"按编号取一行"的工具；模型怎么搭、logits 和 loss 怎么算，是自己写的。

### 4.3 forward 代码逐行讲解

```python
def forward(self, idx, targets=None):
    logits = self.embedding(idx)

    if targets is None:
        loss = None
    else:
        B, T, V = logits.shape
        logits_flat = logits.view(B * T, V)
        targets_flat = targets.view(B * T)
        loss = F.cross_entropy(logits_flat, targets_flat)

    return logits, loss
```

`forward` 叫**前向计算**：从输入算出输出。它有两个参数：

- `idx`：一批编号，形状 `(B, T)`。训练时就是 get_batch 的 x。（idx 是 index 的缩写；生成时传进来的不一定来自 get_batch，所以不叫 x。）
- `targets`：答案，形状 `(B, T)`，训练时就是 get_batch 的 y；生成文本时没有答案，为 `None`。

调用时一般不直接写 `model.forward(x, y)`，而是写 `model(x, y)`，`nn.Module` 会自动转去调用 forward。

**第一步：查表。**

```python
logits = self.embedding(idx)
```

规则只有一条：**把 idx 里的每一个编号，换成表里对应的那一行。** 一个编号是一个数，一行是 V 个数，所以每个数都"展开"成 V 个数，形状末尾多出一个长度为 V 的维度：

```text
输出形状 = idx 的形状 + (V,)
```

以 x 的第 0 行 `[3, 2, 4, 4]` 为例（V = 8）：

```text
idx[0] = [ 3,        2,        4,        4      ]
           ↓         ↓         ↓         ↓
         第 3 行   第 2 行   第 4 行   第 4 行
         (8 个数)  (8 个数)  (8 个数)  (8 个数)

logits[0] 的形状是 (4, 8)；3 段合起来，(3, 4) 变成 (3, 4, 8)
```

几个容易混淆的地方：

- **表一直是 `(V, V)`**，在 `__init__` 里建好，从来不变；三维的是**查表结果** logits。两者是不同的张量。
- **idx 和表都没有被修改**，logits 是新建的张量。
- 两个 4 查的是**同一行**：同一个字符不管出现在哪个位置，得到的分数都一样（这就是 bigram 没有上下文的体现）。
- 取出的是表的**一行**，不是一列；取出的是**分数**，还不是概率：可正可负，加起来也不等于 1。

查表的结果 `logits[b, t]` 就是模型对"`idx[b, t]` 后面是什么"这道题的回答：V 个分数，每个候选一个。

**第二步：没有答案就不算 loss。**

```python
if targets is None:
    loss = None
```

生成文本时只需要分数，不需要 loss。

**第三步：有答案就算 loss。**

```python
B, T, V = logits.shape
logits_flat = logits.view(B * T, V)
targets_flat = targets.view(B * T)
loss = F.cross_entropy(logits_flat, targets_flat)
```

- `logits.shape` 是 `(B, T, V)`，这一行把三个数分别取出来。
- `F.cross_entropy` 是 PyTorch 的通用分类函数，它只认一种格式：输入 `(N, V)`（N 道题，每道 V 个选项的分数）、答案 `(N,)`（每道题的正确选项编号）。它不认识"B 段、每段 T 个位置"的结构。
- 所以先把题目排成一列：`.view(B * T, V)` 把 `(B, T, V)` 变成 `(B·T, V)`，`.view(B * T)` 把 `(B, T)` 变成 `(B·T,)`。B·T 就是这一批一共有多少道题：

```text
logits (3, 4, 8)  →  (12, 8)：12 道题，每道 8 个分数
targets (3, 4)    →  (12,)：  12 个正确答案
```

- `.view` 只是**重新解读形状**，不复制、也不打乱数据，按"先排完第 0 段，再排第 1 段……"的顺序排。logits 和 targets 用同样的方式压平，第 k 道题和第 k 个答案始终对齐。
- `F.cross_entropy` 对每道题算一个"答错程度"，再取平均，得到一个数。具体怎么算见下一小节。（cross_entropy 其实也接受 `(B, V, T)` 这种把选项放在第二维的格式，但那样要先交换维度，不如压平直观。）

**第四步：返回。**

```python
return logits, loss
```

返回的是**压平之前**的 `(B, T, V)` logits。压平只是为了满足 cross_entropy 的格式；生成文本时还要用 `logits[:, -1, :]` 取每段最后一个位置（[第 5 节](#5-generate-一个字一个字地续写)），保持三维才方便。

### 4.4 交叉熵 loss 怎么算

**交叉熵**（cross entropy）是分类问题最常用的 loss：从若干选项里选一个，这里就是从 V 个字符里选下一个。对每一道题：

1. **softmax**：把 V 个分数变成 V 个概率。做法是对每个分数取 e 的指数，再除以总和，结果全为正、加起来等于 1，原来分数越大的概率越大。例如分数 `[2.0, 1.0, 0.1]` → 指数 `[7.39, 2.72, 1.11]` → 除以总和 11.21 → 概率 `[0.66, 0.24, 0.10]`。
2. 取出模型给**正确答案**的概率 p。
3. 这道题的 loss = **−ln(p)**。

最后对所有题取平均。

| 模型给正确答案的概率 | loss |
|---|---|
| 1（完全确定地猜对） | 0 |
| 0.5 | 0.69 |
| 1/8（8 个候选平均瞎猜） | ln 8 ≈ 2.08 |
| 接近 0（很确定地猜错） | 很大 |

所以 **loss 越小越好**。用 −ln 的好处是：对"很有把握却猜错"惩罚特别重，逼模型不要乱下结论。

用实际数字看一次：随机初始化的 8 × 8 表，x 的一段 `h e l l`，答案 `e l l o`：

| 题 | 看到 | 正确答案 | 模型给正确答案的概率 | −ln p |
|---|---|---|---|---|
| 1 | h | e | 0.051 | 2.98 |
| 2 | e | l | 0.134 | 2.01 |
| 3 | l | l | 0.162 | 1.82 |
| 4 | l | o | 0.117 | 2.15 |

4 道题取平均，**loss = 2.24**。logits 是模型对每道题的回答，loss 是这些回答整体错得有多离谱。

**一个很实用的检查**：随机初始化时模型基本在瞎猜，loss 应该接近 **ln(V)**。莎士比亚的 V = 65，ln 65 ≈ 4.17；实际会略高一点（约 4.7），因为随机的表不只是平均瞎猜，还会对某些错误答案"很有把握"。如果一开始的 loss 和这个数差得很远，说明代码有问题。

### 4.5 BigramLanguageModel 的测试

[tests/test_bigram_model.py](../../tests/test_bigram_model.py) 的 6 项测试检查：模型只有一张 `(V, V)` 的参数表；不传答案时 logits 是 `(B, T, V)`、loss 是 None；传答案时 logits 仍是 `(B, T, V)`（不压平）；loss 等于所有位置交叉熵的平均；同一编号在任何位置分数都相同，且不受其他位置影响；loss 能反向传播到表上。

### 4.6 BigramLanguageModel 踩过的坑

- `nn.Embedding(vocab_size, 256)`：列数写成了 256。列数必须是 `vocab_size`，因为每一列对应一个候选字符，有多少种字符就要多少列。写成 256 的话，logits 是 `(B, T, 256)`，和词表对不上。（256 这种写法是 GPT-2 里"把编号变成向量"的做法，bigram 不需要。）
- `if targets is not None: loss = None`：条件写反了。没有答案时才不算 loss。
- `__init__` 末尾残留骨架里的 `raise NotImplementedError`，模型一创建就报错。

### 4.7 BigramLanguageModel 相关问答

#### Q4.1 forward 里的 idx 是什么？查表时 idx 的内容被替换了吗？

**A：** idx 就是 get_batch 返回的 x（题目），y 作为 targets 传入：`logits, loss = model(x, y)`。idx 是二维 `(B, T)`，不是一维，因为 get_batch 用 `torch.stack` 把 B 段叠在了一起。一个编号对应一行，`(3, 4)` 的 idx 里有 12 个编号，就取出 12 行，按原来的位置摆好，得到 `(3, 4, 8)`。

查表**新建**了 logits 张量，idx 和表都不变。输出形状总是"idx 的形状 + (V,)"：输入单个编号得到 `(V,)`，输入 `(T,)` 得到 `(T, V)`，输入 `(B, T)` 得到 `(B, T, V)`。

## 5 generate 一个字一个字地续写

### 5.1 一轮做什么

**给一个开头，让模型一个字一个字地往下写。** 每一轮只生成**一个**新字符，分五步：

```text
idx (B, T)           当前的整段文字（编号）
  │ ① 前向
  ▼
logits (B, T, V)     每个位置的"下一个是谁"打分
  │ ② 只取最后一个位置：logits[:, -1, :]
  ▼
(B, V)               "整段末尾后面接什么"的打分
  │ ③ softmax
  ▼
probs (B, V)         变成概率
  │ ④ torch.multinomial：按概率抽一个
  ▼
idx_next (B, 1)      抽中的编号
  │ ⑤ torch.cat：接到 idx 末尾
  ▼
idx (B, T+1)         长度 +1，进入下一轮
```

重复 `max_new_tokens` 轮，返回整段：原来的开头 + 新生成的部分。

**为什么只取最后一个位置？** 每个位置的 logits 预测的是"它后面那个字符"。前面位置的"后面"已经有字了，只有最后一个位置的"后面"还空着，它才是要的预测。

**生成时 B 和 T 是什么？** T 是开头有几个 token；B 是同时生成几段互不相关的续写。以 hello 的词表（e=0、h=1、l=2、o=3）为例：

```text
从 "hel" 续写 1 段：      idx = [[1, 0, 2]]                          (1, 3)
同一开头续写 3 个版本：    idx = [[1, 0, 2], [1, 0, 2], [1, 0, 2]]      (3, 3)
不同开头 "he" 和 "lo"：    idx = [[1, 0], [2, 3]]                     (2, 2)，每行长度必须相同
```

返回值的 B 不变（进来几段，出去几段），每一段变长：前 T 个是原样保留的开头，后面接上新生成的 `max_new_tokens` 个。

```text
输入 (1, 3)：[[1, 0, 2]]                        "hel"
生成 5 个后 (1, 8)：[[1, 0, 2, 2, 3, ?, ?, ?]]   原开头 3 个 + 新生成 5 个
```

### 5.2 generate 代码逐行讲解

```python
@torch.no_grad()
def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        logits, _ = self(idx)
        logits_last = logits[:, -1, :]
        probs = F.softmax(logits_last, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

- `@torch.no_grad()`：写在函数上方的**装饰器**，表示这个函数里不记录反向传播需要的信息（[第 6 节](#6-训练-让表里的数字变好)会讲反向传播）。生成只做前向、不训练，这样更省内存，也更快。
- `for _ in range(max_new_tokens)`：循环 max_new_tokens 次；`_` 表示"这个循环变量用不到"。

下面用一个 `(2, 3)` 的 idx 走一轮（V = 4）：

```text
            第0列  第1列  第2列
第 0 行  [    1,     0,     2  ]   ← "hel"
第 1 行  [    2,     3,     0  ]   ← "loe"
```

```python
logits, _ = self(idx)
```

① 把**整段** idx 交给前向（`self(idx)` 会调用 forward）。注意不是只取第一个 token：6 个编号各查出一行，logits 是 `(2, 3, 4)`。没传 targets，所以 loss 是 None，用 `_` 忽略。

```python
logits_last = logits[:, -1, :]
```

② 取每段最后一个位置。方括号里三个位置对应三个维度：`:` 表示这一维全取，`-1` 表示最后一个。从 `(2, 3, 4)` 变成 `(2, 4)`：每段剩下一行 4 个分数。中间那一维（T）消失了，因为只挑了一个位置。消失的只是这个中间结果的 T 维，idx 本身仍是 `(2, 3)`。

```python
probs = F.softmax(logits_last, dim=-1)
```

③ softmax 沿最后一维（4 个候选）把分数变成概率，形状仍是 `(2, 4)`。`dim=-1` 指"最后一维"。

```python
idx_next = torch.multinomial(probs, num_samples=1)
```

④ **按概率抽签**：每一行各抽 1 个编号，概率越大越容易被抽中。

```text
probs   第 0 行：[0.1, 0.2, 0.6, 0.1] → 抽中编号 2
        第 1 行：[0.7, 0.1, 0.1, 0.1] → 抽中编号 0
idx_next = [[2], [0]]，形状 (2, 1)
```

`(B, V)` 变成 `(B, 1)`，是因为每行从 V 个候选里只抽 **1 个**（`num_samples=1`）；概率只是抽签的依据，抽完只留下编号。写成 `(2, 1)` 这种"竖着的一列"，而不是 `(2,)`，是为了下一步能直接作为新的一列拼到 idx 右边。

抽中的位置下标恰好就是字符编号，因为表的第 j 列给"编号为 j 的字符"打分，probs 的第 j 个位置也就对应编号 j。

```python
idx = torch.cat((idx, idx_next), dim=1)
```

⑤ `torch.cat` 沿第 1 维（列的方向，也就是长度方向）拼接：

```text
[[1, 0, 2],    +   [[2],    →   [[1, 0, 2, 2],    "hell"
 [2, 3, 0]]         [0]]         [2, 3, 0, 0]]    "loee"
```

B 仍是 2，T 从 3 变成 4。`cat` 返回一个**新**张量，再让 idx 指向它，所以调用方传进来的原张量不受影响。下一轮把这个 `(2, 4)` 整段交给前向，如此重复。

一轮的形状变化总结：

```text
idx         (B, T)       整段编号
logits      (B, T, V)    每个位置 V 个分数
logits_last (B, V)       只留最后一个位置
probs       (B, V)       变成概率
idx_next    (B, 1)       每段抽出 1 个编号
idx         (B, T+1)     接到末尾
```

**每轮都把整段交给前向，不浪费吗？** bigram 其实只用得到最后一个字符，确实有点浪费。但这样接口和 GPT-2 的生成完全一样，以后几乎不用改。到了 GPT-2，还要把太长的序列截到 block_size 以内（阶段 4）；"每轮重算整段"的浪费，由 KV cache 解决（阶段 6）。

### 5.3 为什么抽签而不是选最高分

用 `"hello world"` 训练好的表实际跑一下，从 h 开始各生成 20 个字符：

```text
总是选最高分：  'helllllllllllllllllll'
按概率抽签：    'heldrldrldrllorlorlo '
               'heldwo worlorlorlo wo'
```

训练后，l 那一行给 l、o、d 各 1/3 的概率（并列最高）。"选最高分"每次都选同一个（这里是 l），陷进 l → l → l 的死循环；抽签每次都有机会选到不同候选，结果更多样，也更符合原文的统计规律。所以**同一个开头生成两次，结果可能不一样**。阶段 4 的 temperature、top-k、top-p 就是对这一步加控制，调节抽签有多"放飞"。

抽签结果里还出现了 d → e，可原文里 d 是最后一个字符，后面什么也没有。所以 d 那一行**从来没被训练过**，仍是随机数：没出现过的搭配，模型学不到。

### 5.4 generate 的测试

[tests/test_generate.py](../../tests/test_generate.py) 的 5 项测试检查：输出形状、类型和编号范围；开头原样保留，且不修改传入的 idx；生成 0 个时原样返回；每一步都根据**最后一个**编号生成；用的是抽签而不是总选最高分。后两项是把表设成指定的分数来构造确定的情况：比如让"当前是 0 时，1 和 2 各占一半概率"，抽 200 次必须两个都出现过。

### 5.5 generate 踩过的坑

五步一次写对。未训练的模型从 h 开始生成 20 个字符：`'h dlelewhlohl lrewhrh'`，完全是乱码，因为表还是随机的。

### 5.6 generate 相关问答

#### Q5.1 模型怎么知道从哪里开始写？

**A：** **它不能从零开始，必须先给至少一个 token。** 模型只会做一件事：看到已有的内容，预测下一个。

- bigram 的 main 从编号 0 开始，在莎士比亚的词表里，0 是换行符，相当于"从新的一行写起"。
- GPT-2 有一个特殊 token `<|endoftext|>`（编号 50256），训练时用来分隔不同的文章；从它开始，就是"一篇新文章开始了"。
- 聊天模型的开头是整段对话，用特殊标记标出角色，模型从"助手"标记之后开始续写。它在训练中见过大量对话，学会了在那里接什么内容。

所以"提笔写什么"，就是读完开头之后预测第一个 token，后面每个字都是同样的"看前面、猜下一个"。因为是抽签，同一个问题两次的开头也可能不同。

#### Q5.2 续写几段是什么意思？大模型不是只给一个回答吗？提问和回答是一体的吗？

**A：** B 表示**一次同时计算几条互不相干的序列**，是为了并行，不是为了给多个回答。

| 场景 | B 是什么 |
|---|---|
| 训练 | 32 段随机切出的文本 |
| 用户聊天 | 这一条对话，对用户来说是 1 |
| 服务器实际计算 | 很多用户的对话拼成一批，一起算、互不影响，GPU 利用率更高 |
| 两个回答让人选 | 同一个问题生成 2 条，抽签让两条不同 |

B 是几条一起算，不代表模型"想得更多"：每一条的结果和单独算它时一样。

**提问和回答是一体的。** 对模型来说只有一条连续的 token 序列，回答就是它的续写：

```text
[系统指令][用户]你好，你是谁？[助手]我是 Claude……
└──────────── 已有的开头 ───────────┘└─ 续写出来的 ─┘
```

回答的每个字，在生成下一个字时都成了输入的一部分，这就是 generate 里 `torch.cat` 那一行。由此：

- **多轮对话也是一体的。** 问第二个问题时，交给模型的是"第一个问题 + 第一个回答 + 第二个问题"整段。模型"记得"之前聊过什么，是因为整段历史每次都重新放进输入，而不是有单独的记忆。
- **对话有长度上限。** 模型一次能看的 token 数叫**上下文窗口**（context window），和 `block_size` 是同一个概念：bigram 是 8，GPT-2 是 1024，现在的大模型有几十万甚至更多。超过上限，最早的内容就放不进去，产品通常会截断或压缩历史。
- **不同对话默认相互独立**，新对话看不到旧对话，除非产品提供"记忆"功能，把以前的信息再放进输入。

## 6 训练 让表里的数字变好

### 6.1 训练到底改了什么

先不看代码，用 `"hello world"` 的 8 × 8 表看清楚"训练"这件事。下面的反向传播和更新是为了演示临时写的。

**训练前**：第 4 行（l）经过 softmax 后的概率是随机的，模型认为 l 后面最可能是 r：

```text
候选：  ' '    d     e     h     l     o     r     w
概率：  0.06  0.11  0.06  0.13  0.16  0.12  0.20  0.16
```

用 `h e l l` → `e l l o` 这 4 道题算出 loss = 2.24（见[第 4.4 节](#44-交叉熵-loss-怎么算)的表）。

**算梯度**：PyTorch 根据 loss 算出表里**每个数**"应该往哪个方向调、调多少"，这叫**梯度**（下一小节详细讲）。结果：

1. **只有用到的行有梯度。** 这 4 道题里出现的当前字符只有 h、e、l，所以只有第 3、2、4 行的梯度不为 0，其他 5 行这一步学不到东西。
2. **用到的行里，正确答案的分数被调高，其他被调低。** 第 4 行（l）出现在两道题里，答案分别是 l 和 o。

**更新一步后**：

```text
候选：          ' '    d     e     h     l     o     r     w
更新前概率：   0.06  0.11  0.06  0.13  0.16  0.12  0.20  0.16
更新一步后：   0.05  0.10  0.06  0.12  0.19  0.14  0.18  0.15
                                       ↑ 升   ↑ 升   ↓ 降
```

loss 从 2.24 降到 2.02。这就是一次"训练"。

**训练 500 步以后**：loss 降到 0.47，再看两行：

```text
第 3 行（h 后面）：e 0.99，其他接近 0
第 4 行（l 后面）：l 0.33，o 0.33，d 0.33，其他 0
```

对照原文：h 后面永远是 e；l 后面分别出现过 l（he**ll**o）、o（hel**lo**）、d（wor**ld**）各一次。**表学到的，就是训练数据里"哪个字符后面跟什么、各占多少比例"**，这就是 bigram 能学到的全部知识。loss 降不到 0，是因为 l 后面本来就有三种可能，只看当前一个字符，不可能确定地猜中。

### 6.2 梯度 学习率和优化器

**梯度**：对表里的每一个数，"如果把它稍微调大一点，loss 会变大还是变小、变化有多快"。

- 梯度为**负**：调大这个数，loss 会**变小**，所以应该调大。
- 梯度为**正**：调大这个数，loss 会**变大**，所以应该调小。
- 绝对值越大，影响越明显。

打个比方：蒙着眼睛站在山坡上，想走到山谷（loss 最低处），梯度就是脚下往哪边倾斜、有多陡。

**最朴素的更新规则**：

```text
新值 = 旧值 − 学习率 × 梯度
```

上面的演示里，第 4 行对候选 l 的梯度是 −0.17，学习率取 1：0.44 − 1 × (−0.17) = 0.61，分数变高了。

**学习率**（learning rate）决定每一步迈多大：太小学得慢，太大会"迈过头"。本项目用 0.01。

**梯度本身不能调**，它是由当前的表和这批数据算出来的事实，就像脚下的坡度由地形决定。能调的是"怎么利用梯度"：

| 能调的东西 | 意思 | 在哪学 |
|---|---|---|
| 学习率 | 每一步迈多大 | 本阶段 |
| 优化器 | 怎么迈步 | 本阶段 |
| 梯度裁剪 | 坡太陡时限制步子，防止一步迈飞 | 阶段 5 |
| 梯度累积 | 攒几批的梯度再迈一步 | 阶段 5 |

**优化器**（optimizer）负责执行更新。它掌握着模型所有参数的名单。最朴素的 SGD 严格按上面的公式更新；本项目用的 **AdamW** 更聪明：会给每个参数自动调整步子大小，并参考前几步的方向（**动量**，类似惯性）；它还有**权重衰减**（weight decay，AdamW 里的 W），每一步都把**所有**参数往 0 拉一点点，防止参数变得过大、防止过拟合。GPT-2 和现在的大模型基本都用 AdamW。

**注意**：因为有动量和权重衰减，用 AdamW 时，没用到的行也会变化。在莎士比亚数据上训练 10 步后检查，65 行全部变了。"只有用到的行有梯度"仍然成立，但"只有用到的行会变"只对朴素的 SGD 成立。

### 6.3 train_step 代码逐行讲解

一步训练 = 四个动作：

```python
def train_step(model, optimizer, x, y):
    _, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```

```python
_, loss = model(x, y)
```

① **前向**：x 进入 forward 叫 idx，y 叫 targets，得到 loss。这时 loss 不只是一个数：PyTorch 还**悄悄记下了"它是怎么从表里的数字一步步算出来的"**，这份计算记录是反向传播的依据。logits 用不到，用 `_` 忽略。

```python
optimizer.zero_grad()
```

② **清空梯度**：把每个参数上残留的梯度清掉。PyTorch 的 `backward()` 默认把新梯度**累加**到旧梯度上，而不是覆盖；不清空，这一步的梯度就会混进上一步的。（这种"默认累加"是故意设计的，阶段 5 的梯度累积正是利用它。）`zero_grad` 只要放在 `backward()` 之前即可；如果放在 `backward()` 和 `step()` 之间，刚算好的梯度就被擦掉了，这一步什么也学不到。

```python
loss.backward()
```

③ **反向传播**：沿着第 ① 步记下的计算过程**倒推回去**，算出每个参数的梯度，存到参数的 `.grad` 里。表的梯度 `self.embedding.weight.grad` 形状和表一样，也是 `(V, V)`，每个数对应一个梯度。

```python
optimizer.step()
```

④ **更新参数**：优化器读取每个参数的 `.grad`，按 AdamW 的规则调整参数。**表里的数字到这一步才真正改变。**

```python
return loss.item()
```

⑤ `.item()` 把只含一个数的张量（0 维张量）变成普通的 Python 数字。只用来打印、记录的话，数字就够了；直接存张量会连带保留整套计算记录，占内存。返回的是第 ① 步算出的值，也就是**更新之前**的 loss。

**前向和反向**，一张表对比：

| | 前向传播（forward pass） | 反向传播（backpropagation） |
|---|---|---|
| 方向 | 输入 → 计算 → 输出和 loss | loss → 倒着往回 → 每个参数的梯度 |
| 回答的问题 | 模型现在的答案是什么，错得多离谱 | 表里每个数该往哪边调，loss 才会变小 |
| PyTorch 里 | `model(x, y)`，自动调用 `forward` | `loss.backward()` |
| 谁来写 | 自己写 | PyTorch 自动求导 |

这正是用 PyTorch 的最大好处：只要写清楚前向怎么算，反向的数学（链式法则求导）它全包了。生成文本时只做前向，不需要反向，因为不改表。

在莎士比亚数据上实际跑 10 步，每一步返回的 loss：

```text
4.731, 4.755, 4.677, 4.72, 4.72, 4.643, 4.663, 4.542, 4.644, 4.618
```

整体在往下走，但**忽上忽下**（第 2 步反而升了），因为每一步抽到的 32 段不同，有的难、有的容易。这就是需要下面 `estimate_loss` 的原因。

### 6.4 estimate_loss 代码逐行讲解

```python
@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size, eval_iters):
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(data, block_size, batch_size)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)
```

这是**考试**：只做前向看分数，不调用 `zero_grad`、`backward`、`step`，不改参数。

- `@torch.no_grad()`：考试不需要反向传播，不记录计算过程，省内存。测试也验证了：调用后参数不变，也没有留下梯度。
- `model.eval()` / `model.train()`：切换模型的"模式"。有些层在训练和评估时行为不同，最典型的是 **dropout**（训练时随机关掉一部分神经元，评估时全部打开）。bigram 没有这种层，切不切结果都一样；到了 GPT-2，如果用 dropout，评估时必须切到 eval 模式。最后切回训练模式，回到训练循环继续训练。（这里是无条件切回训练模式；本项目的训练循环中模型一直处于训练模式，所以没问题。）
- 循环 `eval_iters` 次：每次用 get_batch 抽**一批**，也就是 B × T = 256 道题；`model(x, y)` 算出的 loss 已经是这 256 道题的平均，是**一个数**，存进 losses。
- 最后对 losses 里的 `eval_iters` 个数再取平均。本项目 `eval_iters = 200`，一次评估相当于考了 200 批 × 256 道 = 51,200 道题。

实际效果：训练前 estimate_loss 为训练集 4.730、验证集 4.724；训练 10 步后训练集 4.614。比单步 loss 稳定得多，能看清真实水平。

### 6.5 训练部分的测试

[tests/test_training.py](../../tests/test_training.py) 的 5 项测试检查：train_step 返回的是更新**前**的 loss，类型是 float；连续两步的结果，和手算"新值 = 旧值 − 学习率 × 这一步自己的梯度"完全一致（用来抓"忘了清空梯度"和"清空梯度的位置放错"）；在 `0 1 2 0 1 2…` 上训练后，能学会这个循环；estimate_loss 等于多批 loss 的平均；estimate_loss 不改参数、不留梯度。

### 6.6 训练部分踩过的坑

两个函数一次写对。`model.eval()` / `model.train()` 是自己加的，骨架里没有要求，是个好习惯。

### 6.7 训练部分相关问答

#### Q6.1 max_iters 是一共训练几步，是不是就是一共训练几轮？

**A：** 不是。两个词在机器学习里有固定含义：

| 叫法 | 英文 | 意思 |
|---|---|---|
| 一步 | step / iteration | 抽**一批**数据，做一次 train_step |
| 一轮 | epoch | 把**整个训练集**完整过一遍 |

算一下这次训练看了多少数据：训练集 1,003,854 个字符；一步 32 × 8 = 256 道题；3000 步共 768,000 道题，约为训练集的 0.77 倍，**连一轮都不到**。而且起点是随机抽的，有的片段被抽到好几次，有的一次也没抽到，和"每个样本恰好过一遍"的一轮不同。语言模型通常按步计数、随机抽样，因为数据量大到常常一轮都跑不完，所以超参数里只有 `max_iters`，没有 epoch。

## 7 main 把一切串起来

### 7.1 数据

训练数据是 **tiny Shakespeare**：莎士比亚剧本的节选，来自 Karpathy 的 [char-rnn](https://github.com/karpathy/char-rnn) 仓库，约 1 MB。在仓库根目录下载（`data/` 不进版本库）：

```bash
mkdir -p data && curl -o data/input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

共 1,115,394 个字符，65 种，所以 V = 65。开头是剧本的台词格式：

```text
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.
```

### 7.2 训练集和验证集

文本的前 90% 用来训练，后 10% **永远不参与训练**，专门用来考试：

- 两边 loss 都低：模型学到了通用规律。
- 训练集 loss 低，验证集明显更高：模型在**死记**训练数据，这叫**过拟合**。

按顺序切而不打乱，因为文本前后是连贯的，打乱就破坏了"下一个字符"的关系。

一个过拟合的实例：用仓库里一份很小的中文文档代替莎士比亚试跑，训练集 2.78、验证集 4.37。那份文档太小、字符种类太多（957 种），后 10% 里有不少训练集里罕见甚至没出现过的字。

（叫法上的小区分：我们切的是**训练集**和**验证集**。**测试集**通常指第三份数据，等所有调参都结束后才最终用一次；本项目没有切测试集。）

### 7.3 main 代码逐行讲解

```python
def main() -> None:
    torch.manual_seed(1337)

    block_size = 8
    batch_size = 32
    learning_rate = 1e-2
    max_iters = 3000
    eval_interval = 300
    eval_iters = 200
```

- `torch.manual_seed(1337)`：**固定随机种子**。表的初始值、get_batch 抽的起点、generate 的抽签都依赖随机数；固定种子后，每次运行结果完全一致，便于对比修改前后的效果。
- 下面六个是**超参数**：训练前人为定好，不靠训练学出来。依次是：每段几个字符、每批几段、学习率、一共训练几步、每隔几步评估一次、每次评估抽几批。

```python
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
```

**选设备**：有 NVIDIA 显卡用 cuda，Apple 芯片用 mps，都没有就用 cpu。**模型和数据必须放在同一个设备上**，否则计算时会报错。`f"..."` 是 f-string，花括号里的变量会被替换成它的值。

```python
    with open("data/input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharTokenizer(text)
    data = torch.tensor(
        tokenizer.encode(text), dtype=torch.long, device=device
    )
```

- `with open(...) as f`：打开文件，`with` 保证读完后自动关闭。`encoding="utf-8"` 避免不同系统默认编码不同导致乱码。`f.read()` 把整个文件读成一个字符串。
- 用全文建词表（[第 2 节](#2-chartokenizer-把文字变成编号)），V = 65。
- 全文编码成编号列表，再用 `torch.tensor` 转成张量：`dtype=torch.long` 是查表要求的整数类型；`device=device` 直接把张量建在选好的设备上。形状 `(N,)`。

```python
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
```

按顺序切：前 90% 是训练集（约 100 万个字符），后 10% 是验证集（约 11 万个）。`data[:n]` 是从开头到 n（不含），`data[n:]` 是从 n 到结尾。

```python
    model = BigramLanguageModel(tokenizer.vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
```

- 建模型（[第 4 节](#4-bigramlanguagemodel-查表打分并算-loss)），`.to(device)` 把模型的参数（那张 65 × 65 的表）搬到选好的设备上。
- 建优化器（[第 6.2 节](#62-梯度-学习率和优化器)）：`model.parameters()` 把模型全部参数的名单交给它，`lr` 是学习率。

```python
    print("--- 训练前生成结果 ---")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    initial_gen = model.generate(context, max_new_tokens=200)[0].tolist()
    print(tokenizer.decode(initial_gen))
    print("----------------------\n")
```

**训练前先生成一段**，看看随机的表会写出什么：

- `torch.zeros((1, 1), ...)`：形状 `(1, 1)` 的 `[[0]]`，B = 1 段、T = 1 个字符；编号 0 在这份词表里是换行符，相当于"从新的一行开头写起"。也要放到设备上，和模型一致。
- `model.generate(context, max_new_tokens=200)` 返回 `(1, 201)`：开头 1 个 + 新生成 200 个（[第 5 节](#5-generate-一个字一个字地续写)）。
- `[0]` 取第 0 行，得到 `(201,)`；`.tolist()` 转成 Python 列表，才能交给 decode。
- `"\n"` 是换行符，让输出多空一行。

```python
    for iter_step in range(max_iters):
        if iter_step % eval_interval == 0:
            train_loss = estimate_loss(model, train_data, block_size, batch_size, eval_iters)
            val_loss = estimate_loss(model, val_data, block_size, batch_size, eval_iters)
            print(f"step {iter_step}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")

        xb, yb = get_batch(train_data, block_size, batch_size)
        train_step(model, optimizer, xb, yb)
```

**训练循环**（为了阅读方便，这里把源文件中分成多行的函数调用合并成了一行）：

- `iter_step` 从 0 到 2999，一共 3000 步。
- `iter_step % eval_interval == 0`：`%` 是取余数，余数为 0 表示"是 300 的倍数"，所以第 0、300、600……步各评估一次，在训练集和验证集上各考一次（[第 6.4 节](#64-estimate_loss-代码逐行讲解)）。`:.4f` 表示保留 4 位小数。
- 每一步：从**训练集**抽一批（xb、yb 的 b 表示 batch，[第 3 节](#3-get_batch-切出题目和答案)），然后训练一步（[第 6.3 节](#63-train_step-代码逐行讲解)）。get_batch 切出的 x、y 自动和 data 在同一个设备上。

```python
    final_train_loss = estimate_loss(model, train_data, block_size, batch_size, eval_iters)
    final_val_loss = estimate_loss(model, val_data, block_size, batch_size, eval_iters)
    print(f"step {max_iters}: train loss {final_train_loss:.4f}, val loss {final_val_loss:.4f}")

    print("\n--- 训练后生成结果 ---")
    final_gen = model.generate(context, max_new_tokens=200)[0].tolist()
    print(tokenizer.decode(final_gen))
    print("----------------------")
```

- 循环里最后一次评估在第 2700 步，这里补上训练完之后的结果，标成 step 3000。
- 用**同一个开头** context 再生成 200 个字符，和训练前对比。

```python
if __name__ == "__main__":
    main()
```

只有直接运行这个文件（`python -m warmup.bigram`）时才执行 main；被别的文件 import（比如测试）时不执行，只提供里面的类和函数。

### 7.4 运行结果

```text
step 0:    train loss 4.7305, val loss 4.7241
step 300:  train loss 2.8110, val loss 2.8249
step 600:  train loss 2.5434, val loss 2.5682
step 900:  train loss 2.4932, val loss 2.5088
step 1200: train loss 2.4863, val loss 2.5035
step 1500: train loss 2.4665, val loss 2.4921
step 1800: train loss 2.4683, val loss 2.4936
step 2100: train loss 2.4696, val loss 2.4846
step 2400: train loss 2.4638, val loss 2.4879
step 2700: train loss 2.4738, val loss 2.4911
step 3000: train loss 2.4613, val loss 2.4897
```

在 Apple M5 的 GPU（mps）上约 5 秒跑完。三点观察：

1. **loss 从 4.73 降到 2.46。** 起点高于 ln 65 ≈ 4.17，原因见[第 4.4 节](#44-交叉熵-loss-怎么算)。
2. **训练集和验证集几乎一样**（2.46 和 2.49），没有过拟合。bigram 只能记 65 × 65 种搭配，想死记也记不住。
3. **约 1000 步之后就降不动了**，停在 2.46 左右。这不是训练出了问题，而是 **bigram 的能力上限**：它已经把"每个字母后面跟什么、各占多少比例"学到头了（[第 6.1 节](#61-训练到底改了什么)），只看一个字符，最多做到这里。要再往下降，就得让模型看到更多上下文，也就是 GPT-2 的注意力。

训练前后的生成对比：

```text
训练前：
jhRysNp$;M:bZ!PcxfEQ!
IDjJ.sV
dbT?dt3fNmxq$wRju3MmJi.SnxquBkgjgm,skyZPPWNhOVJddEvyh

训练后：
S:

Wh I:

Anaturveloromo's.
LOULUMI ak
FiEOn ted nn cthik isth gha ch'thim s Helll y atisin m tomeiss
```

训练前是完全随机的字符，大小写、标点乱飞。训练后能看到很多真实规律：

- **剧本格式**：大写的人名后面跟冒号再换行（`S:`、`Wh I:`），和原文的 `First Citizen:` 一样。
- **空格切出了"单词"**，长度也接近英文。
- **常见组合**：th、the、is、's。
- 但**没有一个完整的真词**，因为它只看一个字母，不知道自己正在拼哪个词。

这就是 bigram 能做到的全部，也是阶段 0 想亲眼看到的东西。

### 7.5 main 踩过的坑

写完后，函数末尾还残留着骨架里的 `raise NotImplementedError`：训练和生成都正常跑完，最后却以报错结束（退出码 1）。这类"写完没删骨架"在整个阶段 0 出现了好几次，之前残留的只是注释，这次是会执行的代码。每个任务写完后，搜一下 `TODO` 和 `NotImplementedError` 再提交。

## 8 回顾 编号在代码里怎么流动

最后回头看一遍：**tokenizer 只在两头用**，开头 encode、结尾 decode；中间所有地方都只是整数，在不同的变量里传来传去。

**训练时：**

```text
data/input.txt 里的文字
   │ tokenizer.encode(text)                    编号在这里产生
   ▼
list[int]
   │ torch.tensor(..., dtype=torch.long)
   ▼
data          (N,)       全部编号
   │ get_batch 从 data 里切片
   ▼
x, y          (B, T)     编号是从 data 复制过来的
   │ model(x, y)
   ▼
forward 里的 idx = x，targets = y               进入前向后换了名字
   │ self.embedding(idx)
   ▼
logits                   编号已经被换成了分数
```

**生成时**，编号有两个来源：开头由自己给，比如 `torch.tensor([tokenizer.encode("h")])` 或 `torch.zeros((1, 1), dtype=torch.long)`；后面新增的，是 `torch.multinomial` 抽出来的。编号通过 `idx` 一路传进前向：

```text
start ──▶ generate 的 idx ──self(idx)──▶ forward 的 idx ──▶ self.embedding(idx)
             ↑                                                   │
             └──────── torch.cat 接上新编号，下一轮再传 ─────────┘
```

generate 的 idx 和 forward 的 idx 是**两个函数各自的参数**，只是恰好同名；`self(idx)` 这次调用把 generate 手上的张量交给 forward，forward 用自己的参数名接住。**不管训练还是生成，编号进入前向只有一个入口：forward 的第一个参数 idx。** 最后 `tokenizer.decode(out[0].tolist())` 把编号变回文字。

| 变量 | 在哪 | 装的是什么 | 形状 |
|---|---|---|---|
| `data` | main | 全文的编号 | (N,) |
| `x`, `y` | get_batch 返回 | 从 data 切出的编号 | (B, T) |
| `idx` | forward 参数 | 传进来的 x，或生成时的序列 | (B, T) |
| `targets` | forward 参数 | 传进来的 y | (B, T) |
| `logits` | forward 内 | 每个位置 V 个分数 | (B, T, V) |
| `idx` | generate 内 | 开头 + 已生成的编号，每轮变长 | (B, T) → (B, T+1) … |
| `idx_next` | generate 内 | 这一轮抽中的新编号 | (B, 1) |

**为什么抽中的位置下标恰好是字符编号？** 因为表是按编号排的：第 j 列给"编号为 j 的字符"打分，probs 的第 j 个位置对应编号 j，抽中 j，decode 出来就是 `itos[j]`。这也是为什么**模型和 tokenizer 必须配套**。

### 8.1 回顾相关问答

#### Q8.1 这个模型只能预测训练用的那句话吗？

**A：** 不是，它学的是喂给它的任何数据。用 tiny Shakespeare 训练，它学到英文里哪个字母后面常跟哪个字母，比如 q 后面几乎总是 u。但能力有上限：只看一个字符，表最多记录 V × V 种前后搭配，生成的只是看起来有点像英文的字母串，拼不出完整的词，更谈不上语法；数据再多也突破不了。要突破，就得让模型看到更多上下文，这是 GPT-2 的注意力要解决的问题。

## 9 延伸问答

从这个小模型出发，聊到 GPT-2 和现在的大模型。阶段 2 会亲手实现其中的注意力。

### Q9.1 输出既然是查表，是不是固定的？大模型怎么知道上下文？是 Transformer 解决的吗？

**A：** bigram 的输出确实固定：当前字符是 l，取出的永远是第 4 行，前面写过什么都不影响。这是它没有上下文的致命局限。

大模型训练结束后，权重也同样固定，回答时不会修改参数。区别在于：**bigram 的输出只取决于一个字符；大模型的输出是整段上下文经过一个固定函数算出来的。** 好比计算器的电路是固定的，但输入 2+3 和 7×8 结果不同；bigram 这台计算器只有一个输入口，Transformer 能把整段上下文都放进去。

以 GPT-2 为例：

```text
文本 → tokenizer 切成 token 编号
① 查 wte 表：每个 token 取一行向量（和 bigram 一样，固定、不看上下文）
② 经过 12 层，每层：
     注意力：每个位置回头看前面所有位置，按相关程度把它们的信息混进来
     MLP：对混合后的结果再加工
③ 最后一个位置的向量融合了整段信息，乘以输出矩阵得到 V 个分数
④ 按概率抽出下一个 token，接到末尾，回到 ① 再来一遍
```

关键在 ②：**"看前面哪些词、看多少"不是存在表里的，而是根据当前内容实时计算的。** 例如 river bank 中，注意力重点看 river，bank 被调整成"河岸"的意思；bank account 中重点看 account，bank 被调整成"银行"。同一个词在 ① 查出的向量相同，经过注意力后变成随上下文变化的向量。第 ④ 步就是[第 5 节](#5-generate-一个字一个字地续写)的抽签，所以同一个问题两次的回答可能不同。

利用上下文不是 Transformer 首创：RNN、LSTM 从左到右逐字读，把读过的内容压缩进一个"记忆"向量，但容易忘掉较早的内容，也必须按顺序计算、无法并行。2017 年的论文 *Attention Is All You Need* 提出 Transformer，让每个位置直接看到前面所有位置，而且可以并行计算；GPT 系列、Claude、Gemini 等都建立在它之上。

### Q9.2 "每个 token 取一行向量"是什么意思？注意力加工的是谁、存在哪里？最后不也是查表选下一个 token 吗？

**A：** **每个 token 取一行向量**：和 bigram 查表是同一个动作，只是表的形状不同：

| | 表的形状 | 查出的一行是什么 |
|---|---|---|
| bigram | V × V | V 个分数，直接就是"下一个是谁"的答案 |
| GPT-2 的 `wte` | 50257 × 768 | 768 个数，是这个 token 的特征描述，还不是答案 |

形状变化也一样，末尾加一维：`(B, T)` → `(B, T, 768)`，每个位置有了一个 768 维向量。

**注意力加工的是这些向量**，每个位置一个。以"我 爱 猫"为例，刚查完 wte 时，v我、v爱、v猫 各自只代表自己。注意力给每个位置换一个新向量，由自己和前面各位置按比例混合而成。简化后（省略了 Q、K、V 的细节）：

```text
新 v猫 = 0.1 × v我 + 0.3 × v爱 + 0.6 × v猫
```

位置 2 的向量由此融入了"我""爱"的信息。

**注意力存在哪里**：要分清两样东西。注意力的权重矩阵（W_q、W_k、W_v 等，每层几张 768 × 768 的表）是模型参数，训练后固定，相当于"怎么计算混合比例"的规则；混合比例本身（上例的 0.1、0.3、0.6）不存，每次根据当前的向量现场计算。好比菜谱是存好的，菜每次现做。换成"我 怕 猫"，同一套规则会算出不同的比例。GPT-2 把这样的加工做 12 层，每一层之后，每个位置都融入更多上下文。

**最后一步不是按当前 token 查表**：

```text
bigram：当前字符编号 ──查表──▶ 一行分数（只取决于这一个编号）
GPT-2： 最后一个位置的向量（经过 12 层，已融合整句）
        ──乘以输出矩阵 (768 × 50257)──▶ 50257 个分数（取决于整句）
```

最后一个位置的向量在"我 爱 猫"和"我 怕 猫"里不同，算出的分数和下一个 token 也就不同。开头按编号查表是固定的；上下文是在中间的注意力层被"注意到"的；最后用融合了上下文的向量算分数。

### Q9.3 注意力就是对每个位置的向量乘以比例再相加吗？为什么是 12 层？

**A：** 对，每个位置的新向量是各位置向量按各自比例的加权和。更准确地说：

- 向量排成张量，一句话是 `(T, 768)`，一批是 `(B, T, 768)`，不是 Python 列表。
- 每个位置有自己的一组比例，而且**只能看自己和前面的位置**：

```text
                 看 我    看 爱    看 猫
新 v我   =      1.0      0        0        ← 前面没有词，只能看自己
新 v爱   =      0.4      0.6      0
新 v猫   =      0.1      0.3      0.6
```

  每一行加起来等于 1（由 softmax 算出）；右上角全是 0，因为模型的任务是预测下一个词，看到后面就等于看到了答案。这叫**因果遮罩**。它也是[第 3 节](#3-get_batch-切出题目和答案)"同一行由短到长的 T 道题"能成立的原因。
- 实际计算中：混合前向量先乘 W_v 做变换；比例由 Q、K 的点积算出，内容越相关比例越大；GPT-2 把 768 维分成 12 个"头"，每头 64 维，各自独立做一套混合（**多头注意力**），不同的头可以关注不同的关系；每层注意力后面还接一个 MLP，对每个位置单独再加工。核心始终是"按内容算比例，再加权混合"。

**12 层**是 OpenAI 为 GPT-2 最小版本选的**超参数**，不是定律：

| 模型 | 层数 | 向量维度 | 参数量 |
|---|---|---|---|
| GPT-2 small（本项目） | 12 | 768 | 124M |
| GPT-2 medium | 24 | 1024 | 355M |
| GPT-2 large | 36 | 1280 | 774M |
| GPT-2 XL | 48 | 1600 | 1.5B |
| GPT-3 | 96 | 12288 | 175B |

层数越多，上下文被混合、加工的轮数越多，能学到更复杂的规律，但计算量、显存和训练难度也更大。2020 年 OpenAI 的 scaling laws 研究发现，在合理范围内，模型能力主要取决于总参数量和数据量，参数分配得更深还是更宽，影响相对较小。所以层数通常和维度、算力预算一起权衡。本项目阶段 2–3 必须是 12 层、768 维，否则装不进官方 124M 权重；阶段 5 自己训练时会选更小的配置。

### Q9.4 注意力的比例表和输入有什么区别？输入不是 tokenizer 转成的编号吗？比例表是概率吗？

**A：** 从 tokenizer 到注意力，出现了三样不同的东西：

| | 是什么 | 形状（一句话，T = 3） | 从哪来 | 代表什么 |
|---|---|---|---|---|
| ① token 编号 | 整数 | (3,)，如 [12, 87, 305] | tokenizer | "是谁"，纯身份标识 |
| ② 向量 | 实数 | (3, 768) | 用编号查 wte | 这个 token 的特征和内容 |
| ③ 比例表 | 0 到 1 的实数 | (3, 3) | 注意力用 ② 现场计算 | 每个位置看别人看多少 |

```text
① [12, 87, 305] → 查 wte → ② 3 个 768 维向量 → 算出 ③ 3 × 3 比例表
→ 用 ③ 混合 ② → 新的 3 个 768 维向量（融入上下文）
```

① 只在开头用来查表，进入注意力后处理的是 ②；③ 混合的是向量，不是编号；③ 的大小只和句子长度有关，与 768 无关。

③ 的每行非负、和为 1，因为也由 softmax 算出，看起来像概率，但含义是**分配注意力**：这个位置从各位置取多少信息，不是"下一个词是谁"的概率，后者在最后一层才算。论文常把 ③ 叫"注意力权重"（attention weights），但它不是模型权重：

| | 模型权重（参数） | 注意力权重（比例表 ③） |
|---|---|---|
| 例子 | wte、W_q、W_k、W_v | 上面的 3 × 3 表 |
| 存不存 | 存着，训练后固定 | 不存，每次现场算 |
| 随输入变吗 | 不变 | 每句话都不同 |

③ 是用模型权重（W_q、W_k）对向量 ② 计算出来的，但它本身不是模型权重。

### Q9.5 所有大模型都有这样的表吗？量化是什么？和神经元有什么关系？

**A：** 所有这类模型的知识都存在一批数字表（矩阵）里。bigram 只有一张表，大模型有成百上千张。GPT-2 124M 中，`wte` 是 50257 × 768，`wpe` 是 1024 × 768 的位置表，12 层里每层还有注意力和 MLP 的几张表。全部数字约 1.24 亿个，就是"124M 参数"。ChatGPT、Gemini、Grok、Claude 都是基于 Transformer 的神经网络，同样是"很多张表 + 用表做计算"，只是规模大得多；具体结构没有完全公开。

**量化**是用更少的位数存每个参数：

| 格式 | 每个数占 | 70 亿参数约占 |
|---|---|---|
| float32 | 4 字节 | 28 GB |
| float16 / bfloat16 | 2 字节 | 14 GB |
| int8 | 1 字节 | 7 GB |
| 4-bit | 0.5 字节 | 3.5 GB |

数值变粗糙了（比如 0.6143 存成 0.61），但模型通常仍然可用，而且更省内存、更快，所以本地能跑的模型很多是量化版。

**神经元**是借用生物学的比喻，实际是一个很简单的计算：把一组输入各乘一个权重，加起来，再经过一个简单的函数。权重就是表里的数字。

#### Q9.5 追问：神经元的输入和权重分别是什么？乘起来做什么？

**A：** 输入不是 B·T：B·T 只是一批里有多少道题，每道题独立经过同一组神经元，所以只看一道题。神经元只能对数做乘法，而编号是身份标识，不能直接拿来乘（否则编号 7 就比编号 1 "大 7 倍"了），所以先把当前字符变成 **one-hot**：长度为 V，只有自己的位置是 1。

```text
字符 l（编号 4） → one-hot：[0, 0, 0, 0, 1, 0, 0, 0]
```

权重就是 V 表。实际运行结果：

```text
one-hot @ 表 = [-0.61, 0.03, -0.49, 0.25, 0.44, 0.11, 0.64, 0.44]
表的第 4 行  = [-0.61, 0.03, -0.49, 0.25, 0.44, 0.11, 0.64, 0.44]   ← 完全一样
```

整批 `(2, 4)` 的 idx 也验证过：one-hot 乘表与 `nn.Embedding` 查表完全相等。看给候选 o 打分的那个神经元：它的权重是表的第 5 列，输入只有一个 1，各乘各的再相加后，只剩 1 那个位置的权重。所以：

- **表的每一列是一个神经元**，共 V 个，各给一个候选字符打分。
- **one-hot × 表 = 查表**。`nn.Embedding` 直接取行，省掉大量乘以 0 的运算，结果一样、速度快得多。

同一张表，横着看和竖着看含义不同：

| 怎么看 | 对应什么 | 什么时候用到 |
|---|---|---|
| 一行（第 i 行） | 当前字符是 i 时，所有 V 个神经元各自给出的分数 | 查表时取出来的就是一行，即这一道题的完整答案 |
| 一列（第 j 列） | 一个神经元的全部权重，只负责给候选 j 打分 | 这个神经元面对不同输入字符时分别打几分 |

（提前提醒：阶段 2 会用到的 `nn.Linear`，存权重的方向正好相反，形状是"(输出数, 输入数)"，一**行**对应一个神经元。行列只是存储约定。）

输入是一串各不相同的实数时，乘法才真正起作用。GPT-2 中 `wte` 查出的是 768 个实数，后面的层对它们做真正的加权求和。比如一个判断"是不是动物"的神经元：

```text
输入：   [会动 0.9,  有毛 0.8,  是金属 0.1]
权重：   [   2.0,       1.5,      −3.0   ]
加权和： 0.9×2.0 + 0.8×1.5 + 0.1×(−3.0) = 2.7   → 分数高，像动物
```

权重表示每个特征有多重要、是加分还是减分；再经过 GELU 之类的函数并叠很多层，就能表达复杂的规律。bigram 的输入是 one-hot，加权求和退化成"取一个数"，也没有额外的函数，所以它是最简单、最退化的一层，只能记住"谁后面跟谁"。

### Q9.6 更大的大语言模型也是用 torch 的零件拼出来的吗？

**A：** 模型结构基本是。比如 Meta 公开的 Llama，官方参考实现就是几百行 PyTorch，用的仍是 `nn.Embedding`、`nn.Linear`、归一化层、注意力和 MLP，和 GPT-2 是同一套骨架，只是部分零件换了新版本：

| 部件 | GPT-2 | 较新的模型（如 Llama） |
|---|---|---|
| 位置信息 | 可学习的位置表 `wpe` | RoPE（旋转位置编码） |
| 归一化 | LayerNorm | RMSNorm |
| MLP | GELU | SwiGLU |
| 注意力 | 标准多头注意力 | GQA（多个头共用 K/V，省显存） |
| 规模 | 12 层，768 维 | 几十到上百层，几千维；有的使用 MoE（混合专家） |

难点主要在模型定义之外：分布式训练（FSDP、张量并行、流水线并行）、几万亿 token 的数据处理、FlashAttention 等高效 kernel 和混合精度、长时间训练的稳定性、指令微调和 RLHF 等对齐训练，以及推理部署（KV cache、量化、批量服务）。框架也不只 PyTorch，例如 Google 的 Gemini 主要用 JAX 在 TPU 上训练。

## 附录 术语表

按在笔记中第一次出现的顺序排列。

| 术语 | 意思 | 详见 |
|---|---|---|
| 语言模型 | 看到前面的文字，猜下一个是什么 | [1.1](#11-语言模型在做什么) |
| bigram | 只看当前一个字符猜下一个的模型 | [1.2](#12-为什么先做-bigram) |
| 张量（tensor） | PyTorch 里的多维数字表格 | [1.4](#14-三个符号-b-t-v) |
| 形状（shape） | 张量每一维有多长，如 (B, T) | [1.4](#14-三个符号-b-t-v) |
| B / batch_size | 一批有几段 | [1.4](#14-三个符号-b-t-v) |
| T / block_size | 每段有几个字符；模型一次能看的长度上限 | [1.4](#14-三个符号-b-t-v) |
| V / vocab_size | 词表里有几种不同的字符 | [1.4](#14-三个符号-b-t-v) |
| tokenizer（分词器） | 在文字和编号之间翻译 | [2.1](#21-为什么需要编号) |
| token | 切分文本得到的最小单位；这里一个字符就是一个 token | [2.1](#21-为什么需要编号) |
| 词表 | 所有 token 种类和它们的编号 | [2.2](#22-chartokenizer-代码逐行讲解) |
| encode / decode | 文字 → 编号 / 编号 → 文字 | [2.2](#22-chartokenizer-代码逐行讲解) |
| batch（一批） | 一次随机切出的 B 段，共 B × T 道题 | [3.1](#31-题目-x-和答案-y) |
| x / y | 题目 / 答案，y 是 x 的窗口往后挪一格 | [3.1](#31-题目-x-和答案-y) |
| 参数 / 权重 | 模型里会被训练调整的数字，这里就是那张表 | [4.1](#41-模型本体是一张表) |
| logits | 模型给每个候选打的分数，可正可负 | [4.1](#41-模型本体是一张表) |
| nn.Module | 所有 PyTorch 模型的父类 | [4.2](#42-__init__-代码逐行讲解) |
| nn.Embedding | 按编号取出表中一行的工具 | [4.2](#42-__init__-代码逐行讲解) |
| 前向（forward） | 从输入算出输出和 loss | [4.3](#43-forward-代码逐行讲解) |
| softmax | 把一组分数变成概率（全为正、和为 1） | [4.4](#44-交叉熵-loss-怎么算) |
| loss | 模型答得有多差的一个数，越小越好 | [4.4](#44-交叉熵-loss-怎么算) |
| 交叉熵 | 分类问题的 loss：−ln(正确答案的概率) 的平均 | [4.4](#44-交叉熵-loss-怎么算) |
| 生成 / 抽样 | 按概率抽出下一个 token，接上，重复 | [5.1](#51-一轮做什么) |
| 上下文窗口 | 模型一次能看的 token 数上限 | [5.6](#56-generate-相关问答) |
| 梯度 | 每个参数往哪个方向调、调多少能让 loss 变小 | [6.2](#62-梯度-学习率和优化器) |
| 学习率 | 每一步更新迈多大 | [6.2](#62-梯度-学习率和优化器) |
| 优化器 / AdamW | 按梯度更新参数的工具 | [6.2](#62-梯度-学习率和优化器) |
| 反向传播 | 从 loss 倒推出每个参数的梯度 | [6.3](#63-train_step-代码逐行讲解) |
| 步（step） / 轮（epoch） | 一次参数更新 / 把训练集完整过一遍 | [6.7](#67-训练部分相关问答) |
| 训练集 / 验证集 | 用来训练的数据 / 不参与训练、专门用来考试的数据 | [7.2](#72-训练集和验证集) |
| 过拟合 | 死记训练数据，在验证集上表现明显更差 | [7.2](#72-训练集和验证集) |
| 超参数 | 训练前人为定好、不靠训练学出来的设置 | [7.3](#73-main-代码逐行讲解) |
| 设备（cuda / mps / cpu） | 在哪个硬件上计算；模型和数据必须在同一个设备上 | [7.3](#73-main-代码逐行讲解) |
| 注意力 | 让每个位置按内容"回头看"前面位置的机制 | [9](#9-延伸问答) |
| 因果遮罩 | 只允许看自己和前面，不允许看后面 | [9](#9-延伸问答) |
| 神经元 | 输入乘权重、相加、再过一个函数 | [9](#9-延伸问答) |
| 量化 | 用更少的位数存参数，省内存、更快 | [9](#9-延伸问答) |
