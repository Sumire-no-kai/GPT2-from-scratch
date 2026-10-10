"""阶段 0：字符级 bigram 语言模型。

整体流程：
    文本 --CharTokenizer.encode--> 编号序列 data
         --get_batch--> 题目 x、答案 y，形状都是 (B, T)
         --BigramLanguageModel(x, y)--> 每个位置的打分 logits (B, T, V) 和 loss
         --训练循环（下一步）--> 调整表里的数字，让 loss 变小
         --generate--> 一个字符一个字符地续写文本

符号约定（全文通用）：
    B：batch_size，一批有几段
    T：block_size，每段有几个字符
    V：vocab_size，词表里有几种不同的字符
"""

import torch  # 张量、随机数、自动求导等基础功能
from torch import nn  # 搭模型用的零件，如 nn.Module、nn.Embedding
from torch.nn import functional as F  # 不带参数的函数，如 softmax、cross_entropy


class CharTokenizer:
    """字符级 tokenizer：在字符和编号之间互相翻译。

    一个字符就是一个 token。词表在创建时根据语料建好，之后固定不变。

    Attributes:
        stoi: 字符 → 编号的字典（string to index），字符排序后从 0 连续编号。
        itos: 编号 → 字符的字典（index to string），与 stoi 互为反向映射。
        vocab_size: 语料中不同字符的数量 V。

    此处字符指 Python 字符串迭代得到的 Unicode 码点。

    例：CharTokenizer("hello")
        stoi = {'e': 0, 'h': 1, 'l': 2, 'o': 3}
        itos = {0: 'e', 1: 'h', 2: 'l', 3: 'o'}
        vocab_size = 4
    """

    def __init__(self, corpus: str) -> None:
        """从 corpus 建立词表；重复出现的字符只占一个编号。

        Args:
            corpus: 用来建词表的语料字符串，例如 "hello"。
        """
        # set 去重，得到字符种类 {'h', 'e', 'l', 'o'}，但集合没有固定顺序；
        # sorted 排成确定的列表 ['e', 'h', 'l', 'o']，保证每次运行编号都一样。
        unique_chars = sorted(set(corpus))

        # enumerate 依次给出 (0, 'e'), (1, 'h'), (2, 'l'), (3, 'o')，
        # 用字典推导式建两个方向的映射。
        self.stoi = {char: idx for idx, char in enumerate(unique_chars)}  # 'h' → 1
        self.itos = {idx: char for idx, char in enumerate(unique_chars)}  # 1 → 'h'

        # 词表大小 = 字符种类数，不是语料长度（"hello" 长 5，词表大小 4）。
        self.vocab_size = len(unique_chars)

    def encode(self, text: str) -> list[int]:
        """把字符串翻译成编号列表。

        Args:
            text: 要编码的字符串，其中每个字符都必须在词表里。

        Returns:
            长度为 len(text) 的编号列表，保留原文顺序和重复字符。
            例："hello" → [1, 0, 2, 2, 3]

        Raises:
            KeyError: text 中有词表外的字符。
        """
        # 逐个字符查 stoi，收集成列表；遇到词表外的字符时字典查询直接报 KeyError。
        return [self.stoi[char] for char in text]

    def decode(self, ids: list[int]) -> str:
        """把编号列表翻译回字符串。

        Args:
            ids: 编号列表，每个编号都必须在 0 到 vocab_size - 1 之间。

        Returns:
            还原出的字符串。例：[1, 0, 2, 2, 3] → "hello"

        Raises:
            KeyError: ids 中有词表外的编号。
        """
        # 逐个编号查 itos 得到字符，再用 "".join 无分隔地拼成一个字符串。
        return "".join(self.itos[idx] for idx in ids)


def get_batch(
    data: torch.Tensor, block_size: int, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """从 data 中随机切出一批训练样本：题目 x 和答案 y。

    每段切 T + 1 个字符：前 T 个是 x，后 T 个是 y。y 的每个位置，
    就是 x 同一位置"下一个字符"的正确答案，所以一段包含 T 道题。

    Args:
        data: 整份语料编码后的一维编号序列，形状 (N,)，dtype 为 torch.long。
        block_size: 每段的长度 T。
        batch_size: 一批的段数 B。

    Returns:
        (x, y)，形状都是 (B, T)，dtype 与 data 相同。
        x[b] = data[s : s + T]，y[b] = data[s + 1 : s + T + 1]，
        即 y 的窗口比 x 往后挪一格（挪的是位置，不是数值加 1）。
        每段的起点 s 独立随机选取，所有合法起点都有机会被选中。

    例：data = [3, 2, 4, 4, 5, 0, 7, 5, 6, 4, 1]（"hello world"），T = 4，起点 s = 0
        x 的这一行 = [3, 2, 4, 4]（"hell"）
        y 的这一行 = [2, 4, 4, 5]（"ello"）

    假设 N >= T + 1；结果放在 data 所在的设备上，不在这里搬到 GPU。
    """
    length = len(data)  # N，整份语料的编号个数

    # 随机选 B 个起点。起点为 s 时最后用到的下标是 s + T，必须 <= N - 1，
    # 所以合法起点是 0 到 N - T - 1；randint 的 high 取不到，正好填 N - T。
    # size 是结果的形状，必须是元组；(batch_size,) 末尾的逗号表示单元素元组。
    # 结果是形状 (B,) 的张量，例如 tensor([0, 3, 4])。
    ix = torch.randint(low=0, high=length - block_size, size=(batch_size,))

    # 对每个起点 i 切出一段长 T 的片段（得到 B 个形状 (T,) 的张量），
    # torch.stack 把它们上下叠成 (B, T)。
    x = torch.stack([data[i : i + block_size] for i in ix])  # 题目
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])  # 答案：窗口后移一格

    return x, y


class BigramLanguageModel(nn.Module):
    """只根据当前字符预测下一个字符的语言模型。

    模型只有一张 (V, V) 的表，也是它全部的可学习参数：
        第 i 行 = "当前字符编号为 i 时，下一个字符是各个候选的分数（logits）"
        第 j 列 = 给候选字符 j 打分的那个"神经元"的全部权重
    表一开始是随机数，训练就是调整这些数字。

    继承 nn.Module 后，PyTorch 会自动登记模型里的参数（这张表），
    训练时优化器就能找到并更新它们；model(...) 会自动调用 forward。
    """

    def __init__(self, vocab_size: int) -> None:
        """建一张 V × V 的表。

        Args:
            vocab_size: 词表大小 V。
        """
        super().__init__()  # 先初始化父类 nn.Module，参数登记等机制才能工作

        # nn.Embedding(行数, 列数)：一张可学习的表，传入编号就取出对应的行。
        # 行数 V：每种当前字符一行；列数 V：每个候选的下一个字符一个分数。
        # 表本身存放在 self.embedding.weight，形状 (V, V)，初始值随机。
        self.embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """前向计算：查表得到每个位置的下一个字符分数，再可选地对答案算 loss。

        通常不直接调用 forward，而是写 model(idx, targets)。

        Args:
            idx: 输入编号，形状 (B, T)，dtype 为 torch.long。训练时就是 get_batch 的 x。
            targets: 目标编号，形状 (B, T)，训练时就是 get_batch 的 y；生成时为 None。

        Returns:
            (logits, loss)。
            logits: 形状 (B, T, V)。logits[b, t] 是模型对"idx[b, t] 后面是谁"
                给出的 V 个分数；传入 targets 时也保持这个形状。
            loss: 所有 B * T 道题交叉熵的平均值，是 0 维张量；targets 为 None 时为 None。
        """
        # 查表：idx 里每个编号换成表里对应的一行（V 个分数），形状末尾多出一维。
        # 例：idx 的一行 [3, 2, 4, 4] 依次取第 3、2、4、4 行，得到 (4, V)。
        # 整批：(B, T) → (B, T, V)。idx 和表本身都不会被修改，结果是新张量。
        logits = self.embedding(idx)

        if targets is None:
            # 生成文本时没有答案，不算 loss。
            loss = None
        else:
            # F.cross_entropy 只认 (N, V) 的分数和 (N,) 的答案：N 道题排成一列。
            # 所以把 B 段、每段 T 道题压平成 B * T 道题。
            B, T, V = logits.shape
            # .view 只重新解读形状，不复制也不打乱数据：按"先第 0 段、再第 1 段"的顺序排。
            logits_flat = logits.view(B * T, V)  # (B * T, V)：每道题 V 个分数
            targets_flat = targets.view(B * T)  # (B * T,)：每道题的正确答案
            # logits 和 targets 用同样的顺序压平，第 k 道题和第 k 个答案始终对齐。

            # 交叉熵：对每道题，softmax 把 V 个分数变成概率，取正确答案的概率 p，
            # 算 -ln(p)；再对 B * T 道题取平均。答得越准，loss 越小。
            loss = F.cross_entropy(logits_flat, targets_flat)

        # 返回压平前的 logits (B, T, V)：压平只是为了算 loss，
        # 生成时还要用 logits[:, -1, :] 取每段最后一个位置。
        return logits, loss

    # @torch.no_grad()：这个函数里不记录反向传播需要的信息。
    # 生成只做前向、不训练，这样更省内存，也更快。
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """从 idx 开始，每次按概率抽一个新编号接到末尾，重复 max_new_tokens 次。

        Args:
            idx: 开头的编号序列，形状 (B, T)，dtype 为 torch.long。
                例：从 "h" 开始续写一段，就是 [[1]]（形状 (1, 1)）。
            max_new_tokens: 要新生成的编号个数。

        Returns:
            形状 (B, T + max_new_tokens) 的张量：前 T 列是原样保留的 idx，后面是新生成的编号。
            不修改传入的 idx。用 tokenizer 的 decode 可以把每一行还原成文本。
        """
        for _ in range(max_new_tokens):  # 每一轮生成一个新编号
            # ① 前向：self(idx) 会调用 forward；不传 targets，所以 loss 是 None，用 _ 忽略。
            #    得到 (B, T, V)：每个位置的"下一个是谁"打分。
            logits, _ = self(idx)

            # ② 只取每段最后一个位置：只有它的"后面"还是空的，正是要预测的。
            #    : 表示这一维全取，-1 表示最后一个。(B, T, V) → (B, V)
            logits_last = logits[:, -1, :]

            # ③ softmax 沿最后一维（V 个候选）把分数变成概率：非负、每行和为 1。仍是 (B, V)。
            probs = F.softmax(logits_last, dim=-1)

            # ④ 按概率抽签：每一行各抽 1 个编号，概率越大越容易被抽中。(B, V) → (B, 1)
            #    不用"直接选最高分"，否则容易陷入同一个字符反复出现的循环。
            idx_next = torch.multinomial(probs, num_samples=1)

            # ⑤ 沿第 1 维（长度方向）把新编号接到末尾：(B, T) + (B, 1) → (B, T + 1)。
            #    cat 返回新张量，再让 idx 指向它；调用方传入的原张量不受影响。
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


def train_step(
    model: nn.Module, optimizer: torch.optim.Optimizer, x: torch.Tensor, y: torch.Tensor
) -> float:
    """用一批数据训练一步：前向 → 清空梯度 → 反向 → 更新参数。

    Args:
        model: 要训练的模型，model(x, y) 返回 (logits, loss)。
        optimizer: 负责更新 model 参数的优化器，例如 torch.optim.AdamW。
        x: 题目，形状 (B, T)。
        y: 答案，形状 (B, T)。

    Returns:
        这一步更新参数之前的 loss，Python float。

    例：在 tiny Shakespeare 上，前 10 步返回的 loss 大约是
        4.73, 4.76, 4.68, 4.72, 4.72, 4.64, 4.66, 4.54, 4.64, 4.62，
        整体下降但忽上忽下，因为每一步抽到的段落难度不同。
    """
    # ① 前向：x 进入 forward 叫 idx，y 叫 targets，得到 0 维的 loss。
    #    PyTorch 同时记下"loss 是怎么从表里的数字一步步算出来的"，供第 ③ 步倒推。
    #    logits 这里用不到，用 _ 忽略。
    _, loss = model(x, y)

    # ② 清空梯度：backward() 默认把新梯度累加到旧梯度上，不清空就会混进上一步的梯度。
    #    只要放在 backward() 之前即可；放在 backward() 和 step() 之间会擦掉刚算好的梯度。
    optimizer.zero_grad()

    # ③ 反向传播：沿第 ① 步的计算记录倒推，算出每个参数的梯度，
    #    存到参数的 .grad 里（表的梯度 self.embedding.weight.grad 形状也是 (V, V)）。
    loss.backward()

    # ④ 更新参数：优化器读取每个参数的 .grad，按自己的规则调整参数，表里的数字到这一步才真正改变。
    #    朴素的 SGD 是"新值 = 旧值 − 学习率 × 梯度"；AdamW 还会为每个参数自动调整步子、
    #    参考前几步的方向（动量），并把所有参数往 0 拉一点（权重衰减），
    #    所以这一步没用到的行也可能发生变化。
    optimizer.step()

    # ⑤ .item() 把 0 维张量变成 Python 数字。只用来打印、记录时用数字即可；
    #    直接保存张量会连带保留整套计算记录，占内存。
    #    这是第 ① 步算出的值，也就是更新之前的 loss。
    return loss.item()


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    eval_iters: int,
) -> float:
    """在 data 上随机抽 eval_iters 批，返回平均 loss；不更新参数。

    单独一批的 loss 随抽到的片段波动很大，取多批平均才能看清趋势。

    Args:
        model: 要评估的模型。
        data: 一维编号序列，形状 (N,)，训练集或验证集。
        block_size: 每段的长度 T。
        batch_size: 每批的段数 B。
        eval_iters: 抽几批来平均。

    Returns:
        eval_iters 批 loss 的平均值，Python float。

    这是"考试"：只做前向看分数，不做 zero_grad / backward / step，不改参数。
    上方的 @torch.no_grad() 让前向不记录反向传播需要的信息，更省内存。
    """
    # 切换到评估模式。有些层在训练和评估时行为不同（如 dropout：训练时随机关掉一部分，
    # 评估时全部打开）。bigram 没有这类层，切不切结果一样；GPT-2 用 dropout 时必须切换。
    model.eval()

    # 每抽一批（get_batch 的一个 batch，共 B × T 道题）存一个数：这批题的平均 loss。
    # 循环结束后 losses 里有 eval_iters 个数。
    losses = []
    for _ in range(eval_iters):
        # 随机抽一批题目和答案，形状都是 (B, T)。
        x, y = get_batch(data, block_size, batch_size)

        # 只前向，取出 loss；logits 用不到。
        _, loss = model(x, y)

        # 存 Python 数字而不是张量。
        losses.append(loss.item())

    # 切回训练模式，回到训练循环后继续训练。
    # 注意：这里无条件切回训练模式；本项目的训练循环中模型一直处于训练模式，所以没问题。
    model.train()

    # 多批取平均：单批 loss 波动大，平均后才能看清真实水平。
    return sum(losses) / len(losses)


def main() -> None:
    """在 tiny Shakespeare 上训练 bigram，并对比训练前后的生成结果。

    在仓库根目录运行：python -m warmup.bigram
    数据文件放在 data/input.txt（不进版本库）。

    流程：
        选设备 → 读文本、编码、切成训练集和验证集 → 建模型和优化器
        → 训练前生成一段 → 训练 max_iters 步（定期评估）→ 最终评估 → 训练后再生成一段

    预期输出：loss 从约 4.7 降到约 2.5，训练集和验证集接近；
    生成的文字从完全随机的乱码，变成有剧本格式和常见字母组合、但拼不出真词的字母串。
    """
    # 固定随机种子：初始化的表、get_batch 抽的起点、generate 的抽签都依赖随机数，
    # 固定后每次运行结果完全一致，便于对比修改前后的效果。
    torch.manual_seed(1337)

    # ---------- 超参数：训练前人为定好，不靠训练学出来 ----------
    block_size = 8  # T：每段几个字符
    batch_size = 32  # B：每批几段；一步训练 B × T = 256 道题
    learning_rate = 1e-2  # 学习率：每一步迈多大
    max_iters = 3000  # 一共训练几步（一步 = 抽一批 + 一次 train_step；不是"几轮"）
    eval_interval = 300  # 每隔几步评估一次
    eval_iters = 200  # 每次评估抽几批来平均

    # ---------- 选设备：有 NVIDIA 显卡用 cuda，Apple 芯片用 mps，都没有用 cpu ----------
    # 模型和数据必须放在同一个设备上，否则计算时会报错。
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # ---------- 准备数据 ----------
    # with 语句保证读完后自动关闭文件；encoding="utf-8" 避免不同系统默认编码不同导致乱码。
    with open("data/input.txt", "r", encoding="utf-8") as f:
        text = f.read()  # 整个文件读成一个字符串，约 111 万个字符

    # 用全文建词表：tiny Shakespeare 有 65 种字符，所以 V = 65。
    tokenizer = CharTokenizer(text)

    # 全文编码成编号列表，再转成张量：dtype=torch.long 是查表要求的整数类型，
    # device=device 直接把张量建在选好的设备上。形状 (N,)。
    data = torch.tensor(
        tokenizer.encode(text), dtype=torch.long, device=device
    )

    # 按顺序切：前 90% 训练，后 10% 验证。不打乱，因为打乱会破坏"下一个字符"的关系。
    # 验证集永远不参与训练，用来检查模型学到的是通用规律还是死记训练数据（过拟合）。
    n = int(0.9 * len(data))
    train_data = data[:n]  # 约 100 万个字符
    val_data = data[n:]  # 约 11 万个字符

    # ---------- 模型和优化器 ----------
    # .to(device) 把模型的参数（那张 65 × 65 的表）搬到选好的设备上。
    model = BigramLanguageModel(tokenizer.vocab_size).to(device)

    # 优化器拿到模型全部参数的名单，之后每次 optimizer.step() 都按梯度更新它们。
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # ---------- 训练前生成：看看随机的表会写出什么 ----------
    print("--- 训练前生成结果 ---")
    # 开头：形状 (1, 1) 的 [[0]]，B = 1 段，T = 1 个字符；编号 0 在这份词表里是换行符，
    # 相当于"从新的一行开头写起"。也要放到设备上，和模型一致。
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    # generate 返回 (1, 201)：开头 1 个 + 新生成 200 个。
    # [0] 取第 0 行得到 (201,)，.tolist() 转成 Python 列表，才能交给 decode。
    initial_gen = model.generate(context, max_new_tokens=200)[0].tolist()
    print(tokenizer.decode(initial_gen))
    print("----------------------\n")

    # ---------- 训练循环 ----------
    for iter_step in range(max_iters):  # iter_step 从 0 到 max_iters - 1
        # 每隔 eval_interval 步（包括第 0 步）评估一次，在训练集和验证集上各"考一次试"。
        # 单步 loss 波动大，所以用 estimate_loss 抽 eval_iters 批取平均。
        if iter_step % eval_interval == 0:
            train_loss = estimate_loss(
                model, train_data, block_size, batch_size, eval_iters
            )
            val_loss = estimate_loss(
                model, val_data, block_size, batch_size, eval_iters
            )
            # :.4f 表示保留 4 位小数。
            print(
                f"step {iter_step}: train loss {train_loss:.4f}, val loss {val_loss:.4f}"
            )

        # 一步训练：只从训练集抽一批（xb、yb 的 b 表示 batch），
        # 然后前向 → 清空梯度 → 反向 → 更新参数。
        xb, yb = get_batch(train_data, block_size, batch_size)
        train_step(model, optimizer, xb, yb)

    # ---------- 训练结束：最终评估 ----------
    # 循环里最后一次评估在第 2700 步，这里补上训练完之后的结果。
    final_train_loss = estimate_loss(
        model, train_data, block_size, batch_size, eval_iters
    )
    final_val_loss = estimate_loss(
        model, val_data, block_size, batch_size, eval_iters
    )
    print(
        f"step {max_iters}: train loss {final_train_loss:.4f}, val loss {final_val_loss:.4f}"
    )

    # ---------- 训练后生成：用同一个开头，和训练前对比 ----------
    print("\n--- 训练后生成结果 ---")
    final_gen = model.generate(context, max_new_tokens=200)[0].tolist()
    print(tokenizer.decode(final_gen))
    print("----------------------")


# 直接运行这个文件（python -m warmup.bigram）时才执行 main；
# 被别的文件 import（例如测试）时不执行，只提供里面的类和函数。
if __name__ == "__main__":
    main()
