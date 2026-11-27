"""GPT-2 的模型结构，从输入 token ID 到下一个 token 的分数。"""

import math

import torch
from torch import nn


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


class CausalSelfAttention(nn.Module):
    """按各头的注意力权重汇总当前位置和之前位置的信息。"""

    # 设置头数和每头的宽度，建立 Q/K/V 共用的投影层和输出投影层。
    def __init__(self, embedding_dim: int, num_heads: int) -> None:
        """用向量维度 C 和头数 H 创建注意力，要求 C 可被 H 整除。

        每头处理 D = C // H 个数。两个线性层的权重形状为 (3C, C) 和 (C, C)。
        """
        super().__init__()
        if embedding_dim <= 0 or num_heads <= 0 or embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be a positive multiple of num_heads")

        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.c_attn = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.c_proj = nn.Linear(embedding_dim, embedding_dim)

        # 沿用原版 GPT-2 的线性层初始化，权重标准差为 0.02，偏置为 0。
        nn.init.normal_(self.c_attn.weight, std=0.02)
        nn.init.zeros_(self.c_attn.bias)
        nn.init.normal_(self.c_proj.weight, std=0.02)
        nn.init.zeros_(self.c_proj.bias)

    # 对每个头计算注意力分数、遮住未来位置，再汇总 V 并拼回各头。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """接收 (B, T, C) 的浮点输入并返回 (B, T, C) 的注意力结果。

        B 是批量大小，T 是序列长度，C 是向量维度。
        H 是注意力头数，D 是每头的向量维度。
        每个位置的结果只依赖该位置及其之前的输入。
        """
        if x.ndim != 3 or x.shape[-1] != self.c_attn.in_features:
            raise ValueError("x must have shape (batch_size, sequence_length, embedding_dim)")
        batch_size, sequence_length, embedding_dim = x.shape

        qkv = self.c_attn(x)  # (B, T, 3C)
        qkv = qkv.reshape(
            batch_size, sequence_length, 3, self.num_heads, self.head_dim
        )  # (B, T, 3, H, D)
        # 把 Q/K/V 放到最前面，取出三个形状为 (B, H, T, D) 的张量。
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D)
        query, key, value = qkv.unbind(dim=0)

        # 每行是一处查询位置，每列是可供比较的一个键位置。
        transposed_key = key.transpose(-2, -1)  # (B, H, D, T)
        scores = (query @ transposed_key) / math.sqrt(self.head_dim)  # (B, H, T, T)

        positions = torch.arange(sequence_length, device=x.device)  # (T,)
        future_mask = positions.unsqueeze(0) > positions.unsqueeze(1)  # (T, T)
        # 遮罩对所有序列和注意力头生效，未来位置经过 softmax 后的权重为零。
        scores = scores.masked_fill(future_mask, float("-inf"))  # (B, H, T, T)
        weights = torch.softmax(scores, dim=-1)  # (B, H, T, T)

        context = weights @ value  # (B, H, T, D)
        context = context.transpose(1, 2)  # (B, T, H, D)
        context = context.reshape(batch_size, sequence_length, embedding_dim)  # (B, T, C)
        return self.c_proj(context)  # (B, T, C)


class MLP(nn.Module):
    """在每个位置独立处理特征，先扩到 4C 再投影回 C。"""

    # 建立升维线性层、GELU 激活和输出投影，同一套参数用于所有位置。
    def __init__(self, embedding_dim: int) -> None:
        """用向量维度 C 创建 MLP，两个线性层的权重是 (4C, C) 和 (C, 4C)。"""
        super().__init__()
        hidden_dim = 4 * embedding_dim
        self.c_fc = nn.Linear(embedding_dim, hidden_dim)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(hidden_dim, embedding_dim)

        # 两个线性层沿用原版 GPT-2 的基础初始化。
        nn.init.normal_(self.c_fc.weight, std=0.02)
        nn.init.zeros_(self.c_fc.bias)
        nn.init.normal_(self.c_proj.weight, std=0.02)
        nn.init.zeros_(self.c_proj.bias)

    # 仅沿每个位置的特征维变换，保持批量大小和序列长度。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """把 (B, T, C) 的输入经过 (B, T, 4C) 的中间层变回 (B, T, C)。"""
        expanded = self.c_fc(x)  # (B, T, 4C)
        activated = self.gelu(expanded)  # (B, T, 4C)
        return self.c_proj(activated)  # (B, T, C)


class TransformerBlock(nn.Module):
    """按 GPT-2 的 Pre-LayerNorm 顺序执行注意力和 MLP 两次残差更新。"""

    # 给注意力和 MLP 各配一份独立的 LayerNorm，复用已实现的因果自注意力。
    def __init__(self, embedding_dim: int, num_heads: int) -> None:
        """用向量维度 C 和头数 H 创建一个 block，每个子模块都接收 C 维向量。"""
        super().__init__()
        self.ln_1 = nn.LayerNorm(embedding_dim, eps=1e-5)
        self.attn = CausalSelfAttention(embedding_dim, num_heads)
        self.ln_2 = nn.LayerNorm(embedding_dim, eps=1e-5)
        self.mlp = MLP(embedding_dim)

    # 两条分支都先归一化，分支的结果再加回进入这一段计算时的 x。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """接收 (B, T, C) 的输入，依次执行注意力和 MLP，返回 (B, T, C)。"""
        attention_input = self.ln_1(x)  # (B, T, C)
        attention_output = self.attn(attention_input)  # (B, T, C)
        x = x + attention_output  # (B, T, C)

        # 第二次归一化使用已经加过注意力结果的 x。
        mlp_input = self.ln_2(x)  # (B, T, C)
        mlp_output = self.mlp(mlp_input)  # (B, T, C)
        return x + mlp_output  # (B, T, C)


class GPT2(nn.Module):
    """堆叠因果 Transformer block，为每个位置预测下一个 token。"""

    # 组装输入嵌入、L 个 block、最终归一化以及与 token 表共享权重的输出层。
    def __init__(
        self,
        vocab_size: int = 50_257,
        max_seq_len: int = 1024,
        embedding_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
    ) -> None:
        """默认构建 GPT-2 124M，可传入较小的尺寸进行练习和测试。"""
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.embeddings = GPT2Embeddings(vocab_size, max_seq_len, embedding_dim)
        self.blocks = nn.ModuleList(
            TransformerBlock(embedding_dim, num_heads) for _ in range(num_layers)
        )
        self.ln_f = nn.LayerNorm(embedding_dim, eps=1e-5)
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)

        # wte 用 (V, C) 的权重查向量，lm_head 用同一张表给 V 个候选打分。
        self.lm_head.weight = self.embeddings.wte.weight

        # 每个 block 有两条残差分支，按整网的分支数缩放输出投影的初始标准差。
        residual_std = 0.02 / math.sqrt(2 * num_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.c_proj.weight, std=residual_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=residual_std)

    # 用已有前缀生成每个位置的词表分数，最后一维保留原始 logits。
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """接收 (B, T) 的整数 ID，返回 (B, T, V) 的下一个 token 分数。

        B 是批量大小，T 是输入长度，C 是向量维度，V 是词表大小。
        第 t 个位置的分数只依赖第 0 到 t 个位置的输入。
        """
        x = self.embeddings(input_ids)  # (B, T) → (B, T, C)
        for block in self.blocks:
            x = block(x)  # (B, T, C)
        x = self.ln_f(x)  # (B, T, C)
        return self.lm_head(x)  # (B, T, V)
