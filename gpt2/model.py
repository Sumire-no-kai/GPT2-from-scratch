"""GPT-2 的输入嵌入和多头因果自注意力。"""

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
