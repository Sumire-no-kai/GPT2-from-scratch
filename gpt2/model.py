"""GPT-2 的模型结构，从输入 token ID 到下一个 token 的分数。"""

import math

import torch
from torch import nn
from torch.nn import functional as F


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
    def __init__(self, embedding_dim: int, num_heads: int, *, use_sdpa: bool = False) -> None:
        """用向量维度 C 和头数 H 创建注意力，要求 C 可被 H 整除。

        每头处理 D = C // H 个数。两个线性层的权重形状为 (3C, C) 和 (C, C)。
        use_sdpa 让 PyTorch 根据输入选择 SDPA 后端，默认走显式的参考实现。
        """
        super().__init__()
        if embedding_dim <= 0 or num_heads <= 0 or embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be a positive multiple of num_heads")

        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.use_sdpa = use_sdpa
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

        if self.use_sdpa:
            # PyTorch 在内核里缩放并施加因果遮罩。关闭 dropout 以对齐参考实现。
            context = F.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=True,
            )  # (B, H, T, D)
        else:
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
    def __init__(self, embedding_dim: int, num_heads: int, *, use_sdpa: bool = False) -> None:
        """创建一个接收 C 维向量的 block，并把 use_sdpa 传给注意力层。"""
        super().__init__()
        self.ln_1 = nn.LayerNorm(embedding_dim, eps=1e-5)
        self.attn = CausalSelfAttention(embedding_dim, num_heads, use_sdpa=use_sdpa)
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
        *,
        use_sdpa: bool = False,
    ) -> None:
        """默认构建 GPT-2 124M，设 use_sdpa=True 让所有 block 使用 SDPA。"""
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.embeddings = GPT2Embeddings(vocab_size, max_seq_len, embedding_dim)
        self.blocks = nn.ModuleList(
            TransformerBlock(embedding_dim, num_heads, use_sdpa=use_sdpa)
            for _ in range(num_layers)
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
