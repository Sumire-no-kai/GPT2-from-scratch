"""共用的训练运算与阶段 5 的字符级数据入口。"""

import hashlib
import math
from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from torch.nn import functional as F

from gpt2.model import GPT2
from warmup.bigram import CharTokenizer, get_batch


@dataclass(frozen=True)
class TrainConfig:
    """一次训练的固定设置。batch_size 是一次前向的段数，steps 是总更新次数。"""

    block_size: int = 256
    num_layers: int = 6
    num_heads: int = 6
    embedding_dim: int = 384
    batch_size: int = 16
    accumulation_steps: int = 2
    steps: int = 1200
    warmup_steps: int = 50
    peak_lr: float = 1e-3
    min_lr: float = 1e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    seed: int = 1337

    def __post_init__(self) -> None:
        """在分配模型或读取训练数据前检查窗口、批量和学习率日程。"""
        if min(self.block_size, self.batch_size, self.accumulation_steps) < 1:
            raise ValueError("block_size, batch_size and accumulation_steps must be positive")
        if not 0 <= self.warmup_steps < self.steps:
            raise ValueError("warmup_steps must be in [0, steps)")
        if not 0 < self.min_lr <= self.peak_lr or self.grad_clip <= 0:
            raise ValueError("expected 0 < min_lr <= peak_lr and positive grad_clip")


def read_corpus(path: Path, block_size: int) -> tuple[CharTokenizer, torch.Tensor, torch.Tensor, str]:
    """按阶段 0 的字符表编码，返回词表、两段一维 CPU long 张量和语料指纹。"""
    text = path.read_text(encoding="utf-8")
    cut = int(0.9 * len(text))
    if min(cut, len(text) - cut) <= block_size:
        raise ValueError("each split needs at least block_size + 1 characters")
    tokenizer = CharTokenizer(text)
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)  # (N,)
    # 先分开再在每段内部抽窗口，最后一个训练目标也落在训练集。
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return tokenizer, ids[:cut], ids[cut:], digest


def learning_rate(step: int, config: TrainConfig) -> float:
    """第 step 次更新使用的学习率（从 1 计数）。线性预热后沿余弦降至 min_lr。"""
    if not 1 <= step <= config.steps:
        raise ValueError("step outside training schedule")
    peak_step = max(1, config.warmup_steps)  # 关闭预热时，第一步直接用峰值。
    if step <= peak_step:
        return config.peak_lr * step / peak_step
    progress = (step - peak_step) / (config.steps - peak_step)
    cosine = (1.0 + math.cos(math.pi * progress)) / 2.0
    return config.min_lr + (config.peak_lr - config.min_lr) * cosine


def make_optimizer(model: GPT2, config: TrainConfig) -> torch.optim.AdamW:
    """矩阵使用 AdamW 的权重衰减，偏置和归一化的一维参数只做梯度更新。"""
    # parameters() 会消除共享引用，token 表与输出层共享的矩阵只会更新一次。
    matrices = [p for p in model.parameters() if p.ndim >= 2]
    vectors = [p for p in model.parameters() if p.ndim < 2]
    groups = [
        {"params": matrices, "weight_decay": config.weight_decay},
        {"params": vectors, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=config.peak_lr, betas=(0.9, 0.95))


def batch_loss(model: GPT2, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """输入和标签都是 (B,T)。返回这 B×T 个下一个 token 的平均负对数概率（标量）。"""
    device = model.embeddings.wte.weight.device
    # 在原生支持 bf16 的 CUDA 卡上只降低部分前向运算的精度，主权重仍是 float32。
    precision = nullcontext()
    if device.type == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False):
        precision = torch.autocast("cuda", dtype=torch.bfloat16)
    with precision:
        logits = model(x.to(device))  # (B,T,V)，V 是词表大小。
    # y 是向后移动一格的目标。按同一顺序展平 B 和 T，保留 V 个类别。
    return F.cross_entropy(logits.float().flatten(0, 1), y.to(device).reshape(-1))


def train_step(
    model: GPT2, optimizer: torch.optim.Optimizer, data: torch.Tensor,
    sampler: torch.Generator, config: TrainConfig, step: int,
) -> float:
    """从训练集抽取等大的微批，累积平均梯度、裁剪并做一次更新，返回更新前的平均 loss。"""
    batches = (
        get_batch(data, config.block_size, config.batch_size, generator=sampler)
        for _ in range(config.accumulation_steps)
    )
    return update_parameters(model, optimizer, batches, config, step)


def update_parameters(
    model: GPT2, optimizer: torch.optim.Optimizer,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]], config: TrainConfig, step: int,
) -> float:
    """对 config.accumulation_steps 个等大的 (B,T) 微批做一次参数更新。"""
    model.train()
    rate = learning_rate(step, config)
    for group in optimizer.param_groups:
        group["lr"] = rate
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for x, y in batches:
        loss = batch_loss(model, x, y)  # 每个微批都有 B×T 个有效目标。
        (loss / config.accumulation_steps).backward()
        losses.append(loss.detach())  # 只为日志保留标量，可释放这个微批的计算图。
    if len(losses) != config.accumulation_steps:
        raise ValueError("microbatch count differs from accumulation_steps")
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, error_if_nonfinite=True)
    optimizer.step()
    return torch.stack(losses).mean().item()


def evaluate(model: GPT2, data: torch.Tensor, config: TrainConfig, batches: int) -> float:
    """用固定窗口估算一个数据分区的平均 loss；评估不消耗训练抽样器的状态。"""
    if batches < 1:
        raise ValueError("evaluation needs at least one batch")
    sampler = torch.Generator().manual_seed(config.seed + 1)
    windows = (
        get_batch(data, config.block_size, config.batch_size, generator=sampler)
        for _ in range(batches)
    )
    return evaluate_batches(model, windows)


@torch.no_grad()
def evaluate_batches(model: GPT2, batches: Iterable[tuple[torch.Tensor, torch.Tensor]]) -> float:
    """计算等大的 (B,T) 微批的平均 loss，并恢复调用之前的训练模式。"""
    was_training = model.training
    model.eval()
    try:
        losses = []
        for x, y in batches:
            losses.append(batch_loss(model, x, y))
        return torch.stack(losses).mean().item()
    finally:
        model.train(was_training)


def save_checkpoint(
    path: Path, model: GPT2, optimizer: torch.optim.Optimizer,
    sampler: torch.Generator, config: TrainConfig, step: int,
    corpus_sha256: str, alphabet: str,
) -> None:
    """在完整更新之后保存。先写同目录临时文件，写完再替换上一个可恢复的断点。"""
    device = model.embeddings.wte.weight.device
    state = {
        "step": step, "config": asdict(config), "alphabet": alphabet, "corpus_sha256": corpus_sha256,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "sampler_rng": sampler.get_state(), **capture_rng(device),
    }
    atomic_save(path, state)


def capture_rng(device: torch.device) -> dict:
    """保存 CPU 与当前计算设备的随机状态。"""
    state = {"cpu_rng": torch.get_rng_state(), "backend": device.type}
    if device.type != "cpu":
        state["device_rng"] = torch.get_device_module(device).get_rng_state(device)
    return state


def restore_rng(state: dict, device: torch.device) -> None:
    """相同后端时恢复设备随机数，迁移设备时复用 CPU 状态。"""
    torch.set_rng_state(state["cpu_rng"])
    if device.type != "cpu" and device.type == state["backend"]:
        torch.get_device_module(device).set_rng_state(state["device_rng"], device)


def atomic_save(path: Path, state: dict) -> None:
    """先在同目录写完存档，再替换目标。中途失败保留上一份完整断点。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = NamedTemporaryFile(dir=path.parent, delete=False)
    tmp_path = Path(tmp.name)
    try:
        with tmp:
            torch.save(state, tmp)
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


def load_checkpoint(
    path: Path, model: GPT2, optimizer: torch.optim.Optimizer,
    sampler: torch.Generator, config: TrainConfig, corpus_sha256: str,
) -> int:
    """恢复权重、AdamW 和抽样位置，返回已完成步数。相同后端时也恢复设备随机状态。"""
    # 在 CPU 解包，随机数状态需保留为 CPU 字节张量；模型与优化器各自加载到参数设备。
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["config"] != asdict(config) or state["corpus_sha256"] != corpus_sha256:
        raise ValueError("training config or corpus differs from checkpoint")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    sampler.set_state(state["sampler_rng"])
    restore_rng(state, model.embeddings.wte.weight.device)
    return state["step"]
