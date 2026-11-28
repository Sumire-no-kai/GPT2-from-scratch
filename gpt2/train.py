"""在仓库根目录用 python -m gpt2.train 从零训练字符级 GPT，产物默认留在 checkpoints/。"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from gpt2.model import GPT2
from gpt2.performance import monitor_training
from gpt2.training import (
    TrainConfig, evaluate, learning_rate, load_checkpoint, make_optimizer, read_corpus,
    save_checkpoint, train_step,
)


def run_training(
    data_path: Path,
    checkpoint: Path,
    config: TrainConfig,
    device: torch.device,
    *,
    resume: bool = False,
    stop_after: int | None = None,
    eval_interval: int = 100,
    eval_batches: int = 12,
    monitor_gpu: bool = True,
) -> None:
    """训练到指定步数，记录耗时、评估 loss 并保存断点。CUDA 默认采集板卡读数。"""
    end = config.steps if stop_after is None else stop_after
    if not 1 <= end <= config.steps or min(eval_interval, eval_batches) <= 0:
        raise ValueError("invalid stop step or evaluation interval/batch count")
    if checkpoint.exists() and not resume:
        raise FileExistsError(f"{checkpoint} already exists; choose a new path or use --resume")
    tokenizer, train, val, digest = read_corpus(data_path, config.block_size)
    alphabet = tokenizer.decode(list(range(tokenizer.vocab_size)))
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")  # 剩余 float32 矩阵乘允许使用 TF32。
    # 直接在目标设备创建共享的 token 表，再把它交给优化器。
    with torch.device(device):
        model = GPT2(
            vocab_size=tokenizer.vocab_size,
            max_seq_len=config.block_size,
            embedding_dim=config.embedding_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
        )
    optimizer = make_optimizer(model, config)
    sampler = torch.Generator().manual_seed(config.seed)  # 独立 CPU 抽样器，状态即数据进度。
    start = 0
    if resume:
        start = load_checkpoint(checkpoint, model, optimizer, sampler, config, digest)
    if end < start:
        raise ValueError("stop step precedes the loaded checkpoint")
    tokens_per_step = config.batch_size * config.block_size * config.accumulation_steps
    metadata = {
        "device": str(device), "parameters": sum(p.numel() for p in model.parameters()),
        "train_chars": len(train), "val_chars": len(val), "vocab": tokenizer.vocab_size,
        "start_step": start, "tokens_per_step": tokens_per_step,
    }
    print(json.dumps(metadata), flush=True)

    # 从初始或恢复位置评估；之后的每行都对应已经完成该步更新的权重。
    run_metadata = dict(
        metadata, config=asdict(config), end_step=end, eval_interval=eval_interval, eval_batches=eval_batches,
    )
    with monitor_training(checkpoint, device, run_metadata, sample_gpu=monitor_gpu) as performance:
        started = time.monotonic()
        for step in range(start, end + 1):
            if step > start:
                with performance.measure("update", step):
                    train_step(model, optimizer, train, sampler, config, step)
            if step == start or step == end or step % eval_interval == 0:
                with performance.measure("evaluation", step):
                    train_loss = evaluate(model, train, config, eval_batches)
                    val_loss = evaluate(model, val, config, eval_batches)
                metrics = {
                    "step": step, "tokens_seen": step * tokens_per_step,
                    "lr": learning_rate(step, config) if step else 0.0,
                    "train_loss": train_loss, "val_loss": val_loss,
                    "elapsed_s": round(time.monotonic() - started, 3),
                }
                if step > start:
                    with performance.measure("checkpoint", step):
                        save_checkpoint(checkpoint, model, optimizer, sampler, config, step, digest, alphabet)
                print(json.dumps(metrics), flush=True)
                performance.record("progress", **(metrics | performance.timing()))
        if end == config.steps:
            with performance.measure("generation", end):
                model.eval()
                prompt = torch.zeros((1, 1), dtype=torch.long, device=device)  # (1,1)，0 是换行。
                sample = model.generate(prompt, 160, do_sample=True, temperature=0.8, top_k=20)
                text = tokenizer.decode(sample[0].tolist())
            print(json.dumps({"sample": text}), flush=True)


def main() -> None:
    """解析命令行并选用 CUDA、MPS 或 CPU。--resume 要沿用首次训练的语料和超参数。"""
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/input.txt"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/shakespeare.pt"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int, help="本次运行的最后一步，保留 --steps 的学习率日程")
    parser.add_argument("--steps", type=int, default=defaults.steps)
    parser.add_argument("--warmup-steps", type=int, default=defaults.warmup_steps)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size, help="单个微批的段数")
    parser.add_argument("--accumulation-steps", type=int, default=defaults.accumulation_steps)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=12)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"))
    parser.add_argument("--no-gpu-monitor", action="store_true", help="关闭 NVIDIA 板卡采样，仍记录耗时与 CUDA 显存峰值")
    args = parser.parse_args()
    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    config = TrainConfig(
        steps=args.steps, warmup_steps=args.warmup_steps,
        batch_size=args.batch_size, accumulation_steps=args.accumulation_steps,
    )
    run_training(
        args.data, args.checkpoint, config, torch.device(device),
        resume=args.resume, stop_after=args.stop_after,
        eval_interval=args.eval_interval, eval_batches=args.eval_batches, monitor_gpu=not args.no_gpu_monitor,
    )


if __name__ == "__main__":
    main()
