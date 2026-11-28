"""记录训练阶段耗时与 NVIDIA 板卡读数，供不同训练配置复用。"""

import csv
import json
import shutil
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from tempfile import mkdtemp
from threading import Event, Thread
from typing import TextIO

import torch


GPU_SAMPLE_INTERVAL_S = 1.0
GPU_FIELDS = {
    "utilization.gpu": "gpu_busy_pct",
    "utilization.memory": "memory_busy_pct",
    "memory.used": "vram_used_mib",
    "memory.total": "vram_total_mib",
    "power.draw": "board_power_w",
    "enforced.power.limit": "power_limit_w",
    "temperature.gpu": "temperature_c",
    "clocks.sm": "sm_clock_mhz",
    "clocks.mem": "memory_clock_mhz",
}


def parse_gpu_sample(output: str) -> dict[str, float | None]:
    """读取一张卡的 CSV 行，驱动不提供的指标在 JSON 中记为 null。"""
    row, = csv.reader(output.strip().splitlines(), skipinitialspace=True)
    return {
        name: None if value.strip("[] ").lower() in {"n/a", "not supported"} else float(value)
        for name, value in zip(GPU_FIELDS.values(), row, strict=True)
    }


class TrainingPerformance:
    """计时点沿用训练本身的同步边界，采样线程独立查询板卡。"""

    def __init__(self, log: TextIO, tokens_per_step: int) -> None:
        self.log = log
        self.tokens_per_step = tokens_per_step
        self.started = time.monotonic()
        self.phase_seconds = dict.fromkeys(("update", "evaluation", "checkpoint", "generation"), 0.0)
        self.activity: tuple[str, int | None] = ("between_phases", None)
        self.updates = 0
        self.gpu_samples = 0
        self.gpu_error: str | None = None

    @contextmanager
    def measure(self, phase: str, step: int) -> Iterator[None]:
        """累计这段操作的墙钟时间，给同期 GPU 样本标注阶段和步号。"""
        started = time.monotonic()
        self.activity = (phase, step)
        try:
            yield
            if phase == "update":
                self.updates += 1
        finally:
            self.activity = ("between_phases", None)
            self.phase_seconds[phase] += time.monotonic() - started

    def record(self, event: str, **fields) -> None:
        """持续写入事件，途中停止时仍可读到已写出的样本。"""
        self.log.write(json.dumps({"event": event, **fields}, allow_nan=False) + "\n")

    def timing(self) -> dict:
        """给出本段累计耗时和更新吞吐，续训时从本段的首个更新开始计数。"""
        update_s = self.phase_seconds["update"]
        return {
            "completed_updates": self.updates,
            "phase_seconds": {name: round(seconds, 6) for name, seconds in self.phase_seconds.items()},
            "update_tokens_per_s": round(self.updates * self.tokens_per_step / update_s, 2) if update_s else None,
            "elapsed_s": round(time.monotonic() - self.started, 6),
        }

    def sample_gpu(self, command: list[str], path: Path, stop: Event) -> None:
        """在后台按秒采样。单次查询有超时，采样失败的原因留在训练日志。"""
        try:
            with path.open("x", encoding="utf-8", buffering=1) as log:
                while not stop.is_set():
                    started = time.monotonic()
                    phase, step = self.activity
                    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=3)
                    sample = parse_gpu_sample(result.stdout)
                    log.write(json.dumps({
                        "elapsed_s": round(started - self.started, 6),
                        "phase_at_query": phase, "step": step, **sample,
                    }, allow_nan=False) + "\n")
                    self.gpu_samples += 1
                    stop.wait(max(0, GPU_SAMPLE_INTERVAL_S - (time.monotonic() - started)))
        except subprocess.CalledProcessError as error:
            self.gpu_error = (error.stderr or error.stdout or str(error)).strip()
        except (OSError, subprocess.TimeoutExpired, ValueError, csv.Error) as error:
            self.gpu_error = str(error)


@contextmanager
def monitor_training(
    checkpoint: Path, device: torch.device, metadata: dict, *, sample_gpu: bool = True,
) -> Iterator[TrainingPerformance]:
    """为本段创建独立日志。断点续训、新实验及并行实验各自保留原始数据。"""
    directory = checkpoint.with_name(checkpoint.name + ".performance")
    directory.mkdir(parents=True, exist_ok=True)
    run = Path(mkdtemp(prefix=f"step-{metadata['start_step']:06d}-", dir=directory))
    gpu_hardware = {}
    command = None
    gpu_status = "not_cuda"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        properties = torch.cuda.get_device_properties(device)
        gpu_id = "GPU-" + str(properties.uuid).removeprefix("GPU-")
        gpu_hardware = {"gpu_name": properties.name, "gpu_uuid": gpu_id, "device_total_bytes": properties.total_memory}
        gpu_status = "disabled"
        if sample_gpu:
            smi = shutil.which("nvidia-smi")
            gpu_status = "nvidia_smi_missing"
            if smi is not None:
                command = [
                    smi, f"--id={gpu_id}", f"--query-gpu={','.join(GPU_FIELDS)}",
                    "--format=csv,noheader,nounits",
                ]
                gpu_status = "sampling"

    with (run / "training.jsonl").open("x", encoding="utf-8", buffering=1) as log:
        performance = TrainingPerformance(log, metadata["tokens_per_step"])
        performance.record(
            "start", **metadata, **gpu_hardware, torch_version=torch.__version__,
            started_at_utc=datetime.now(UTC).isoformat(), gpu_status=gpu_status,
            gpu_sample_interval_s=GPU_SAMPLE_INTERVAL_S if command else None,
        )
        print(json.dumps({"performance_log": str(run), "gpu_status": gpu_status}), flush=True)
        stop = Event()
        worker = None
        if command:
            worker = Thread(target=performance.sample_gpu, args=(command, run / "gpu.jsonl", stop))
            worker.start()
        status = "interrupted"
        try:
            yield performance
            status = "complete"
        finally:
            timing = performance.timing()
            stop.set()
            if worker is not None:
                worker.join()
                gpu_status = "error" if performance.gpu_error else ("ok" if performance.gpu_samples else "no_samples")
            peaks = {}
            if device.type == "cuda":
                peaks = {
                    "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                }
            summary = {
                "status": status, **timing, **peaks, "gpu_status": gpu_status,
                "gpu_samples": performance.gpu_samples, "gpu_error": performance.gpu_error,
            }
            performance.record("end", **summary)
            print(json.dumps({"performance_summary": summary}), flush=True)
