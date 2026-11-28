"""性能日志的可执行验收，本机用模拟驱动返回值覆盖 CUDA 采样路径。"""

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

import torch

from gpt2.performance import monitor_training


class TestTrainingPerformance(unittest.TestCase):
    def run_with_gpu_sample(self, output: str) -> tuple[list[dict], list[dict], list[str]]:
        """启动实际采样线程，用事件等待第一次只读查询及其收尾。"""
        queried = Event()

        def read_sample(command, **kwargs):
            queried.set()
            return SimpleNamespace(stdout=output)

        properties = SimpleNamespace(name="NVIDIA test GPU", uuid="00000000-0000-0000-0000-000000000abc", total_memory=2**34)
        with (
            TemporaryDirectory() as folder,
            redirect_stdout(io.StringIO()),
            patch("shutil.which", return_value="/mock/nvidia-smi"),
            patch("torch.cuda.get_device_properties", return_value=properties),
            patch("torch.cuda.reset_peak_memory_stats"),
            patch("torch.cuda.max_memory_allocated", return_value=1048576),
            patch("torch.cuda.max_memory_reserved", return_value=2097152),
            patch("subprocess.run", side_effect=read_sample) as mock_query,
        ):
            path = Path(folder) / "model.pt"
            with monitor_training(path, torch.device("cuda:2"), {"start_step": 400, "tokens_per_step": 32}) as monitor:
                with monitor.measure("update", 401):
                    self.assertTrue(queried.wait(5), "the sampler did not issue its query")
            run, = path.with_name("model.pt.performance").iterdir()
            training = [json.loads(line) for line in (run / "training.jsonl").read_text().splitlines()]
            samples = [json.loads(line) for line in (run / "gpu.jsonl").read_text().splitlines()]
            return training, samples, mock_query.call_args.args[0]

    def test_cuda_samples_use_device_uuid_and_capture_units_and_peak_memory(self):
        training, samples, command = self.run_with_gpu_sample("87, 31, 3200, 40960, 117.25, 400, 63, 1410, 1215\n")
        self.assertIn("--id=GPU-00000000-0000-0000-0000-000000000abc", command)
        self.assertEqual(training[0]["start_step"], 400)
        self.assertEqual(training[-1]["gpu_status"], "ok")
        self.assertEqual(training[-1]["gpu_samples"], len(samples))
        self.assertEqual(training[-1]["completed_updates"], 1)
        self.assertEqual(training[-1]["torch_peak_allocated_bytes"], 1048576)
        self.assertEqual(training[-1]["torch_peak_reserved_bytes"], 2097152)
        self.assertEqual(samples[0]["gpu_busy_pct"], 87)
        self.assertEqual(samples[0]["memory_busy_pct"], 31)
        self.assertEqual(samples[0]["vram_used_mib"], 3200)
        self.assertEqual(samples[0]["board_power_w"], 117.25)

    def test_missing_sensor_is_null_and_malformed_output_is_reported(self):
        training, samples, _ = self.run_with_gpu_sample("87, 31, 3200, 40960, [Not Supported], N/A, 63, 1410, 1215\n")
        self.assertEqual(training[-1]["gpu_status"], "ok")
        self.assertIsNone(samples[0]["board_power_w"])
        self.assertIsNone(samples[0]["power_limit_w"])
        training, samples, _ = self.run_with_gpu_sample("87, 31\n")
        self.assertEqual(training[-1]["gpu_status"], "error")
        self.assertTrue(training[-1]["gpu_error"])
        self.assertEqual(training[-1]["status"], "complete")
        self.assertEqual(samples, [])


if __name__ == "__main__":
    unittest.main()
