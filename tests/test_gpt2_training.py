"""阶段 5 的可执行验收，使用极小的 CPU 模型和临时断点。"""

import copy
import io
import json
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from gpt2.model import GPT2
from gpt2.train import run_training
from gpt2.training import (
    TrainConfig, batch_loss, evaluate, learning_rate, load_checkpoint, make_optimizer,
    read_corpus, save_checkpoint, train_step,
)
from warmup.bigram import get_batch


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        self.cfg = TrainConfig(block_size=4, num_layers=1, num_heads=2, embedding_dim=8,
                               batch_size=3, accumulation_steps=2, steps=6, warmup_steps=2)
        self.model = GPT2(5, 4, 8, 2, 1)
        self.data = torch.randint(5, (80,))

    def test_corpus_split_precedes_sampling_and_retains_the_last_target(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "text.txt"
            path.write_text("abcd " * 9 + "雪山河海\n", encoding="utf-8")
            tok, train, val, digest = read_corpus(path, 4)
            self.assertEqual(tok.decode(train.tolist()), "abcd " * 9)
            self.assertEqual(tok.decode(val.tolist()), "雪山河海\n")
            self.assertEqual(train.dtype, torch.long)
            self.assertEqual(len(digest), 64)
            x, y = get_batch(val, 4, 2, generator=torch.Generator().manual_seed(1))
            self.assertEqual(tok.decode(x[0].tolist()), "雪山河海")
            self.assertEqual(tok.decode(y[0].tolist()), "山河海\n")
            with self.assertRaises(ValueError):
                read_corpus(path, 5)

    def test_loss_averages_the_correct_target_at_every_position(self):
        x = torch.tensor([[0, 1, 2], [4, 0, 3]])
        y = torch.tensor([[1, 2, 3], [0, 3, 2]])
        log_probs = self.model(x).log_softmax(dim=-1)
        expected = -sum(log_probs[b, t, y[b, t]] for b in range(2) for t in range(3)) / 6
        torch.testing.assert_close(batch_loss(self.model, x, y), expected)

    def test_schedule_at_warmup_midpoint_and_final_update(self):
        c = self.cfg
        for step, expected in [(1, c.peak_lr / 2), (2, c.peak_lr),
                               (4, (c.peak_lr + c.min_lr) / 2), (6, c.min_lr)]:
            self.assertAlmostEqual(learning_rate(step, c), expected)
        flat_start = replace(c, warmup_steps=0)
        self.assertEqual(learning_rate(1, flat_start), c.peak_lr)
        self.assertEqual(learning_rate(6, flat_start), c.min_lr)
        with self.assertRaises(ValueError):
            learning_rate(0, c)

    def test_adamw_updates_shared_matrix_once_and_only_decays_matrices(self):
        opt = make_optimizer(self.model, self.cfg)
        self.assertIsInstance(opt, torch.optim.AdamW)
        params = []
        for group in opt.param_groups:
            for p in group["params"]:
                params.append(id(p))
                self.assertEqual(group["weight_decay"], self.cfg.weight_decay if p.ndim >= 2 else 0.0)
        self.assertCountEqual(params, [id(p) for p in self.model.parameters()])
        self.assertEqual(params.count(id(self.model.lm_head.weight)), 1)

    def test_accumulation_and_clipping_match_one_large_batch_over_two_updates(self):
        cfg = replace(self.cfg, peak_lr=0.1, grad_clip=0.1)
        other = copy.deepcopy(self.model)
        opt = torch.optim.SGD(self.model.parameters(), lr=1)
        sampler = torch.Generator().manual_seed(32)
        reference_sampler = torch.Generator().manual_seed(32)
        for step in (1, 2):
            actual = train_step(self.model, opt, self.data, sampler, cfg, step)
            x, y = get_batch(self.data, cfg.block_size, cfg.batch_size * cfg.accumulation_steps,
                             generator=reference_sampler)
            other.zero_grad()
            expected = batch_loss(other, x, y)
            expected.backward()
            norm = torch.nn.utils.clip_grad_norm_(other.parameters(), cfg.grad_clip)
            self.assertGreater(norm.item(), cfg.grad_clip)
            with torch.no_grad():
                for p in other.parameters():
                    p.add_(p.grad, alpha=-learning_rate(step, cfg))
            self.assertAlmostEqual(actual, expected.item(), places=5)
            torch.testing.assert_close(self.model.state_dict(), other.state_dict(), rtol=2e-5, atol=1e-7)

    def test_evaluation_uses_fixed_windows_without_gradients_or_rng_side_effects(self):
        cfg = self.cfg
        gen = torch.Generator().manual_seed(cfg.seed + 1)
        expected = sum(batch_loss(self.model, *get_batch(self.data, 4, 3, generator=gen)).item()
                       for _ in range(3)) / 3
        modes = []
        hook = self.model.register_forward_pre_hook(
            lambda model, _: modes.append((model.training, torch.is_grad_enabled()))
        )
        before = torch.get_rng_state()
        try:
            for mode in (True, False):
                self.model.train(mode)
                self.assertAlmostEqual(evaluate(self.model, self.data, cfg, 3), expected, places=5)
                self.assertEqual(self.model.training, mode)
        finally:
            hook.remove()
        self.assertEqual(modes, [(False, False)] * 6)
        self.assertTrue(torch.equal(torch.get_rng_state(), before))

    def test_conflict_free_fixed_batch_overfits(self):
        # 唯一合法窗口。0 在不同位置后面分别跟 1、2、3、4，迫使模型使用上下文或位置。
        data = torch.tensor([0, 1, 0, 2, 0, 3, 0, 4, 0])
        cfg = replace(self.cfg, block_size=8, embedding_dim=32, steps=100, warmup_steps=4,
                      batch_size=1, accumulation_steps=1, peak_lr=0.01, min_lr=0.001)
        model = GPT2(5, 8, 32, 2, 1)
        opt = make_optimizer(model, cfg)
        gen = torch.Generator().manual_seed(9)
        for step in range(1, cfg.steps + 1):
            train_step(model, opt, data, gen, cfg, step)
        x, y = get_batch(data, 8, 1)
        self.assertLess(batch_loss(model, x, y).item(), 0.02)
        self.assertTrue(torch.equal(model(x).argmax(dim=-1), y))

    def test_checkpoint_restores_next_batch_rng_adamw_and_exact_cpu_trajectory(self):
        cfg = self.cfg
        sampler = torch.Generator().manual_seed(99)
        opt = make_optimizer(self.model, cfg)
        with TemporaryDirectory() as folder:
            path = Path(folder) / "step.pt"
            for step in (1, 2, 3):
                train_step(self.model, opt, self.data, sampler, cfg, step)
            save_checkpoint(path, self.model, opt, sampler, cfg, 3, "test-digest", "abcde")
            expected_rng = torch.rand(4)
            expected = [train_step(self.model, opt, self.data, sampler, cfg, step) for step in (4, 5, 6)]
            fresh = GPT2(5, 4, 8, 2, 1)
            opt2 = make_optimizer(fresh, cfg)
            gen2 = torch.Generator().manual_seed(2026)
            start = load_checkpoint(path, fresh, opt2, gen2, cfg, "test-digest")
            self.assertEqual(start, 3)
            self.assertTrue(torch.equal(torch.rand(4), expected_rng))
            actual = [train_step(fresh, opt2, self.data, gen2, cfg, step) for step in range(start + 1, 7)]
            self.assertEqual(actual, expected)
            self.assertTrue(torch.equal(sampler.get_state(), gen2.get_state()))
            torch.testing.assert_close(fresh.state_dict(), self.model.state_dict(), rtol=0, atol=0)
            torch.testing.assert_close(opt2.state_dict(), opt.state_dict(), rtol=0, atol=0)
            for conf, digest in [(replace(cfg, steps=7), "test-digest"), (cfg, "changed")]:
                with self.assertRaises(ValueError):
                    load_checkpoint(path, fresh, opt2, gen2, conf, digest)

    def test_failed_save_leaves_last_complete_checkpoint_in_place(self):
        opt = make_optimizer(self.model, self.cfg)
        with TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            path.write_bytes(b"previous")
            with patch("torch.save", side_effect=OSError("disk full")), self.assertRaises(OSError):
                save_checkpoint(path, self.model, opt, torch.Generator(), self.cfg, 1, "digest", "abcde")
            self.assertEqual(path.read_bytes(), b"previous")
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_runner_resumes_with_original_schedule_and_preserves_existing_run(self):
        with TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            path = Path(folder) / "text.txt"
            path.write_text("abcde" * 30, encoding="utf-8")
            uninterrupted = Path(folder) / "full.pt"
            resumed = Path(folder) / "resumed.pt"
            cfg = replace(self.cfg, steps=4, warmup_steps=2)
            args = dict(config=cfg, device=torch.device("cpu"), eval_batches=2, eval_interval=3)
            run_training(path, uninterrupted, **args)
            run_training(path, resumed, stop_after=2, **args)
            with self.assertRaises(FileExistsError):
                run_training(path, resumed, **args)
            run_training(path, resumed, resume=True, **args)
            full = torch.load(uninterrupted, weights_only=True)
            split = torch.load(resumed, weights_only=True)
            self.assertEqual(split["step"], 4)
            self.assertEqual(split["alphabet"], "abcde")
            self.assertEqual(full.keys(), split.keys())
            for name in ("step", "config", "alphabet", "corpus_sha256", "backend"):
                self.assertEqual(full[name], split[name])
            for name in ("model", "optimizer", "sampler_rng", "cpu_rng"):
                torch.testing.assert_close(full[name], split[name], rtol=0, atol=0)

            log_root = resumed.with_name("resumed.pt.performance")
            self.assertEqual(len(list(log_root.iterdir())), 2)
            for start in (0, 2):
                log, = log_root.glob(f"step-{start:06d}-*/training.jsonl")
                rows = [json.loads(line) for line in log.read_text().splitlines()]
                self.assertEqual(rows[0]["start_step"], start)
                self.assertEqual(rows[-1]["status"], "complete")
                self.assertEqual(rows[-1]["gpu_status"], "not_cuda")
                self.assertEqual(rows[-1]["completed_updates"], 2)
                self.assertGreater(rows[-1]["update_tokens_per_s"], 0)


if __name__ == "__main__":
    unittest.main()
