"""验证 GPT-2 的生成循环、上下文窗口和采样分布。"""

import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from gpt2.model import GPT2


class GPT2GenerateTests(unittest.TestCase):
    def setUp(self):
        self.model = GPT2(
            vocab_size=7, max_seq_len=4, embedding_dim=8, num_heads=2, num_layers=1
        )

    def cycling_logits(self, ids):
        # 每个位置给 (当前 ID + 1) % 7 最高分，两个不同的末尾会得到不同结果。
        self.assertFalse(torch.is_grad_enabled())
        return F.one_hot((ids.long() + 1) % 7, num_classes=7).float()

    def test_appends_the_greedy_token_for_each_row_and_preserves_input(self):
        ids = torch.tensor([[1, 4], [2, 5]])
        original = ids.clone()
        with patch.object(self.model, "forward", side_effect=self.cycling_logits):
            result = self.model.generate(ids, max_new_tokens=2)

        self.assertEqual(result.tolist(), [[1, 4, 5, 6], [2, 5, 6, 0]])
        torch.testing.assert_close(ids, original)

    def test_passes_only_the_latest_context_after_reaching_the_limit(self):
        ids = torch.tensor([[0, 2, 3]])
        with patch.object(self.model, "forward", side_effect=self.cycling_logits) as forward:
            result = self.model.generate(ids, max_new_tokens=3)

        self.assertEqual(result.tolist(), [[0, 2, 3, 4, 5, 6]])
        self.assertEqual(
            [call.args[0].tolist() for call in forward.call_args_list],
            [[[0, 2, 3]], [[0, 2, 3, 4]], [[2, 3, 4, 5]]],
        )

    def test_keeps_an_initial_prompt_longer_than_the_context(self):
        ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
        with patch.object(self.model, "forward", side_effect=self.cycling_logits) as forward:
            result = self.model.generate(ids, max_new_tokens=1)

        self.assertEqual(forward.call_args.args[0].tolist(), [[3, 4, 5, 6]])
        self.assertEqual(result.tolist(), [[1, 2, 3, 4, 5, 6, 0]])

    def test_zero_new_tokens_skips_the_forward_pass(self):
        ids = torch.tensor([[1, 3, 5]])
        with patch.object(self.model, "forward") as forward:
            result = self.model.generate(ids, max_new_tokens=0)

        forward.assert_not_called()
        torch.testing.assert_close(result, ids)

    def test_preserves_int32_token_ids_with_a_real_forward_pass(self):
        ids = torch.tensor([[1, 2]], dtype=torch.int32)
        result = self.model.generate(ids, max_new_tokens=1)

        self.assertEqual(result.shape, (1, 3))
        self.assertEqual(result.dtype, torch.int32)
        torch.testing.assert_close(result[:, :2], ids)
        self.assertTrue(((result >= 0) & (result < 7)).all())

    def test_rejects_empty_or_unbatched_ids(self):
        for shape in ((2, 0), (0, 2), (2,)):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "nonempty shape"):
                self.model.generate(torch.zeros(shape, dtype=torch.long), max_new_tokens=1)

    def test_rejects_noninteger_ids_and_negative_length(self):
        with self.assertRaisesRegex(TypeError, "torch.int32 or torch.int64"):
            self.model.generate(torch.zeros(1, 2), max_new_tokens=1)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            self.model.generate(torch.tensor([[1, 2]]), max_new_tokens=-1)


class GPT2SamplingTests(unittest.TestCase):
    def setUp(self):
        self.model = GPT2(7, 4, 8, num_heads=2, num_layers=1)

    def sample_once(self, scores, **options):
        # 从公开的 generate() 入口捕获传给 multinomial 的实际概率。
        ids = torch.zeros((scores.size(0), 1), dtype=torch.long)
        with (
            torch.random.fork_rng(devices=[]),
            patch.object(self.model, "forward", return_value=scores.unsqueeze(1)),
            patch("torch.multinomial", wraps=torch.multinomial) as sample,
        ):
            result = self.model.generate(ids, max_new_tokens=1, do_sample=True, **options)
        return result[:, -1], sample.call_args.args[0]

    def test_default_sampling_and_k_at_least_vocab_keep_the_distribution(self):
        scores = torch.tensor([[2.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0]])
        for dtype in (torch.float32, torch.float16, torch.bfloat16, torch.float64):
            with self.subTest(dtype=dtype):
                values = scores.to(dtype)
                expected = scores.double() if dtype == torch.float64 else scores
                _, probs = self.sample_once(values)
                _, wider_k = self.sample_once(values, top_k=20)
                torch.testing.assert_close(probs, expected.softmax(dim=-1))
                torch.testing.assert_close(wider_k, probs)

    def test_temperature_changes_concentration(self):
        scores = torch.tensor([[0.0, 2.0, 1.0, -1.0, -2.0, -3.0, -4.0]])
        _, cold = self.sample_once(scores, temperature=0.5)
        _, hot = self.sample_once(scores, temperature=2.0)

        torch.testing.assert_close(cold, (scores / 0.5).softmax(dim=-1))
        torch.testing.assert_close(hot, (scores / 2.0).softmax(dim=-1))
        self.assertGreater(cold[0, 1].item(), hot[0, 1].item())
        self.assertEqual(cold.argmax(dim=-1).item(), hot.argmax(dim=-1).item())

    def test_extreme_temperatures_keep_the_distribution_finite(self):
        scores = torch.tensor([[3.0, 3.0, 1.0, -1.0, -2.0, -3.0, -4.0]])
        _, cold = self.sample_once(scores, temperature=1e-300)
        _, hot = self.sample_once(scores, temperature=1e300)

        torch.testing.assert_close(cold, torch.tensor([[0.5, 0.5, 0, 0, 0, 0, 0]]))
        torch.testing.assert_close(hot, torch.full_like(hot, 1 / 7))

    def test_top_k_keeps_exactly_k_candidates_even_if_cutoff_scores_tie(self):
        scores = torch.tensor([[6.0, 5, 5, 1, 0, -1, -2], [0.0, 1, 2, 3, 5, 5, 6]])
        ids, probs = self.sample_once(scores, top_k=2)

        self.assertEqual(probs.count_nonzero(dim=-1).tolist(), [2, 2])
        self.assertGreater(probs[0, 0].item(), 0)
        self.assertGreater(probs[1, 6].item(), 0)
        self.assertTrue((probs[0, 3:] == 0).all())
        self.assertTrue((probs[1, :4] == 0).all())
        self.assertTrue((probs.gather(-1, ids[:, None]) > 0).all())

    def test_top_p_retains_the_token_that_reaches_the_threshold(self):
        scores = torch.tensor([[0.10, 0.40, 0.05, 0.25, 0.05, 0.10, 0.05]]).log()
        _, probs = self.sample_once(scores, top_p=0.60)
        expected = torch.tensor([[0, 0.40 / 0.65, 0, 0.25 / 0.65, 0, 0, 0]])
        torch.testing.assert_close(probs, expected)

        # 两个并列第一的 token 各占 0.5。保留其中一个就恰好达标。
        tied = torch.tensor([[0.0, 0.0, -torch.inf, -torch.inf, -torch.inf, -torch.inf, -torch.inf]])
        for threshold in (0.5, 0.01):
            with self.subTest(threshold=threshold):
                ids, probs = self.sample_once(tied, top_p=threshold)
                self.assertEqual(probs.count_nonzero().item(), 1)
                self.assertIn(ids.item(), (0, 1))

    def test_combines_temperature_then_top_k_then_top_p(self):
        scores = torch.tensor([[0.10, 0.15, 0, 0.25, 0, 0.50, 0]]).log()
        ids, probs = self.sample_once(scores, temperature=0.5, top_k=2, top_p=0.75)

        # 降温后对概率平方。在 top-k 的两项中，最高项重新归一化为 0.8。
        self.assertEqual(ids.item(), 5)
        torch.testing.assert_close(probs, torch.tensor([[0, 0, 0, 0, 0, 1.0, 0]]))

    def test_fixed_seed_reproduces_sampling_and_draws_both_candidates(self):
        ids = torch.zeros((24, 1), dtype=torch.long)
        scores = torch.tensor([0.0, 3, 3, 0, 0, 0, 0])
        with (
            torch.random.fork_rng(devices=[]),
            patch.object(self.model, "forward", side_effect=lambda x: scores.expand(*x.shape, -1)),
        ):
            torch.manual_seed(7)
            first = self.model.generate(ids, max_new_tokens=3, do_sample=True, top_k=2)
            torch.manual_seed(7)
            second = self.model.generate(ids, max_new_tokens=3, do_sample=True, top_k=2)

        torch.testing.assert_close(first, second)
        self.assertEqual(set(first[:, 1:].flatten().tolist()), {1, 2})

    def test_default_greedy_does_not_sample(self):
        ids = torch.tensor([[5]])
        scores = torch.tensor([[[0.0, 1, 2, 6, 4, 3, 5]]])
        with (
            patch.object(self.model, "forward", return_value=scores),
            patch("torch.multinomial") as sample,
        ):
            result = self.model.generate(ids, 1, temperature=0.5, top_k=2, top_p=0.60)
        sample.assert_not_called()
        self.assertEqual(result.tolist(), [[5, 3]])

    def test_rejects_invalid_sampling_parameters(self):
        ids = torch.tensor([[1]])
        invalid_options = {
            "temperature": (0, -0.5, float("inf"), float("nan")),
            "top_k": (0, -1, 2.5, True),
            "top_p": (0, -0.1, 1.1, float("nan")),
        }
        for name, values in invalid_options.items():
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                    self.model.generate(ids, 1, do_sample=True, **{name: value})
