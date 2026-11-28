"""验收完整 GPT-2 模型的形状、因果性和权重共享。"""

import math
import unittest
from unittest.mock import patch

import torch
from torch import nn

from gpt2.model import GPT2


class GPT2Tests(unittest.TestCase):
    def setUp(self):
        self.model = GPT2(
            vocab_size=11, max_seq_len=6, embedding_dim=8, num_heads=2, num_layers=2
        )

    def test_default_configuration_has_124m_parameters(self):
        # 只数参数的形状，无需给 124M 权重分配实际内存。
        with torch.device("meta"):
            model = GPT2()
        self.assertEqual(len(model.blocks), 12)
        self.assertEqual(model.embeddings.wpe.weight.shape, (1024, 768))
        self.assertEqual(model.embeddings.wte.weight.shape, (50_257, 768))
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 124_439_808)

    def test_returns_logits_after_all_blocks_and_final_normalization(self):
        input_ids = torch.tensor([[1, 2, 4], [3, 0, 1]])
        x = self.model.embeddings(input_ids)
        for block in self.model.blocks:
            x = block(x)
        normalized = self.model.ln_f(x)
        expected = normalized @ self.model.embeddings.wte.weight.T

        self.assertIsNone(self.model.lm_head.bias)
        logits = self.model(input_ids)
        self.assertEqual(logits.shape, (2, 3, 11))
        torch.testing.assert_close(logits, expected)

    def test_output_head_and_token_embeddings_share_one_trainable_parameter(self):
        weight = self.model.embeddings.wte.weight
        self.assertIs(self.model.lm_head.weight, weight)
        self.assertEqual(sum(parameter is weight for parameter in self.model.parameters()), 1)

        self.model(torch.tensor([[1, 2, 4]])).square().mean().backward()
        self.assertIsNotNone(weight.grad)
        self.assertTrue(torch.isfinite(weight.grad).all())
        self.assertGreater(weight.grad.abs().sum().item(), 0)

    def test_changing_future_token_ids_does_not_change_prefix_logits(self):
        input_ids = torch.tensor([[1, 2, 3, 4], [6, 5, 0, 1]])
        changed_ids = torch.tensor([[1, 2, 7, 8], [6, 5, 2, 3]])

        original = self.model(input_ids)
        changed = self.model(changed_ids)
        prefix_only = self.model(input_ids[:, :2])
        torch.testing.assert_close(original[:, :2], changed[:, :2])
        torch.testing.assert_close(original[:, :2], prefix_only)

    def test_sdpa_reaches_all_blocks_and_loads_reference_weights(self):
        sdpa = GPT2(11, 6, 8, 2, num_layers=2, use_sdpa=True)
        sdpa.load_state_dict(self.model.state_dict(), strict=True)
        ids = torch.tensor([[1, 3, 2, 8], [4, 7, 2, 0]])
        expected = self.model(ids)
        with patch.object(nn.functional, "scaled_dot_product_attention",
                          wraps=nn.functional.scaled_dot_product_attention) as attention:
            actual = sdpa(ids)

        self.assertEqual(attention.call_count, len(sdpa.blocks))
        for call in attention.call_args_list:
            self.assertTrue(call.kwargs["is_causal"])
            self.assertEqual(call.kwargs["dropout_p"], 0.0)
            self.assertEqual(call.args[0].shape, (2, 2, 4, 4))  # (B, H, T, D)
        self.assertIs(sdpa.lm_head.weight, sdpa.embeddings.wte.weight)
        torch.testing.assert_close(actual, expected)

    def test_scales_all_residual_output_weights_by_model_depth(self):
        with patch.object(nn.init, "normal_", wraps=nn.init.normal_) as initialize:
            model = GPT2(11, 6, 8, 2, num_layers=3)
        standard_deviations = {
            id(call.args[0]): call.kwargs["std"]
            for call in initialize.call_args_list
            if "std" in call.kwargs
        }
        expected = 0.02 / math.sqrt(6)
        for block in model.blocks:
            self.assertEqual(standard_deviations[id(block.attn.c_proj.weight)], expected)
            self.assertEqual(standard_deviations[id(block.mlp.c_proj.weight)], expected)
            self.assertEqual(standard_deviations[id(block.attn.c_attn.weight)], 0.02)
            self.assertEqual(standard_deviations[id(block.mlp.c_fc.weight)], 0.02)

    def test_rejects_a_model_without_transformer_blocks(self):
        with self.assertRaisesRegex(ValueError, "num_layers"):
            GPT2(11, 6, 8, 2, num_layers=0)
