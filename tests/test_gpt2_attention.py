"""验收 GPT-2 的多头因果自注意力。"""

import math
import unittest

import torch

from gpt2.model import CausalSelfAttention


class CausalSelfAttentionTests(unittest.TestCase):
    def test_projection_and_output_shapes(self):
        attention = CausalSelfAttention(embedding_dim=8, num_heads=2)

        self.assertEqual(attention.c_attn.weight.shape, (24, 8))
        self.assertEqual(attention.c_proj.weight.shape, (8, 8))
        self.assertEqual(attention(torch.randn(2, 4, 8)).shape, (2, 4, 8))

    def test_changing_future_positions_leaves_prefix_outputs_unchanged(self):
        attention = CausalSelfAttention(embedding_dim=8, num_heads=2)
        inputs = torch.randn(2, 4, 8)
        changed = inputs.clone()
        changed[:, 2:] += 3.0

        torch.testing.assert_close(attention(inputs)[:, :2], attention(changed)[:, :2])

    def test_multiplies_values_by_scaled_attention_weights_in_each_head(self):
        attention = CausalSelfAttention(embedding_dim=4, num_heads=2)
        query_projection = torch.eye(4)
        key_projection = torch.eye(4)
        key_projection[1, :2] = torch.tensor([0.5, 0.5])
        value_projection = torch.diag(torch.tensor([1.0, 3.0, 2.0, 1.0]))
        with torch.no_grad():
            attention.c_attn.weight.copy_(
                torch.cat([query_projection, key_projection, value_projection], dim=0)
            )
            attention.c_attn.bias.zero_()
            attention.c_proj.weight.copy_(2.0 * torch.eye(4))
            attention.c_proj.bias.zero_()

        inputs = torch.tensor([[[1.0, 0.0, 0.0, 1.0], [0.0, 2.0, 2.0, 1.0]]])
        # 第二个位置的两个头，当前分数减前一个位置的分数分别是 1 和 4。
        first_weight = 1 / (1 + math.exp(-1 / math.sqrt(2)))
        second_weight = 1 / (1 + math.exp(-4 / math.sqrt(2)))
        expected = torch.tensor(
            [
                [2.0, 0.0, 0.0, 2.0],
                [2 * (1 - first_weight), 12 * first_weight, 8 * second_weight, 2.0],
            ]
        ).unsqueeze(0)
        torch.testing.assert_close(attention(inputs), expected)

    def test_backward_reaches_input_and_projection_weights(self):
        attention = CausalSelfAttention(embedding_dim=8, num_heads=2)
        inputs = torch.randn(2, 4, 8, requires_grad=True)
        attention(inputs).square().mean().backward()

        for tensor in (inputs, attention.c_attn.weight, attention.c_proj.weight):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(tensor.grad.abs().sum().item(), 0)

    def test_rejects_invalid_head_sizes_and_input_shape(self):
        for embedding_dim, num_heads in ((8, 0), (7, 2), (0, 2)):
            with self.subTest(embedding_dim=embedding_dim, num_heads=num_heads):
                with self.assertRaisesRegex(ValueError, "embedding_dim"):
                    CausalSelfAttention(embedding_dim, num_heads)

        attention = CausalSelfAttention(embedding_dim=8, num_heads=2)
        for inputs in (torch.zeros(2, 8), torch.zeros(2, 4, 7)):
            with self.subTest(shape=inputs.shape):
                with self.assertRaisesRegex(ValueError, "shape"):
                    attention(inputs)
