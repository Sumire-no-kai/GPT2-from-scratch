"""验收 GPT-2 的 MLP 和 Pre-LayerNorm Transformer block。"""

import math
import unittest

import torch
from torch import nn

from gpt2.model import MLP, CausalSelfAttention, TransformerBlock


class MLPTests(unittest.TestCase):
    def test_linear_layer_and_output_shapes(self):
        mlp = MLP(embedding_dim=8)

        self.assertEqual(mlp.c_fc.weight.shape, (32, 8))
        self.assertEqual(mlp.c_proj.weight.shape, (8, 32))
        self.assertEqual(mlp(torch.randn(2, 4, 8)).shape, (2, 4, 8))

    def test_applies_tanh_gelu_between_the_two_projections(self):
        mlp = MLP(embedding_dim=2).double()
        inputs = torch.tensor([[[2.0, -1.0], [-1.0, 1.5]]], dtype=torch.float64)
        with torch.no_grad():
            mlp.c_fc.weight.zero_()
            mlp.c_fc.weight[:2].copy_(torch.eye(2, dtype=inputs.dtype))
            mlp.c_fc.bias.zero_()
            mlp.c_fc.bias[1] = 0.25
            mlp.c_proj.weight.zero_()
            mlp.c_proj.weight[:, :2].copy_(inputs.new_tensor([[1.0, 2.0], [-1.0, 0.5]]))
            mlp.c_proj.bias.copy_(inputs.new_tensor([0.125, -0.5]))

        # 只有前两个隐藏特征参与输出，按公式分别算出它们的激活。
        expanded = inputs + inputs.new_tensor([0.0, 0.25])
        inner = math.sqrt(2 / math.pi) * (expanded + 0.044715 * expanded.pow(3))
        first, second = (0.5 * expanded * (1 + torch.tanh(inner))).unbind(dim=-1)
        expected = torch.stack(
            [first + 2 * second + 0.125, -first + 0.5 * second - 0.5], dim=-1
        )
        torch.testing.assert_close(mlp(inputs), expected)


class TransformerBlockTests(unittest.TestCase):
    def test_connects_attention_and_mlp_and_backpropagates(self):
        block = TransformerBlock(embedding_dim=8, num_heads=2)
        inputs = torch.randn(2, 4, 8, requires_grad=True)

        self.assertIsInstance(block.attn, CausalSelfAttention)
        self.assertIsInstance(block.mlp, MLP)
        self.assertIsNot(block.ln_1, block.ln_2)
        output = block(inputs)
        self.assertEqual(output.shape, inputs.shape)
        output.square().mean().backward()

        for tensor in (
            inputs,
            block.ln_1.weight,
            block.attn.c_attn.weight,
            block.ln_2.weight,
            block.mlp.c_fc.weight,
        ):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(tensor.grad.abs().sum().item(), 0)

    def test_normalizes_before_each_branch_and_uses_updated_residual(self):
        block = TransformerBlock(embedding_dim=4, num_heads=2).double()
        block.attn = nn.Identity()
        block.mlp = nn.Identity()
        inputs = torch.tensor(
            [[[1.0, 3.0, 2.0, 0.0], [4.0, -1.0, 1.0, 2.0]]], dtype=torch.float64
        )
        with torch.no_grad():
            block.ln_1.bias.copy_(inputs.new_tensor([0.0, 2.0, -1.0, 0.5]))
            block.ln_2.weight.fill_(0.5)

        # 用恒等分支让两次归一化的结果直接参与残差，便于对照顺序。
        after_attention = inputs + block.ln_1(inputs)
        expected = after_attention + block.ln_2(after_attention)
        torch.testing.assert_close(block(inputs), expected)

    def test_zero_residual_branches_return_original_input(self):
        block = TransformerBlock(embedding_dim=8, num_heads=2)
        with torch.no_grad():
            block.attn.c_proj.weight.zero_()
            block.attn.c_proj.bias.zero_()
            block.mlp.c_proj.weight.zero_()
            block.mlp.c_proj.bias.zero_()

        inputs = torch.randn(2, 4, 8)
        torch.testing.assert_close(block(inputs), inputs, rtol=0, atol=0)

    def test_future_input_changes_leave_prefix_outputs_unchanged(self):
        block = TransformerBlock(embedding_dim=8, num_heads=2)
        inputs = torch.randn(2, 4, 8)
        changed = inputs.clone()
        # 只改变后两个位置的一个特征，避免整体平移被 LayerNorm 抵消。
        changed[:, 2:, 0] += 6.0

        original_output = block(inputs)
        changed_output = block(changed)
        torch.testing.assert_close(original_output[:, :2], changed_output[:, :2])
        self.assertFalse(torch.allclose(original_output[:, 2:], changed_output[:, 2:]))
