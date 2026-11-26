"""用小词表验收 GPT-2 的输入嵌入。"""

import unittest

import torch

from gpt2.model import GPT2Embeddings


class GPT2EmbeddingsTests(unittest.TestCase):
    def setUp(self):
        self.model = GPT2Embeddings(vocab_size=4, max_seq_len=3, embedding_dim=3)

    def test_embedding_table_and_output_shapes(self):
        self.assertEqual(self.model.wte.weight.shape, (4, 3))
        self.assertEqual(self.model.wpe.weight.shape, (3, 3))

        input_ids = torch.tensor([[0, 1, 2], [3, 1, 0]])
        self.assertEqual(self.model(input_ids).shape, (2, 3, 3))

    def test_adds_token_and_position_vectors_for_each_sequence(self):
        with torch.no_grad():
            self.model.wte.weight.copy_(
                torch.tensor(
                    [[5.0, 0.0, 1.0], [2.0, 1.0, 4.0], [0.0, 3.0, 2.0], [0.0, 0.0, 0.0]]
                )
            )
            self.model.wpe.weight.copy_(
                torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 1.0], [4.0, 2.0, 0.0]])
            )

        result = self.model(torch.tensor([[1, 1], [2, 0]]))
        expected = torch.tensor(
            [[[3.0, 1.0, 6.0], [2.0, 4.0, 5.0]], [[1.0, 3.0, 4.0], [5.0, 3.0, 2.0]]]
        )
        torch.testing.assert_close(result, expected)

    def test_backward_accumulates_gradients_in_both_tables(self):
        input_ids = torch.tensor([[1, 1], [2, 1]])
        self.model(input_ids).sum().backward()

        expected_token_grad = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 3.0, 3.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]
        )
        expected_position_grad = torch.tensor(
            [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [0.0, 0.0, 0.0]]
        )
        torch.testing.assert_close(self.model.wte.weight.grad, expected_token_grad)
        torch.testing.assert_close(self.model.wpe.weight.grad, expected_position_grad)

    def test_rejects_invalid_input_shape_or_sequence_length(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            self.model(torch.tensor([0, 1]))
        with self.assertRaisesRegex(ValueError, "max_seq_len"):
            self.model(torch.tensor([[0, 1, 2, 3]]))
