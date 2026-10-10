"""Bigram 模型前向计算与 loss 的行为检查；从仓库根目录运行。"""

import unittest

import torch
from torch.nn import functional as F

from warmup.bigram import BigramLanguageModel

VOCAB_SIZE = 8


class BigramLanguageModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = BigramLanguageModel(VOCAB_SIZE)
        self.idx = torch.randint(VOCAB_SIZE, (3, 5))
        self.targets = torch.randint(VOCAB_SIZE, (3, 5))

    def test_only_parameters_are_one_vocab_by_vocab_table(self):
        shapes = [tuple(p.shape) for p in self.model.parameters()]
        self.assertEqual(shapes, [(VOCAB_SIZE, VOCAB_SIZE)])

    def test_logits_shape_without_targets(self):
        logits, loss = self.model(self.idx)
        self.assertEqual(logits.shape, (3, 5, VOCAB_SIZE))
        self.assertIsNone(loss)

    def test_logits_keep_shape_with_targets(self):
        logits, loss = self.model(self.idx, self.targets)
        self.assertEqual(logits.shape, (3, 5, VOCAB_SIZE))
        self.assertEqual(loss.shape, ())

    def test_loss_is_mean_cross_entropy_over_all_positions(self):
        logits, loss = self.model(self.idx, self.targets)
        expected = sum(
            F.cross_entropy(logits[b, t].unsqueeze(0), self.targets[b, t].unsqueeze(0))
            for b in range(3)
            for t in range(5)
        ) / 15
        self.assertTrue(torch.allclose(loss, expected))

    def test_prediction_depends_only_on_current_token(self):
        idx = torch.tensor([[1, 2, 3, 2], [2, 0, 0, 1]])
        logits, _ = self.model(idx)
        # 同一个编号无论出现在哪个位置、哪一行，分数都相同。
        self.assertTrue(torch.equal(logits[0, 1], logits[0, 3]))
        self.assertTrue(torch.equal(logits[0, 1], logits[1, 0]))
        # 改动其他位置，不影响这个位置的分数。
        changed = idx.clone()
        changed[0, 0] = 7
        changed[0, 2] = 6
        changed_logits, _ = self.model(changed)
        self.assertTrue(torch.equal(changed_logits[0, 1], logits[0, 1]))

    def test_loss_backpropagates_to_table(self):
        _, loss = self.model(self.idx, self.targets)
        loss.backward()
        (table,) = self.model.parameters()
        self.assertIsNotNone(table.grad)
        self.assertGreater(table.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
