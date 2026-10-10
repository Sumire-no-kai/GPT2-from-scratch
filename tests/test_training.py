"""训练一步与评估 loss 的行为检查；从仓库根目录运行。"""

import copy
import unittest

import torch
from torch.nn import functional as F

from warmup.bigram import BigramLanguageModel, estimate_loss, get_batch, train_step

VOCAB_SIZE = 6


def table_of(model):
    (table,) = model.parameters()
    return table


class TrainStepTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = BigramLanguageModel(VOCAB_SIZE)
        self.x = torch.randint(VOCAB_SIZE, (4, 5))
        self.y = torch.randint(VOCAB_SIZE, (4, 5))

    def test_returns_loss_before_update_as_float(self):
        _, expected = self.model(self.x, self.y)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        loss = train_step(self.model, optimizer, self.x, self.y)
        self.assertIsInstance(loss, float)
        self.assertAlmostEqual(loss, expected.item(), places=5)

    def test_matches_manual_gradient_descent_over_two_steps(self):
        # 用 SGD 时，每一步应该是：新值 = 旧值 − 学习率 × 这一步自己的梯度。
        # 漏掉清空梯度，第二步就会混进第一步的梯度。
        lr = 0.5
        reference = copy.deepcopy(self.model)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        batches = [(self.x, self.y), (self.y, self.x)]
        for x, y in batches:
            train_step(self.model, optimizer, x, y)
        for x, y in batches:
            table = table_of(reference)
            table.grad = None
            _, loss = reference(x, y)
            loss.backward()
            with torch.no_grad():
                table -= lr * table.grad
        self.assertTrue(torch.allclose(table_of(self.model), table_of(reference)))

    def test_learns_a_repeating_sequence(self):
        data = torch.tensor([0, 1, 2] * 50)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.1)
        losses = [
            train_step(self.model, optimizer, *get_batch(data, block_size=4, batch_size=8))
            for _ in range(200)
        ]
        self.assertLess(losses[-1], losses[0])
        logits, _ = self.model(torch.tensor([[0, 1, 2]]))
        probs = F.softmax(logits[0], dim=-1)
        for current, expected_next in [(0, 1), (1, 2), (2, 0)]:
            self.assertGreater(probs[current, expected_next].item(), 0.95)


class EstimateLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = BigramLanguageModel(VOCAB_SIZE)
        self.data = torch.randint(VOCAB_SIZE, (100,))

    def test_is_mean_loss_over_eval_iters_batches(self):
        torch.manual_seed(1)
        result = estimate_loss(self.model, self.data, block_size=4, batch_size=3, eval_iters=5)
        torch.manual_seed(1)
        expected = sum(
            self.model(*get_batch(self.data, 4, 3))[1].item() for _ in range(5)
        ) / 5
        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result, expected, places=5)

    def test_does_not_change_parameters_or_record_gradients(self):
        before = table_of(self.model).detach().clone()
        estimate_loss(self.model, self.data, block_size=4, batch_size=3, eval_iters=5)
        self.assertTrue(torch.equal(table_of(self.model), before))
        self.assertIsNone(table_of(self.model).grad)


if __name__ == "__main__":
    unittest.main()
