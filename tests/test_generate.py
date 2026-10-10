"""Bigram 生成的行为检查；从仓库根目录运行。"""

import unittest

import torch

from warmup.bigram import BigramLanguageModel

VOCAB_SIZE = 5


def set_table(model, rows):
    """把模型唯一的 (V, V) 表设成指定的分数。"""
    (table,) = model.parameters()
    with torch.no_grad():
        table.copy_(torch.tensor(rows, dtype=table.dtype))


def cycle_rows():
    # 当前是 i 时，下一个几乎必定是 (i + 1) % V。
    rows = [[-1e4] * VOCAB_SIZE for _ in range(VOCAB_SIZE)]
    for i in range(VOCAB_SIZE):
        rows[i][(i + 1) % VOCAB_SIZE] = 0.0
    return rows


class GenerateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = BigramLanguageModel(VOCAB_SIZE)

    def test_shape_dtype_and_range(self):
        idx = torch.randint(VOCAB_SIZE, (3, 4))
        out = self.model.generate(idx, max_new_tokens=6)
        self.assertEqual(out.shape, (3, 10))
        self.assertEqual(out.dtype, torch.long)
        self.assertTrue(((out >= 0) & (out < VOCAB_SIZE)).all())

    def test_prefix_is_kept_and_input_is_not_modified(self):
        idx = torch.tensor([[1, 2, 3], [4, 0, 1]])
        original = idx.clone()
        out = self.model.generate(idx, max_new_tokens=5)
        self.assertTrue(torch.equal(out[:, :3], original))
        self.assertTrue(torch.equal(idx, original))

    def test_zero_new_tokens_returns_input(self):
        idx = torch.tensor([[1, 2, 3]])
        out = self.model.generate(idx, max_new_tokens=0)
        self.assertTrue(torch.equal(out, idx))

    def test_each_step_uses_the_last_token(self):
        set_table(self.model, cycle_rows())
        # 两段的最后一个编号不同，第一个编号与最后一个也不同。
        idx = torch.tensor([[0, 3], [2, 4]])
        out = self.model.generate(idx, max_new_tokens=4)
        self.assertEqual(out.tolist(), [[0, 3, 4, 0, 1, 2], [2, 4, 0, 1, 2, 3]])

    def test_samples_instead_of_always_taking_the_highest_score(self):
        rows = [[-1e4] * VOCAB_SIZE for _ in range(VOCAB_SIZE)]
        rows[0][1] = 0.0
        rows[0][2] = 0.0  # 当前是 0 时，1 和 2 各占一半概率。
        set_table(self.model, rows)
        idx = torch.zeros((200, 1), dtype=torch.long)
        out = self.model.generate(idx, max_new_tokens=1)
        self.assertEqual(set(out[:, 1].tolist()), {1, 2})


if __name__ == "__main__":
    unittest.main()
