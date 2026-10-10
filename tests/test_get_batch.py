"""训练样本分批的行为检查；从仓库根目录运行。"""

import unittest

import torch

from warmup.bigram import get_batch


def make_data(length):
    # 值和下标不同，避免把起点下标误当成数据也能通过检查。
    return torch.arange(length) * 3 + 1


def starts_of(x):
    return (x[:, 0] - 1) // 3


class GetBatchTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_shapes_and_dtype(self):
        x, y = get_batch(make_data(100), block_size=8, batch_size=4)
        self.assertEqual(x.shape, (4, 8))
        self.assertEqual(y.shape, (4, 8))
        self.assertEqual(x.dtype, torch.long)
        self.assertEqual(y.dtype, torch.long)

    def test_rows_are_contiguous_slices_of_data(self):
        data = make_data(100)
        x, y = get_batch(data, block_size=8, batch_size=16)
        for row_x, row_y, start in zip(x, y, starts_of(x).tolist()):
            self.assertTrue(torch.equal(row_x, data[start : start + 8]))
            self.assertTrue(torch.equal(row_y, data[start + 1 : start + 9]))

    def test_targets_are_inputs_shifted_by_one(self):
        x, y = get_batch(make_data(50), block_size=5, batch_size=8)
        self.assertTrue(torch.equal(y[:, :-1], x[:, 1:]))

    def test_data_exactly_one_block_long(self):
        data = make_data(6)
        x, y = get_batch(data, block_size=5, batch_size=3)
        for row_x, row_y in zip(x, y):
            self.assertTrue(torch.equal(row_x, data[:5]))
            self.assertTrue(torch.equal(row_y, data[1:]))

    def test_every_valid_start_can_be_chosen(self):
        # N = 7、T = 4：合法起点是 0、1、2。
        x, _ = get_batch(make_data(7), block_size=4, batch_size=300)
        self.assertEqual(set(starts_of(x).tolist()), {0, 1, 2})


if __name__ == "__main__":
    unittest.main()
