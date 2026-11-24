"""简单 BPE 合并规则训练的验收。"""

import unittest

from gpt2.tokenizer import train_bpe


class TrainBPETests(unittest.TestCase):
    def test_recounts_after_merging_and_returns_rules_in_order(self):
        ids = [1, 2, 1, 2, 1, 2]
        rules = train_bpe(ids, num_merges=2, start_id=3)
        self.assertEqual(list(rules.items()), [((1, 2), 3), ((3, 3), 4)])
        self.assertEqual(ids, [1, 2, 1, 2, 1, 2])

    def test_tie_uses_first_pair_in_current_sequence(self):
        rules = train_bpe([5, 2, 3], num_merges=1, start_id=6)
        self.assertEqual(rules, {(5, 2): 6})

    def test_stops_when_no_pairs_remain(self):
        for ids, expected in (([], {}), ([5], {}), ([5, 5], {(5, 5): 6})):
            with self.subTest(ids=ids):
                self.assertEqual(train_bpe(ids, 10, start_id=6), expected)

    def test_zero_merges_returns_no_rules(self):
        self.assertEqual(train_bpe([1, 2], num_merges=0, start_id=3), {})

    def test_rejects_negative_merge_limit(self):
        with self.assertRaisesRegex(ValueError, "num_merges"):
            train_bpe([1, 2], num_merges=-1, start_id=3)

    def test_rejects_start_id_in_input_range(self):
        with self.assertRaisesRegex(ValueError, "start_id"):
            train_bpe([2, 5], num_merges=1, start_id=5)
