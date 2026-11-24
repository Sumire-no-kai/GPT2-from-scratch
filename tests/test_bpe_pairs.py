"""阶段 1 第一小节的 BPE 练习验收。"""

import unittest

from gpt2.tokenizer import count_pairs, merge_pair


class CountPairsTests(unittest.TestCase):
    def test_counts_ordered_pairs_and_overlapping_occurrences(self):
        ids = [1, 2, 1, 1, 1, 2]
        self.assertEqual(count_pairs(ids), {(1, 2): 2, (2, 1): 1, (1, 1): 2})

    def test_no_pairs_in_empty_or_single_id_input(self):
        for ids in ([], [1]):
            with self.subTest(ids=ids):
                self.assertEqual(count_pairs(ids), {})


class MergePairTests(unittest.TestCase):
    def test_replaces_all_matches_and_preserves_other_ids(self):
        ids = [2, 1, 2, 3, 1, 2, 1]
        self.assertEqual(merge_pair(ids, (1, 2), 4), [2, 4, 3, 4, 1])
        self.assertEqual(ids, [2, 1, 2, 3, 1, 2, 1])

    def test_overlapping_matches_are_consumed_from_left_to_right(self):
        self.assertEqual(merge_pair([1, 1, 1, 1, 1], (1, 1), 2), [2, 2, 1])

    def test_returns_a_new_list_when_there_is_no_match(self):
        for ids in ([], [1], [2, 1]):
            with self.subTest(ids=ids):
                result = merge_pair(ids, (1, 2), 3)
                self.assertEqual(result, ids)
                self.assertIsNot(result, ids)
