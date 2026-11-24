"""用本地小词表验证 GPT-2 字节映射、合并和编解码。"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import call, patch

from gpt2.tokenizer import GPT2Tokenizer, bytes_to_unicode


class ByteMappingTests(unittest.TestCase):
    def test_maps_all_bytes_to_unique_symbols(self):
        mapping = bytes_to_unicode()
        self.assertEqual(set(mapping), set(range(256)))
        self.assertEqual(len(set(mapping.values())), 256)
        self.assertTrue(all(len(symbol) == 1 for symbol in mapping.values()))
        expected = {0: "Ā", 10: "Ċ", 32: "Ġ", 33: "!", 160: "ł", 173: "Ń", 255: "ÿ"}
        self.assertEqual({key: mapping[key] for key in expected}, expected)


class GPT2TokenizerTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        folder = Path(directory.name)
        vocabulary = {
            "a": 0,
            "b": 1,
            "Ġ": 2,
            "Ġa": 3,
            "ä": 4,
            "½": 5,
            "ł": 6,
            "Ċ": 7,
            "ab": 8,
            "<|endoftext|>": 50256,
        }
        encoder_path = folder / "encoder.json"
        merges_path = folder / "vocab.bpe"
        encoder_path.write_text(json.dumps(vocabulary), encoding="utf-8")
        merges_path.write_text("#version: 0.2\nĠ a\na b\n", encoding="utf-8")
        self.tokenizer = GPT2Tokenizer(encoder_path, merges_path)

    def test_loads_vocabulary_and_merge_ranks(self):
        self.assertEqual(self.tokenizer.encoder["Ġa"], 3)
        self.assertEqual(self.tokenizer.decoder[3], "Ġa")
        self.assertEqual(self.tokenizer.bpe_ranks, {("Ġ", "a"): 0, ("a", "b"): 1})

    def test_bpe_chooses_lowest_rank_and_recomputes_pairs(self):
        self.tokenizer.bpe_ranks = {("b", "c"): 0, ("a", "b"): 1, ("a", "bc"): 2}
        self.assertEqual(self.tokenizer._bpe("abc"), ["abc"])

    def test_bpe_merges_overlapping_pairs_from_left_to_right(self):
        self.tokenizer.bpe_ranks = {("a", "a"): 0}
        self.assertEqual(self.tokenizer._bpe("aaaaa"), ["aa", "aa", "a"])

    def test_bpe_stops_when_no_rule_matches(self):
        self.assertEqual(self.tokenizer._bpe("xy"), ["x", "y"])
        self.assertEqual(self.tokenizer._bpe("x"), ["x"])
        self.assertEqual(self.tokenizer._bpe(""), [])

    def test_round_trip_with_actual_bpe(self):
        ids = self.tokenizer.encode("ab ab")
        self.assertEqual(ids, [8, 3, 1])
        self.assertEqual(self.tokenizer.decode(ids), "ab ab")

    def test_encode_passes_pretokenized_byte_symbols_to_bpe(self):
        with patch.object(
            self.tokenizer, "_bpe", side_effect=[["Ġa"], ["Ġ", "ä", "½", "ł"], ["Ċ"]]
        ) as merge:
            self.assertEqual(self.tokenizer.encode(" a 你\n"), [3, 2, 4, 5, 6, 7])
            self.assertEqual(merge.call_args_list, [call("Ġa"), call("Ġä½ł"), call("Ċ")])

    def test_encode_accepts_explicit_special_tokens(self):
        text = "a<|endoftext|><|endoftext|>b"
        ids = self.tokenizer.encode(text, allow_endoftext=True)
        self.assertEqual(ids, [0, 50256, 50256, 1])
        self.assertEqual(self.tokenizer.decode(ids), text)

    def test_decode_joins_bytes_before_utf8_decoding(self):
        self.assertEqual(self.tokenizer.decode([4, 5, 6], errors="strict"), "你")
        self.assertEqual(self.tokenizer.decode([4]), "\ufffd")
        with self.assertRaises(UnicodeDecodeError):
            self.tokenizer.decode([4], errors="strict")
