"""字符编码的行为检查；实现骨架后从仓库根目录运行。"""

import unittest

from warmup.bigram import CharTokenizer


class CharTokenizerTests(unittest.TestCase):
    def test_sorted_unique_vocabulary(self):
        tokenizer = CharTokenizer("baba ")
        self.assertEqual(tokenizer.vocab_size, 3)
        self.assertEqual(tokenizer.stoi, {" ": 0, "a": 1, "b": 2})
        self.assertEqual(tokenizer.itos, {0: " ", 1: "a", 2: "b"})

    def test_encoding_preserves_order_and_repetitions(self):
        tokenizer = CharTokenizer("baba ")
        self.assertEqual(tokenizer.encode("abba "), [1, 2, 2, 1, 0])
        self.assertEqual(tokenizer.decode([2, 1, 2, 1, 0]), "baba ")

    def test_unicode_round_trip(self):
        text = "你好，GPT!\n你好🙂"
        tokenizer = CharTokenizer(text)
        ids = tokenizer.encode(text)
        self.assertEqual(len(ids), len(text))
        self.assertTrue(all(isinstance(token_id, int) for token_id in ids))
        self.assertEqual(tokenizer.decode(ids), text)

    def test_vocabulary_is_independent_of_corpus_order(self):
        first = CharTokenizer("caba")
        second = CharTokenizer("abac")
        self.assertEqual(first.stoi, second.stoi)
        self.assertEqual(first.itos, second.itos)

    def test_empty_text(self):
        tokenizer = CharTokenizer("abc")
        self.assertEqual(tokenizer.encode(""), [])
        self.assertEqual(tokenizer.decode([]), "")

    def test_empty_corpus(self):
        tokenizer = CharTokenizer("")
        self.assertEqual(tokenizer.vocab_size, 0)
        self.assertEqual(tokenizer.stoi, {})
        self.assertEqual(tokenizer.itos, {})
        self.assertEqual(tokenizer.encode(""), [])
        self.assertEqual(tokenizer.decode([]), "")
        with self.assertRaises(KeyError):
            tokenizer.encode("a")
        with self.assertRaises(KeyError):
            tokenizer.decode([0])

    def test_unknown_characters_and_ids_raise(self):
        tokenizer = CharTokenizer("abc")
        with self.assertRaises(KeyError):
            tokenizer.encode("d")
        with self.assertRaises(KeyError):
            tokenizer.decode([tokenizer.vocab_size])
        with self.assertRaises(KeyError):
            tokenizer.decode([-1])
