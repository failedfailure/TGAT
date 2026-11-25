from types import SimpleNamespace
import unittest

from tgat_phase1.g3 import verify_native
from tgat_phase1.g3_reference import sentence_spans, stem_edges
from tgat_phase1.rescoring import BYTE_FROM_SYMBOL, FrozenUnigram


SYMBOLS = {byte: symbol for symbol, byte in BYTE_FROM_SYMBOL.items()}


def fixture(pieces, added=()):
    vocab = {symbol: -10.0 for symbol in SYMBOLS.values()}
    for text, score in pieces.items():
        vocab["".join(SYMBOLS[byte] for byte in text.encode("utf-8"))] = score
    return FrozenUnigram(list(vocab.items()), added)


class RescoringTests(unittest.TestCase):
    def test_zero_weight_retains_tie_order(self):
        model = fixture({"a": -1.0, "b": -1.0, "ab": -2.0})
        original = model.entries[b"ab"][0]
        self.assertEqual(model.encode("ab"), (original,))
        self.assertEqual(model.encode("ab", {1: 0.0}), (original,))
        self.assertEqual(model.boundaries("ab", model.encode("ab", {1: 0.1})), {1})

    def test_unicode_boundary_bonus_and_projection(self):
        model = fixture({"猫": -1.5, "達": -1.5, "猫達": -2.0})
        self.assertEqual(len(model.encode("猫達", {1: 0.5})), 1)
        ranked = model.encode("猫達", {1: 1.5})
        self.assertEqual(len(ranked), 2)
        self.assertEqual(model.boundaries("猫達", ranked), {1})
        byte_model = fixture({})
        self.assertEqual(byte_model.boundaries("猫達", byte_model.encode("猫達")), {1})

    def test_pretokenization_and_literal_added_token(self):
        added = [{"id": 257, "content": "<think>", "single_word": False, "lstrip": False,
                  "rstrip": False, "normalized": False}]
        model = fixture({"a2": 10.0}, added)
        expected = (model.entries[b"a"][0], model.entries[b"2"][0], 257)
        self.assertEqual(model.encode("a2<think>", {3: 99.0}), expected)

    def test_native_verification_rejects_equal_length_different_ids(self):
        class Native:
            def encode(self, text, add_special_tokens):
                return SimpleNamespace(ids=[1, 2])

        rows = [{"lang": "eng_Latn", "text": "ab", "id": "doc"}]
        verify_native({"eng_Latn": Native()}, rows, [(1, 2)])
        with self.assertRaisesRegex(ValueError, "token IDs differ"):
            verify_native({"eng_Latn": Native()}, rows, [(3, 4)])


class ReferenceTests(unittest.TestCase):
    def test_unique_lemma_edges(self):
        self.assertEqual(stem_edges("unhelpful", "help"), [2, 6])
        self.assertEqual(stem_edges("mama", "ma"), [])
        self.assertEqual(stem_edges("people", "person"), [])

    def test_align_multiword_token_and_normal_word(self):
        tokens = [("1-2", "Can't", "_", "_"), ("1", "Ca", "can", "AUX"),
                  ("2", "n't", "not", "PART"), ("3", "players", "player", "NOUN")]
        self.assertEqual(sentence_spans("Can't players", tokens),
                         [{"start": 6, "end": 13, "boundaries": [12]}])
        self.assertEqual(sentence_spans("Can't my players", tokens), [])
