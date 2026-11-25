"""Rescore paths through the frozen Qwen ByteLevel Unigram vocabulary."""

from bisect import bisect_left, bisect_right
from collections import defaultdict
from functools import lru_cache
from itertools import accumulate
import math
import re
import unicodedata

from .data import LANGUAGES, canonical_text, read_json
from .tokenization import langmap_paths, verified_langmap_directory


QWEN_SPLIT = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
SHADOW_SPLIT = re.compile(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\nA-Za-z0-9]?[A-Za-z]+|[0-9]"
    r"| ?[^\sA-Za-z0-9]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    re.ASCII,
)
BYTE_LEVEL = {"type": "ByteLevel", "add_prefix_space": False,
              "trim_offsets": False, "use_regex": False}
PRETOKENIZER = {
    "type": "Sequence",
    "pretokenizers": [
        {"type": "Split", "pattern": {"Regex": QWEN_SPLIT}, "behavior": "Isolated", "invert": False},
        BYTE_LEVEL,
    ],
}
VISIBLE_BYTES = set(range(33, 127)) | set(range(161, 173)) | set(range(174, 256))
BYTE_FROM_SYMBOL = {chr(byte): byte for byte in VISIBLE_BYTES}
BYTE_FROM_SYMBOL.update(
    {chr(i): byte for i, byte in enumerate(sorted(set(range(256)) - VISIBLE_BYTES), 256)}
)


@lru_cache(maxsize=4096)
def shadow_character(char):
    """Keep the regex's character classes and preserve code-point offsets."""
    if char.isascii():
        return char
    kind = unicodedata.category(char)[0]
    if kind == "L":
        folded = char.casefold()
        return folded if len(folded) == 1 and folded.isascii() else "A"
    if kind == "N":
        return "0"
    # The optional literal space in the punctuation branch is ASCII-only.
    return "\t" if char.isspace() else "!"


def pretoken_spans(text, start, end):
    part = text[start:end]
    shadow = part if part.isascii() else "".join(map(shadow_character, part))
    for match in SHADOW_SPLIT.finditer(shadow):
        yield start + match.start(), start + match.end()


class FrozenUnigram:
    def __init__(self, vocab, added_tokens=()):
        self.pieces = [bytes(BYTE_FROM_SYMBOL[char] for char in token) for token, _ in vocab]
        self.entries = {piece: (index, score) for index, (piece, (_, score))
                        in enumerate(zip(self.pieces, vocab, strict=True))}
        if (len(self.pieces) != len(self.entries) or b"" in self.entries
                or any(bytes([byte]) not in self.entries for byte in range(256))
                or any(not math.isfinite(score) for _, score in vocab)):
            raise ValueError("Expected a finite, lossless ByteLevel Unigram vocabulary")
        lengths = defaultdict(set)
        for piece in self.entries:
            lengths[piece[0]].add(len(piece))
        self.lengths = {byte: sorted(sizes) for byte, sizes in lengths.items()}
        self.added_ids = {}
        for token in added_tokens:
            value = token["content"]
            if (token["id"] != len(self.pieces) or not value or value in self.added_ids
                    or any(token[field] for field in ("single_word", "lstrip", "rstrip", "normalized"))):
                raise ValueError("Expected Qwen's literal added tokens in ID order")
            self.added_ids[value] = token["id"]
            self.pieces.append(value.encode("utf-8"))
        literals = sorted(self.added_ids, key=len, reverse=True)
        self.added_pattern = re.compile("|".join(map(re.escape, literals))) if literals else None

    @lru_cache(maxsize=32768)
    def best_path(self, raw, bonuses=()):
        """Solve a pretoken lattice in increasing start order, retaining ties."""
        end = len(raw)
        weights = dict(bonuses)
        scores = [-math.inf] * (end + 1)
        scores[0] = 0.0
        back = [None] * (end + 1)
        for start, first_byte in enumerate(raw):
            for size in self.lengths[first_byte]:
                stop = start + size
                if stop > end:
                    break
                entry = self.entries.get(raw[start:stop])
                if entry is None:
                    continue
                index, score = entry
                candidate = scores[start] + score
                if stop in weights:
                    candidate += weights[stop]
                if candidate > scores[stop]:
                    scores[stop] = candidate
                    back[stop] = start, index
        result = []
        while end:
            end, index = back[end]
            result.append(index)
        return tuple(reversed(result))

    def spans(self, text):
        start = 0
        added = self.added_pattern.finditer(text) if self.added_pattern else ()
        for match in added:
            for left, right in pretoken_spans(text, start, match.start()):
                yield left, right, None
            yield match.start(), match.end(), self.added_ids[match.group()]
            start = match.end()
        for left, right in pretoken_spans(text, start, len(text)):
            yield left, right, None

    def encode(self, text, rewards=None):
        """Return token IDs with bonuses at internal Unicode character offsets."""
        if not text or canonical_text(text) != text:
            raise ValueError("G3 requires nonempty, normalized document text")
        rewards = rewards or {}
        positions = sorted(rewards)
        ids = []
        for start, end, special in self.spans(text):
            if special is not None:
                ids.append(special)
                continue
            offsets = positions[bisect_right(positions, start):bisect_left(positions, end)]
            bonuses = tuple((len(text[start:offset].encode("utf-8")), rewards[offset])
                            for offset in offsets if rewards[offset])
            ids.extend(self.best_path(text[start:end].encode("utf-8"), bonuses))
        if b"".join(self.pieces[index] for index in ids) != text.encode("utf-8"):
            raise ValueError("G3 produced a non-reversible path")
        return tuple(ids)

    def boundaries(self, text, ids):
        """Project token ends onto character offsets, excluding UTF-8 fragments."""
        offsets = accumulate(len(char.encode("utf-8")) for char in text)
        characters = {byte: i for i, byte in enumerate(offsets, 1)}
        ends = accumulate(len(self.pieces[index]) for index in ids)
        return {characters[end] for end in ends if end in characters and characters[end] < len(text)}


def load_segmenters(run):
    directory = verified_langmap_directory(run)
    original = read_json(directory / "model_tokenizer/tokenizer.json")
    original_vocab = original["model"]["vocab"]
    expected = {"normalizer": {"type": "NFC"}, "pre_tokenizer": PRETOKENIZER,
                "decoder": BYTE_LEVEL, "post_processor": BYTE_LEVEL,
                "truncation": None, "padding": None}
    segmenters = {}
    for lang, path in langmap_paths(directory).items():
        config = read_json(path)
        model = config["model"]
        vocab = model["vocab"]
        if (model["type"] != "Unigram" or model["byte_fallback"]
                or any(config.get(key) != value or original.get(key) != value
                       for key, value in expected.items())
                or config["added_tokens"] != original["added_tokens"]
                or len(vocab) != len(original_vocab)
                or any(original_vocab.get(piece) != index for index, (piece, _) in enumerate(vocab))):
            raise ValueError(f"{path}: expected the original Qwen pipeline and token IDs")
        segmenters[lang] = FrozenUnigram(vocab, config["added_tokens"])
    return segmenters
