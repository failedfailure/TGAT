"""Generate soft boundary hints from fixed rules and pilot training text."""

from collections import Counter
import re


LATIN_WORD = re.compile(r"(?<!\w)[A-Za-zÇĞİÖŞÜçğıöşü]+(?!\w)")
TURKISH_CASE = str.maketrans("Iİ", "ıi")
MIN_STEM_COUNT = 2

SUFFIX_GROUPS = {
    "eng_Latn": (
        (0.90, "ness less ment ship ful able"),
        (0.78, "ing ed"),
        (0.65, "est ism ist ly"),
        (0.45, "er es s"),
    ),
    "tur_Latn": (
        (0.90, "lardan lerden ların lerin larla lerle larda lerde larımız lerimiz"),
        (0.85, "lar ler ları leri lık lik luk lük sız siz suz süz"),
        (0.72, "dan den tan ten dır dir dur dür tır tir tur tür mış miş muş müş"),
        (0.68, "mak mek yor sın sin sun sün"),
        (0.55, "ın in un ün da de ta te la le"),
    ),
}
SUFFIXES = {
    lang: sorted(((suffix, weight) for weight, suffixes in groups for suffix in suffixes.split()),
                 key=lambda entry: -len(entry[0]))
    for lang, groups in SUFFIX_GROUPS.items()
}
JAPANESE_ENDINGS = {
    "ませんでした": 0.90, "ました": 0.90, "ません": 0.90, "ましょう": 0.85, "ます": 0.85,
    "なかった": 0.75, "ない": 0.75, "かった": 0.65, "たい": 0.65, "られる": 0.60,
}
JAPANESE_INFLECTION = re.compile(
    r"[㐀-鿿々][ぁ-ゖ]{0,3}?(?P<ending>" + "|".join(JAPANESE_ENDINGS) + ")"
)


def fold_word(word, lang):
    return (word.translate(TURKISH_CASE) if lang == "tur_Latn" else word).lower()


def training_stems(documents):
    counts = {lang: Counter() for lang in SUFFIXES}
    for row in documents:
        lang = row["lang"]
        if lang in counts:
            counts[lang].update(fold_word(match.group(), lang) for match in LATIN_WORD.finditer(row["text"]))
    return counts


def boundary_hints(text, lang, stems):
    if lang == "jpn_Jpan":
        return {match.start("ending"): JAPANESE_ENDINGS[match.group("ending")]
                for match in JAPANESE_INFLECTION.finditer(text)}
    hints = {}
    for match in LATIN_WORD.finditer(text):
        word = fold_word(match.group(), lang)
        if len(word) != len(match.group()):
            continue
        for suffix, weight in SUFFIXES[lang]:
            cut = len(word) - len(suffix)
            if cut >= 3 and word.endswith(suffix) and stems[lang][word[:cut]] >= MIN_STEM_COUNT:
                hints[match.start() + cut] = weight
                break
    return hints
