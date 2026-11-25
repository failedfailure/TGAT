"""Build an independent stem-boundary reference from UD CoNLL-U files."""

import argparse
import hashlib
from pathlib import Path
import re

from .data import LANGUAGES, canonical_text, write_splits


UD_RELEASE = "r2.16"
TREEBANKS = {
    "eng_Latn": ("UD_English-EWT", "en_ewt", "a994b6990643429070ecc6e236d26ad6a04bfefb",
                 "c59667960ae967636606f47bc35f132a5557ad39"),
    "jpn_Jpan": ("UD_Japanese-GSDLUW", "ja_gsdluw", "12e58a442d5923c8e5e2c8935fdbe94fa3e1a24e",
                 "f663d5401573610b7ee5da1e87e14f194065f0ec"),
    "tur_Latn": ("UD_Turkish-Kenet", "tr_kenet", "929d4a7116a7b9f23be9cf9b7f1bf5d0bdc74a4e",
                 "b9929f6a0f0c9a7f94338fae02f61e8e201289a1"),
}
CONTENT_POS = {"ADJ", "ADV", "AUX", "NOUN", "PROPN", "VERB"}


def stem_edges(form, lemma):
    """Return the visible edges of a unique proper lemma substring."""
    if not lemma.isalpha() or len(lemma) >= len(form):
        return []
    start = form.find(lemma)
    if start < 0 or form.find(lemma, start + 1) >= 0:
        return []
    return [pos for pos in (start, start + len(lemma)) if 0 < pos < len(form)]


def sentence_spans(text, tokens):
    spans = []
    cursor = 0
    multiword_end = 0
    for index, form, lemma, pos, *_ in tokens:
        if "." in index:
            continue
        if "-" in index:
            multiword_end = int(index.split("-")[1])
        elif int(index) <= multiword_end:
            continue
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        form = canonical_text(form)
        end = cursor + len(form)
        if not text.startswith(form, cursor):
            return []
        if "-" not in index and pos in CONTENT_POS:
            offsets = stem_edges(form, canonical_text(lemma))
            if offsets:
                spans.append({"start": cursor, "end": end,
                              "boundaries": [cursor + offset for offset in offsets]})
        cursor = end
    return [] if text[cursor:].strip() else spans


def annotated_sentences(path, lang):
    for block in path.read_text(encoding="utf-8").strip().split("\n\n"):
        lines = block.splitlines()
        comments = dict(line[2:].split(" = ", 1) for line in lines
                        if line.startswith("# ") and " = " in line)
        text = canonical_text(comments.get("text"))
        if not text.strip() or re.search(r"<\|[^|\n]{1,80}\|>", text):
            continue
        spans = sentence_spans(text, [line.split("\t") for line in lines
                                      if line and not line.startswith("#")])
        if spans:
            yield {"id": hashlib.sha256(text.encode("utf-8")).hexdigest(), "lang": lang,
                   "text": text, "sent_id": comments["sent_id"], "spans": spans}


def build(source, out):
    splits = {}
    provenance = {}
    seen = set()
    for role, split, hash_index in (("validation", "dev", 2), ("test", "test", 3)):
        records = []
        provenance[role] = {}
        for lang in LANGUAGES:
            repo, prefix, *_ = TREEBANKS[lang]
            filename = f"{prefix}-ud-{split}.conllu"
            path = source / filename
            raw = path.read_bytes()
            blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
            if blob != TREEBANKS[lang][hash_index]:
                raise ValueError(f"{path}: differs from the pinned {UD_RELEASE} treebank file")
            provenance[role][lang] = {
                "source": f"https://github.com/UniversalDependencies/{repo}/blob/{UD_RELEASE}/{filename}",
                "git_blob_sha1": blob,
                "license": "CC BY-SA 4.0",
            }
            for row in annotated_sentences(path, lang):
                if row["id"] not in seen:
                    records.append(row)
                    seen.add(row["id"])
        splits[f"{role}.jsonl"] = records
    write_splits(out, {"sources": provenance, "ud_release": UD_RELEASE,
                       "reference": "visible edges of unique proper lemma substrings",
                       "deduplication": "SHA-256 of normalized sentence text"}, splits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="directory with the six pinned .conllu files")
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    build(args.source, args.out)


if __name__ == "__main__":
    main()
