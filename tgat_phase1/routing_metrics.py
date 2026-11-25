"""Aggregate likelihood when a language label chooses a fixed tokenizer route."""

from collections import defaultdict
import math


INPUT_CELLS = ("G1-BPE", "G4-BPE", "G4-G3")
HYBRID = "G4-hybrid"
CELLS = (*INPUT_CELLS, HYBRID)
POLICY = {"eng_Latn": "G4-BPE", "jpn_Jpan": "G4-BPE", "tur_Latn": "G4-G3"}
_LANGUAGES = tuple(POLICY)
_REFERENCES = {"g1_bpe": "G1-BPE", "g4_g3": "G4-G3", "g4_bpe": "G4-BPE"}


def _positive_int(value, field, key):
    if isinstance(value, str) and value.isdecimal():
        value = int(value)
    if type(value) is not int or value < 1:
        raise ValueError(f"Invalid {field} count for {key}")
    return value


def _index(rows, cell):
    index = {}
    for row in rows:
        split, lang, doc_id = (row[field] for field in ("split", "lang", "id"))
        if (not isinstance(split, str) or not split or lang not in _LANGUAGES
                or type(doc_id) not in (str, int) or doc_id == ""):
            raise ValueError(f"Invalid routing document key in {cell}: {(split, lang, doc_id)}")
        key = split, lang, str(doc_id)
        if key in index:
            raise ValueError(f"Duplicate routing document in {cell}: {key}")
        size = _positive_int(row["bytes"], "byte", key)
        tokens = _positive_int(row["tokens"], "token", key)
        logp = float(row["logp"])
        if not math.isfinite(logp) or logp > 0:
            raise ValueError(f"Invalid log probability in {cell}: {key}")
        index[key] = {"id": doc_id, "split": split, "lang": lang, "bytes": size, "tokens": tokens, "logp": logp}
    return index


def _paired_cells(cells):
    if set(cells) != set(INPUT_CELLS):
        raise ValueError(f"Expected exactly the three scored routing cells: {INPUT_CELLS}")
    indexed = {cell: _index(cells[cell], cell) for cell in INPUT_CELLS}
    anchor = indexed[INPUT_CELLS[0]]
    if not anchor:
        raise ValueError("Routing requires at least one scored document")
    for cell in INPUT_CELLS[1:]:
        if indexed[cell].keys() != anchor.keys():
            raise ValueError(f"Different routing document set in {cell}")
        for key, row in anchor.items():
            if row["bytes"] != indexed[cell][key]["bytes"]:
                raise ValueError(f"Different byte count for routing document in {cell}: {key}")
            if cell == "G4-BPE" and row["tokens"] != indexed[cell][key]["tokens"]:
                raise ValueError(f"BPE token counts differ on the same document: {key}")
    hybrid = {}
    for key in anchor:
        source = POLICY[key[1]]
        hybrid[key] = {**indexed[source][key], "source_cell": source}
    indexed[HYBRID] = hybrid
    return indexed


def hybrid_rows(cells):
    """Return per-document hybrid scores and their scored source cell after pairing checks."""
    return list(_paired_cells(cells)[HYBRID].values())


def _summarize(indexed, keys, split, lang):
    rows = {}
    for cell in CELLS:
        docs = [indexed[cell][key] for key in keys]
        size = sum(row["bytes"] for row in docs)
        tokens = sum(row["tokens"] for row in docs)
        logp = math.fsum(row["logp"] for row in docs)
        rows[cell] = {"split": split, "lang": lang, "cell": cell, "documents": len(docs),
                      "bytes": size, "tokens": tokens, "logp": logp,
                      "bpb": -logp / (math.log(2) * size), "tokens_per_byte": tokens / size}
    for row in rows.values():
        for label, cell in _REFERENCES.items():
            for metric in ("bpb", "tokens_per_byte"):
                row[f"{metric}_vs_{label}"] = row[metric] - rows[cell][metric]
    return rows.values()


def routing_summary(cells):
    """Report byte-weighted scores and paired differences on identical documents.

    The G4 hybrid reuses G4-BPE scores for English and Japanese and G4-G3
    scores for Turkish. Each delta subtracts the named reference cell, so
    a negative BPB delta means a lower cost for the same bytes.
    """
    indexed = _paired_cells(cells)
    splits = defaultdict(lambda: defaultdict(list))
    for key in indexed[INPUT_CELLS[0]]:
        split, lang, _ = key
        splits[split][lang].append(key)
        splits[split]["all"].append(key)
    rows = []
    for split, by_language in sorted(splits.items()):
        for lang in (*_LANGUAGES, "all"):
            if lang in by_language:
                rows.extend(_summarize(indexed, by_language[lang], split, lang))
    return rows
