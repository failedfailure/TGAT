"""Compare G3 with zero boundary reward against frozen native G2 token IDs."""

import argparse
from importlib.metadata import version
from itertools import zip_longest
import json
from pathlib import Path
import unicodedata

from tgat_phase1.boundaries import boundary_hints, training_stems
from tgat_phase1.data import LANGUAGES, LANGMAP_REVISION, MODEL_REVISION, load_split, read_json, sha256_file
from tgat_phase1.g3 import load_selection, native_langmap, rescored_ids
from tgat_phase1.rescoring import load_segmenters
from tgat_phase1.tokenization import encode_documents


def compare_split(rows, native, segmenters, stems):
    counts = {lang: {"checked": 0, "identical": 0, "mismatched": 0} for lang in LANGUAGES}
    mismatches = []
    for row, (lang, g2) in zip(rows, encode_documents(rows, native), strict=True):
        hints = boundary_hints(row["text"], lang, stems)
        g3 = rescored_ids(row, hints, 0.0, segmenters)
        stats = counts[lang]
        stats["checked"] += 1
        if tuple(g2) == g3:
            stats["identical"] += 1
            continue

        stats["mismatched"] += 1
        index = next(index for index, (left, right) in enumerate(zip_longest(g2, g3))
                     if left != right)
        start, stop = max(0, index - 2), index + 3
        mismatches.append({
            "document_id": row["id"], "lang": lang,
            "g2_tokens": len(g2), "g3_zero_weight_tokens": len(g3),
            "first_difference_index": index, "excerpt_start_index": start,
            "g2_ids": g2[start:stop], "g3_zero_weight_ids": list(g3[start:stop]),
        })
    counts["all"] = {key: sum(counts[lang][key] for lang in LANGUAGES)
                     for key in ("checked", "identical", "mismatched")}
    return {"by_language": counts, "mismatches": mismatches}


def check(run):
    pilot_manifest = run / "seed-42/run.json"
    corpus_manifest = run / "corpus/manifest.json"
    langmap_manifest = run / "langmap/manifest.json"
    pilot = read_json(pilot_manifest)
    corpus_digest = sha256_file(corpus_manifest)
    langmap_digest = sha256_file(langmap_manifest)
    tokenizers_version = version("tokenizers")
    if (pilot["epochs"] != 1 or pilot["model_revision"] != MODEL_REVISION
            or pilot["corpus_manifest_sha256"] != corpus_digest
            or pilot["langmap_manifest_sha256"] != langmap_digest
            or read_json(langmap_manifest)["unimixlm_revision"] != LANGMAP_REVISION):
        raise ValueError("Parity requires the frozen one-epoch G2 pilot artifacts")
    if tokenizers_version != pilot["packages"]["tokenizers"]:
        raise ValueError(f"Expected pilot Tokenizers {pilot['packages']['tokenizers']}; "
                         f"found {tokenizers_version}")

    g3_eval = run / "g3-eval"
    selection = load_selection(run, g3_eval / "stem-reference", g3_eval / "boundary-pilot")
    train = load_split(run / "corpus", "train.jsonl")
    heldout = load_split(run / "corpus", "heldout.jsonl")
    segmenters = load_segmenters(run)
    native = native_langmap(run)
    stems = training_stems(train)
    return {
        "lambda": 0.0,
        "tokenizers_version": tokenizers_version,
        "unicode_version": unicodedata.unidata_version,
        "inputs": {**selection["inputs"], "g2_pilot": sha256_file(pilot_manifest)},
        "splits": {name: compare_split(rows, native, segmenters, stems)
                   for name, rows in (("train", train), ("heldout", heldout))},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/pilot"))
    result = check(parser.parse_args().run)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(any(split["mismatches"] for split in result["splits"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
