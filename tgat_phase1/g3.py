"""Measure G3 boundary alignment and base-checkpoint BPB on frozen inputs."""

import argparse
from collections import Counter
import csv
import hashlib
from importlib.metadata import version
from pathlib import Path
import statistics
import struct
from tempfile import TemporaryDirectory
import unicodedata

from . import boundaries, rescoring
from .boundaries import boundary_hints, training_stems
from .data import LANGUAGES, MODEL, MODEL_REVISION, load_split, read_json, sha256_file, write_json
from .rescoring import load_segmenters
from .tokenization import langmap_paths


WEIGHTS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
SPLITS = ("validation", "test")
GROUPS = ("G2-langmap", "G3-boundary")


def fingerprints(run, reference):
    return {
        "corpus": sha256_file(run / "corpus/manifest.json"),
        "langmap": sha256_file(run / "langmap/manifest.json"),
        "g3_eval": sha256_file(run / "g3-eval/manifest.json"),
        "reference": sha256_file(reference / "manifest.json"),
        "scoring_code": sha256_file(Path(rescoring.__file__)),
        "hints_code": sha256_file(Path(boundaries.__file__)),
    }


def verify_pilot_counts(run, train, segmenters):
    """Compare the portable zero-weight decoder with recorded native counts."""
    holdout = load_split(run / "corpus", "heldout.jsonl")
    saved = run / "seed-42/G2-langmap"
    reference = read_json(run / "seed-42/run.json")
    if (reference["epochs"] != 1 or reference["model_revision"] != MODEL_REVISION
            or reference["corpus_manifest_sha256"] != sha256_file(run / "corpus/manifest.json")
            or reference["langmap_manifest_sha256"] != sha256_file(run / "langmap/manifest.json")):
        raise ValueError("The G3 run needs the matching one-epoch pilot baseline")
    counts = Counter()
    for row in holdout:
        counts[row["lang"]] += len(segmenters[row["lang"]].encode(row["text"]))
    expected = {row["lang"]: row["tokens"] for row in read_json(saved / "heldout.json")["before_lora"]}
    counts["all"] = sum(counts.values())
    trained = sum(len(segmenters[row["lang"]].encode(row["text"])) for row in train)
    if counts != expected or trained != read_json(saved / "training.json")["train_tokens"]:
        raise ValueError("The portable zero-weight decoder differs from the pilot's recorded G2 counts")
    return {"training_tokens": trained, "holdout_tokens": dict(counts)}


def prepare(rows, segmenters, stems):
    return [(row, boundary_hints(row["text"], row["lang"], stems),
             segmenters[row["lang"]].encode(row["text"])) for row in rows]


def rescored_ids(row, hints, weight, segmenters):
    return segmenters[row["lang"]].encode(row["text"], {offset: weight * confidence
                                                       for offset, confidence in hints.items()})


def reference_rows(directory, split):
    rows = load_split(directory, f"{split}.jsonl")
    for row in rows:
        previous = 0
        for span in row["spans"]:
            start, end, edges = span["start"], span["end"], span["boundaries"]
            if not (previous <= start < end <= len(row["text"]) and edges
                    and edges == sorted(set(edges)) and all(start < edge < end for edge in edges)):
                raise ValueError(f"Invalid reference span in {row['id']}")
            previous = end
    return rows


def boundary_metrics(docs, segmenters, split, weight):
    counts = {lang: Counter() for lang in LANGUAGES}
    for row, hints, baseline in docs:
        lang = row["lang"]
        ids = rescored_ids(row, hints, weight, segmenters) if weight else baseline
        edges = segmenters[lang].boundaries(row["text"], ids)
        stats = counts[lang]
        stats["documents"] += 1
        for span in row["spans"]:
            gold = set(span["boundaries"])
            predicted = {edge for edge in edges if span["start"] < edge < span["end"]}
            stats.update(spans=1, gold=len(gold), predicted=len(predicted), matches=len(gold & predicted))
    if any(not counts[lang]["gold"] for lang in LANGUAGES):
        raise ValueError(f"{split} needs reference boundaries for all three languages")
    return [
        {"split": split, "lang": lang, "weight": weight, **stats,
         "precision": stats["matches"] / stats["predicted"] if stats["predicted"] else 0.0,
         "recall": stats["matches"] / stats["gold"],
         "f1": 2 * stats["matches"] / (stats["predicted"] + stats["gold"])}
        for lang, stats in counts.items()
    ]


def fineweb_paths(docs, segmenters, weight):
    g2 = [baseline for _, _, baseline in docs]
    g3 = [rescored_ids(row, hints, weight, segmenters) for row, hints, _ in docs]
    return dict(zip(GROUPS, (g2, g3), strict=True))


def path_digest(docs, paths):
    digest = hashlib.sha256()
    for (row, _, _), ids in zip(docs, paths, strict=True):
        digest.update(bytes.fromhex(row["id"]))
        digest.update(struct.pack(f"<{len(ids) + 1}I", len(ids), *ids))
    return digest.hexdigest()


def length_metrics(docs, paths, split):
    rows = []
    for group, ids_by_doc in paths.items():
        stats = {lang: Counter() for lang in LANGUAGES}
        for (row, hints, baseline), ids in zip(docs, ids_by_doc, strict=True):
            stats[row["lang"]].update(
                documents=1, bytes=len(row["text"].encode("utf-8")), tokens=len(ids),
                documents_with_hints=int(bool(hints)), changed_documents=int(ids != baseline),
            )
        aggregate = Counter()
        for values in stats.values():
            aggregate.update(values)
        stats["all"] = aggregate
        rows.extend({"split": split, "group": group, "lang": lang, **values,
                     "tokens_per_byte": values["tokens"] / values["bytes"]}
                    for lang, values in stats.items())
    return rows


def publish(directory, manifest, tables):
    directory.parent.mkdir(parents=True, exist_ok=True)
    if directory.exists():
        raise FileExistsError(f"{directory} already exists")
    with TemporaryDirectory(prefix=".g3-", dir=directory.parent) as temporary:
        staged = Path(temporary) / "result"
        staged.mkdir()
        for filename, rows in tables.items():
            with (staged / filename).open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        write_json(staged / "run.json", manifest)
        staged.rename(directory)


def screen(run, reference, out):
    if out.exists():
        raise FileExistsError(f"{out} already exists")
    frozen = run / "g3-eval"
    checksums = fingerprints(run, reference)
    if read_json(frozen / "manifest.json")["excluded_corpus_manifest_sha256"] != checksums["corpus"]:
        raise ValueError("G3 evaluation texts must match the frozen pilot corpus")
    train = load_split(run / "corpus", "train.jsonl")
    segmenters = load_segmenters(run)
    counts = verify_pilot_counts(run, train, segmenters)
    stems = training_stems(train)
    print(f"Pilot G2 counts verified: {counts}", flush=True)

    validation = prepare(reference_rows(reference, "validation"), segmenters, stems)
    boundary_rows = []
    scores = {}
    for weight in WEIGHTS:
        rows = boundary_metrics(validation, segmenters, "validation", weight)
        scores[weight] = statistics.mean(row["f1"] for row in rows)
        boundary_rows.extend(rows)
        print(f"Validation lambda={weight:g}: macro-F1={scores[weight]:.4f}", flush=True)
    chosen = min(WEIGHTS, key=lambda weight: (-scores[weight], weight))

    test = prepare(reference_rows(reference, "test"), segmenters, stems)
    if {row["id"] for row, _, _ in validation} & {row["id"] for row, _, _ in test}:
        raise ValueError("The stem-boundary validation and test texts overlap")
    for weight in sorted({0.0, chosen}):
        boundary_rows.extend(boundary_metrics(test, segmenters, "test", weight))

    lengths = []
    hashes = {}
    for split in SPLITS:
        docs = prepare(load_split(frozen, f"{split}.jsonl"), segmenters, stems)
        if fineweb_paths(docs, segmenters, 0.0)[GROUPS[1]] != [ids for _, _, ids in docs]:
            raise ValueError(f"Zero-weight path changed on FineWeb {split}")
        paths = fineweb_paths(docs, segmenters, chosen)
        hashes[split] = {group: path_digest(docs, ids) for group, ids in paths.items()}
        lengths.extend(length_metrics(docs, paths, split))

    publish(out, {
        "selected_lambda": chosen, "length_weight": 0.0, "lambda_grid": WEIGHTS,
        "selection": "maximum validation macro-F1, ties resolved by smaller lambda",
        "validation_macro_f1": scores, "pilot_zero_weight_counts": counts,
        "token_path_sha256": hashes, "inputs": checksums,
        "unicode_version": unicodedata.unidata_version,
    }, {"segmentation_ablation.csv": boundary_rows, "fineweb_tokens.csv": lengths})
    print(f"Selected lambda={chosen:g}; results: {out}")


def verify_native(native_tokenizers, rows, paths):
    """Compare exact IDs with the pinned Tokenizers runtime before GPU scoring."""
    for row, expected in zip(rows, paths, strict=True):
        encoding = native_tokenizers[row["lang"]].encode(row["text"], add_special_tokens=False)
        if tuple(encoding.ids) != expected:
            raise ValueError(f"Pinned G2 token IDs differ for {row['lang']} document {row['id']}")


def load_selection(run, reference, out):
    selection = read_json(out / "run.json")
    if selection["inputs"] != fingerprints(run, reference):
        raise ValueError("G3 inputs or tokenizer code changed since validation selection")
    if selection["selected_lambda"] not in WEIGHTS or selection["length_weight"] != 0.0:
        raise ValueError("Invalid G3 selected weights")
    return selection


def native_langmap(run):
    from tokenizers import Tokenizer

    return {lang: Tokenizer.from_file(str(path)) for lang, path in langmap_paths(run / "langmap").items()}


def verified_eval_paths(run, selection, segmenters, stems, native):
    eval_splits = {}
    for split in SPLITS:
        rows = load_split(run / "g3-eval", f"{split}.jsonl")
        docs = prepare(rows, segmenters, stems)
        paths = fineweb_paths(docs, segmenters, selection["selected_lambda"])
        verify_native(native, rows, paths[GROUPS[0]])
        if {group: path_digest(docs, ids) for group, ids in paths.items()} != selection["token_path_sha256"][split]:
            raise ValueError(f"{split} token IDs differ from the recorded G3 paths")
        eval_splits[split] = (rows, paths)
    return eval_splits


def score_checkpoint(run, reference, out, args):
    from .train import evaluate, hardware, load_model

    result = out / "base-bpb"
    if result.exists():
        raise FileExistsError(f"{result} already exists")
    selection = load_selection(run, reference, out)
    device, dtype = hardware(args)
    train = load_split(run / "corpus", "train.jsonl")
    stems = training_stems(train)
    segmenters = load_segmenters(run)
    native = native_langmap(run)
    pilot = read_json(run / "seed-42/run.json")
    for split in SPLITS:
        annotated = reference_rows(reference, split)
        verify_native(native, annotated, (segmenters[row["lang"]].encode(row["text"]) for row in annotated))
    eval_splits = verified_eval_paths(run, selection, segmenters, stems, native)
    model = load_model(device, dtype)
    eos, context = pilot["eos_id"], pilot["context"]
    if model.config.eos_token_id != eos or any(len(seg.pieces) > model.config.vocab_size
                                               for seg in segmenters.values()):
        raise ValueError("G3 tokenizer IDs differ from the base model configuration")
    results = []
    for split, (rows, paths) in eval_splits.items():
        metrics = {}
        for group, token_ids in paths.items():
            if group == GROUPS[0] or token_ids != paths[GROUPS[0]]:
                print(f"Scoring {split} {group}", flush=True)
                encoded = [(row["lang"], ids) for row, ids in zip(rows, token_ids, strict=True)]
                metrics[group] = evaluate(model, rows, encoded, eos, context, device)
            else:
                metrics[group] = metrics[GROUPS[0]]
            results.extend({"split": split, "group": group, **row} for row in metrics[group])
    publish(result, {"model": MODEL, "model_revision": MODEL_REVISION, "context": context, "eos_id": eos,
                     "device": str(device), "dtype": str(dtype),
                     "packages": {name: version(name) for name in ("torch", "transformers", "tokenizers")},
                     "selected_lambda": selection["selected_lambda"], "native_g2_ids_verified": True,
                     "selection_manifest_sha256": sha256_file(out / "run.json")}, {"bpb.csv": results})
    print(f"Results: {result / 'bpb.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("screen", "bpb"))
    parser.add_argument("--run", type=Path, default=Path("runs/pilot"))
    parser.add_argument("--reference", type=Path, help="override the frozen stem-boundary reference")
    parser.add_argument("--out", type=Path, help="override the boundary pilot result directory")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    frozen = args.run / "g3-eval"
    reference = args.reference or frozen / "stem-reference"
    out = args.out or frozen / "boundary-pilot"
    if args.command == "screen":
        screen(args.run, reference, out)
    else:
        score_checkpoint(args.run, reference, out, args)


if __name__ == "__main__":
    main()
