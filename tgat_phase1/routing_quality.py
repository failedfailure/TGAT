"""Evaluate a fixed language-tag routing policy on aligned FLORES-200 text."""

import argparse
from collections import Counter, defaultdict
import csv
import fcntl
import math
from pathlib import Path
from tempfile import TemporaryDirectory

from . import downstream, downstream_data, downstream_models, routing_data, routing_metrics
from .data import LANGUAGES, canonical_text, load_split, read_json, sha256_file, write_json
from .embedding_ablation import SEEDS, verified_runtime

CELLS = {
    "G1-BPE": ("G1", "G1"),
    "G4-BPE": ("G4", "G1"),
    "G4-G3": ("G4", "G4"),
}
BATCH_SIZE = 16


def dataset_audit(run, out, rows):
    """Check exact whole-record matches to the project's earlier datasets."""
    corpora = (
        ("pilot", run / "corpus", ("train.jsonl", "heldout.jsonl")),
        ("g3-fineweb", run / "g3-eval", ("validation.jsonl", "test.jsonl")),
        ("g3-stem", run / "g3-eval/stem-reference", ("validation.jsonl", "test.jsonl")),
    )
    sources = {}
    for label, directory, splits in corpora:
        for split in splits:
            records = load_split(directory, split)
            sources[f"{label}/{split}"] = {(record["lang"], canonical_text(record["text"])) for record in records}
    frozen_pairs = run / "downstream/data"
    if frozen_pairs.exists():
        pairs = downstream_data.load_pairs(frozen_pairs)
        sources["downstream/minimal-pairs"] = {(pair["lang"], pair[side]) for pair in pairs
                                                for side in downstream.SENTENCE_FIELDS}
    by_source = {}
    for source, texts in sources.items():
        overlaps = Counter(f"{row['split']}/{row['lang']}" for row in rows if (row["lang"], row["text"]) in texts)
        by_source[source] = dict(overlaps)
    audit = {"match_scope": "same language and exact normalized whole record",
             "dataset_sha256": sha256_file(out / "data/rows.jsonl"), "overlapping_rows": by_source}
    with TemporaryDirectory(prefix=".audit-", dir=out / "data") as temporary:
        staged = Path(temporary) / "overlap-audit.json"
        write_json(staged, audit)
        staged.replace(out / "data/overlap-audit.json")
    print(f"Recorded exact overlaps against {len(by_source)} prior splits and panels")


def text_key(row):
    return row["split"], row["lang"], str(row["id"])


def smoke_rows(rows):
    anchors = defaultdict(list)
    for row in rows:
        if row["lang"] == LANGUAGES[0]:
            anchors[row["split"]].append(str(row["id"]))
    ids = {split: set(identities[:2]) for split, identities in anchors.items()}
    return [row for row in rows if str(row["id"]) in ids[row["split"]]]


def input_fingerprints(out, seed, smoke):
    return {
        "seed": seed, "smoke": smoke,
        "data_sha256": sha256_file(out / "data/rows.jsonl"),
        "data_manifest_sha256": sha256_file(out / "data/manifest.json"),
        "router_sha256": sha256_file(Path(downstream_models.__file__)),
        "scorer_sha256": sha256_file(Path(downstream.__file__)),
        "metrics_sha256": sha256_file(Path(routing_metrics.__file__)),
        "routing_runner_sha256": sha256_file(Path(__file__)),
        "batch_size": BATCH_SIZE,
        "scoring": "complete sentence conditional on one initial EOS, without a scored end marker",
        "routing_policy": routing_metrics.POLICY,
    }


def cell_settings(router, cell, seed, inputs):
    group, route = CELLS[cell]
    provenance = router.provenance(route, seed)
    tokenizer = {key: value for key, value in provenance.items() if key in (
        "tokenizer", "eos_id", "model_revision", "langmap_manifest_sha256", "g3_selection_sha256",
    )}
    return {**inputs, "cell": cell, "model": router.provenance(group, seed), "scoring_tokenizer": tokenizer}


def verify_cell(directory, rows, expected):
    saved = read_json(directory / "run.json")
    if any(saved.get(field) != value for field, value in expected.items()):
        raise ValueError(f"{directory}: score provenance differs from this routing experiment")
    path = directory / "sentences.csv"
    if sha256_file(path) != saved["sentences_sha256"]:
        raise ValueError(f"{path}: score checksum differs")
    with path.open(newline="", encoding="utf-8") as stream:
        scores = list(csv.DictReader(stream))
    if [text_key(score) for score in scores] != [text_key(row) for row in rows]:
        raise ValueError(f"{path}: missing, reordered or unexpected sentences")
    for row, score in zip(rows, scores, strict=True):
        size, tokens, logp = int(score["bytes"]), int(score["tokens"]), float(score["logp"])
        if (size != len(row["text"].encode("utf-8")) or
                not 0 < tokens <= downstream.PROTOCOL["batch_positions"] or not math.isfinite(logp) or logp > 0):
            raise ValueError(f"{path}: invalid score for {text_key(row)}")
        score.update(bytes=size, tokens=tokens, logp=logp)
    return scores


def score_cell(directory, expected, rows, model, encoded, eos, device):
    scores = downstream.score_sequences(model, encoded, eos, device, BATCH_SIZE)
    data = [{"split": row["split"], "lang": row["lang"], "id": row["id"],
             "bytes": len(row["text"].encode("utf-8")), "tokens": len(tokens), "logp": score}
            for row, tokens, score in zip(rows, encoded, scores, strict=True)]
    with TemporaryDirectory(prefix=".quality-", dir=directory.parent) as temporary:
        staged = Path(temporary) / "cell"
        staged.mkdir()
        path = staged / "sentences.csv"
        downstream.write_csv(path, data)
        write_json(staged / "run.json", {**expected, "sentences_sha256": sha256_file(path)})
        verify_cell(staged, rows, expected)
        staged.rename(directory)


def result_dir(out, seed, smoke=False):
    return out / ("smoke" if smoke else "results") / f"seed-{seed}"


def publish_report(directory, cells):
    for name, rows in (("quality.csv", routing_metrics.routing_summary(cells)),
                       ("hybrid-sentences.csv", routing_metrics.hybrid_rows(cells))):
        result = directory / name
        pending = result.with_suffix(".tmp")
        downstream.write_csv(pending, rows)
        pending.replace(result)


def compare(run, out, seeds, smoke):
    import torch

    device, dtype = verified_runtime(read_json(run / "seed-42/run.json"))
    router = downstream_models.DownstreamModels(run, device=device, dtype=dtype)
    rows = routing_data.load_rows(out / "data")
    if smoke:
        rows = smoke_rows(rows)
    encodings = {}
    for seed in seeds:
        directory = result_dir(out, seed, smoke)
        directory.mkdir(parents=True, exist_ok=True)
        inputs = input_fingerprints(out, seed, smoke)
        with (directory / ".run.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError(f"{directory}: routing evaluation is already running") from None
            cells = {}
            model = None
            loaded_group = None
            for cell, (group, route) in CELLS.items():
                folder = directory / cell
                expected = cell_settings(router, cell, seed, inputs)
                if not folder.exists():
                    print(f"Scoring {cell}, seed {seed}, {len(rows)} sentences", flush=True)
                    if loaded_group != group:
                        del model
                        torch.cuda.empty_cache()
                        model = router.load(group, seed)
                        loaded_group = group
                    if route not in encodings:
                        encodings[route] = [router.encode(row["text"], row["lang"], route) for row in rows]
                    score_cell(folder, expected, rows, model, encodings[route], router.eos_id, device)
                cells[cell] = verify_cell(folder, rows, expected)
            del model
            torch.cuda.empty_cache()
            publish_report(directory, cells)
        print(f"Verified routing evaluation in {directory}", flush=True)


def summary(run, out, seeds):
    router = downstream_models.DownstreamModels(run)
    rows = routing_data.load_rows(out / "data")
    for seed in seeds:
        directory = result_dir(out, seed)
        inputs = input_fingerprints(out, seed, smoke=False)
        cells = {cell: verify_cell(directory / cell, rows, cell_settings(router, cell, seed, inputs))
                 for cell in CELLS}
        reports = routing_metrics.routing_summary(cells)
        print(f"FLORES-200 quality for seed {seed}. Lower BPB is better.")
        print("Split     Language  Cell        Sentences     BPB   Δ vs G1-BPE  Δ vs G4-G3  Δ vs G4-BPE  Tokens/byte")
        for row in reports:
            print(f"{row['split']:9} {row['lang']:9} {row['cell']:11} {row['documents']:9} {row['bpb']:8.4f} "
                  f"{row['bpb_vs_g1_bpe']:+12.4f} {row['bpb_vs_g4_g3']:+11.4f} "
                  f"{row['bpb_vs_g4_bpe']:+12.4f} {row['tokens_per_byte']:12.5f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "smoke", "run", "summary"))
    parser.add_argument("--run", type=Path, default=Path("runs/pilot"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--archive", type=Path, help="local copy of the pinned Meta FLORES-200 download")
    parser.add_argument("--seed", type=int, nargs="+", choices=SEEDS)
    args = parser.parse_args()
    out = args.out or args.run / "routing-quality"
    seeds = list(dict.fromkeys(args.seed or (42,)))
    if args.action == "prepare":
        routing_data.prepare(out / "data", args.archive)
        dataset_audit(args.run, out, routing_data.load_rows(out / "data"))
    elif args.action == "summary":
        summary(args.run, out, seeds)
    else:
        compare(args.run, out, seeds, smoke=args.action == "smoke")


if __name__ == "__main__":
    main()
