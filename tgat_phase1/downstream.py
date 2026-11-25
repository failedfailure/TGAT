"""Evaluate frozen grammatical minimal pairs with the TGAT checkpoints."""

import argparse
from collections import defaultdict
import csv
import fcntl
from itertools import islice
import math
from pathlib import Path
from statistics import mean, stdev
from tempfile import TemporaryDirectory

from . import downstream_data, downstream_models
from .data import canonical_text, load_split, read_json, sha256_file, write_json
from .embedding_ablation import SEEDS, verified_runtime

GROUPS = downstream_models.GROUPS
SENTENCE_FIELDS = ("good", "bad")
METRICS = ("raw", "byte")
PROTOCOL = {
    "primary": "sum of causal log probabilities along each complete sentence's token path",
    "sensitivity": "causal log probability per UTF-8 byte",
    "initial_context": "checkpoint EOS, not scored",
    "terminal_eos_scored": False,
    "correct": "strictly greater score for the acceptable sentence",
    "ties": "counted separately and scored as incorrect",
    "batch_positions": 512,
    "overlap_sensitivity": "exclude pairs with either sentence in the G3 stem-reference validation or test split",
}


def pair_key(pair):
    return tuple(str(pair[field]) for field in ("dataset", "lang", "phenomenon", "id"))


def fixed_smoke(pairs):
    suites = defaultdict(list)
    for pair in pairs:
        suites[pair["dataset"], pair["lang"]].append(pair)
    selected = []
    for suite in suites.values():
        overlapping = next((pair for pair in suite if pair["reference_overlap"]), None)
        separate = next((pair for pair in suite if not pair["reference_overlap"]), None)
        selected.extend([overlapping, separate] if overlapping and separate else suite[:2])
    return selected


def audited_pairs(run, out):
    reference = run / "g3-eval/stem-reference"
    splits = {
        split: {(row["lang"], canonical_text(row["text"]))
                for row in load_split(reference, f"{split}.jsonl")}
        for split in ("validation", "test")
    }
    pairs = downstream_data.load_pairs(out / "data")
    for pair in pairs:
        pair["reference_overlap"] = ",".join(
            split for split, sentences in splits.items()
            if any((pair["lang"], pair[field]) in sentences for field in SENTENCE_FIELDS)
        )
    return pairs, sha256_file(reference / "manifest.json")


def score_sequences(model, sequences, eos, device, batch_size):
    """Return each sentence's total log likelihood with right-padded causal batches."""
    import torch
    from torch.nn import functional as F

    results = [None] * len(sequences)
    positions = PROTOCOL["batch_positions"]
    if any(not 0 < len(sequence) <= positions for sequence in sequences):
        raise ValueError(f"Every sentence needs 1–{positions} targets for complete-sentence scoring")
    order = iter(sorted(range(len(sequences)), key=lambda index: -len(sequences[index])))
    model.eval()
    with torch.inference_mode():
        while (first := next(order, None)) is not None:
            width = len(sequences[first])
            batch = [first, *islice(order, min(batch_size, positions // width) - 1)]
            input_rows, target_rows, masks = [], [], []
            for index in batch:
                ids = list(sequences[index])
                pads = width - len(ids)
                input_rows.append([eos, *ids[:-1], *([eos] * pads)])
                target_rows.append([*ids, *([-100] * pads)])
                masks.append([1] * len(ids) + [0] * pads)
            inputs = torch.tensor(input_rows, dtype=torch.long, device=device)
            targets = torch.tensor(target_rows, dtype=torch.long, device=device)
            mask = torch.tensor(masks, dtype=torch.long, device=device)
            logits = model(input_ids=inputs, attention_mask=mask, use_cache=False).logits
            loss = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten(), reduction="none")
            totals = (-loss.view(len(batch), width).sum(dim=1, dtype=torch.float64)).tolist()
            for index, total in zip(batch, totals, strict=True):
                if not math.isfinite(total) or total > 0:
                    raise RuntimeError("Non-finite or positive sentence log likelihood")
                results[index] = total
    return results


def score_pairs(model, router, pairs, group, device, batch_size):
    sentences = dict.fromkeys((pair["lang"], pair[field]) for pair in pairs for field in SENTENCE_FIELDS)
    encoded = []
    for lang, sentence in sentences:
        encoded.append(router.encode(sentence, lang, group))
    predictions = score_sequences(model, encoded, router.eos_id, device, batch_size)
    for (key, ids, score) in zip(sentences, encoded, predictions, strict=True):
        sentences[key] = score, len(ids), len(key[1].encode("utf-8"))
    rows = []
    for pair in pairs:
        row = {field: pair[field] for field in ("dataset", "lang", "phenomenon", "id", "reference_overlap")}
        for field in SENTENCE_FIELDS:
            score, tokens, size = sentences[pair["lang"], pair[field]]
            row.update({f"{field}_logp": score, f"{field}_tokens": tokens, f"{field}_bytes": size})
        for metric, margin in score_margins(row).items():
            row[f"{metric}_margin"] = margin
            row[f"{metric}_correct"] = int(margin > 0.0)
        rows.append(row)
    return rows


def score_margins(row):
    good, bad = (row[f"{field}_logp"] for field in SENTENCE_FIELDS)
    return {
        "raw": good - bad,
        "byte": good / row["good_bytes"] - bad / row["bad_bytes"],
    }


def aggregate(rows):
    suites = defaultdict(list)
    for row in rows:
        suite = row["dataset"], row["lang"]
        suites[suite].append(row)
    summaries = []
    for (dataset, lang), records in sorted(suites.items()):
        subsets = {"full": records}
        outside = [row for row in records if not row["reference_overlap"]]
        if 0 < len(outside) < len(records):
            subsets["outside-reference"] = outside
        for subset, items in subsets.items():
            phenomena = defaultdict(list)
            for row in items:
                phenomena[row["phenomenon"]].append(row)
            for phenomenon, examples in [("all", items), *sorted(phenomena.items())]:
                scores = {}
                for metric in METRICS:
                    scores[f"{metric}_accuracy"] = mean(row[f"{metric}_correct"] for row in examples)
                    scores[f"{metric}_ties"] = sum(row[f"{metric}_margin"] == 0.0 for row in examples)
                    scores[f"{metric}_mean_margin"] = mean(row[f"{metric}_margin"] for row in examples)
                summaries.append({"dataset": dataset, "lang": lang, "subset": subset, "phenomenon": phenomenon,
                                  "pairs": len(examples), **scores})
    return summaries


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verified_scores(directory, pairs, expected=None):
    saved = read_json(directory / "run.json")
    if expected is not None and any(saved.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{directory}: saved run settings differ; choose another --out for a new protocol")
    path = directory / "pairs.csv"
    if any(sha256_file(directory / name) != digest for name, digest in saved["output_sha256"].items()):
        raise ValueError(f"{directory}: a saved result failed its checksum")
    with path.open(newline="", encoding="utf-8") as handle:
        scores = list(csv.DictReader(handle))
    if [pair_key(row) for row in scores] != [pair_key(pair) for pair in pairs]:
        raise ValueError(f"{directory}: missing, reordered, or unexpected pairs")
    for row, pair in zip(scores, pairs, strict=True):
        if row["reference_overlap"] != pair["reference_overlap"]:
            raise ValueError(f"{directory}: reference-overlap label differs for {pair_key(row)}")
        for field in SENTENCE_FIELDS:
            logp = float(row[f"{field}_logp"])
            size = int(row[f"{field}_bytes"])
            tokens = int(row[f"{field}_tokens"])
            if (not math.isfinite(logp) or logp > 0 or size != len(pair[field].encode("utf-8"))
                    or not 0 < tokens <= PROTOCOL["batch_positions"]):
                raise ValueError(f"{directory}: invalid score or text length for {pair_key(row)}")
            row.update({f"{field}_logp": logp, f"{field}_bytes": size, f"{field}_tokens": tokens})
        for metric, margin in score_margins(row).items():
            if (float(row[f"{metric}_margin"]) != margin
                    or int(row[f"{metric}_correct"]) != int(margin > 0.0)):
                raise ValueError(f"{directory}: invalid {metric} margin or label for {pair_key(row)}")
            row.update({f"{metric}_margin": margin, f"{metric}_correct": int(margin > 0.0)})
    return scores


def result_path(out, group, seed, smoke=False):
    suffix = "base" if group == "G0" else f"seed-{seed}"
    return out / ("smoke" if smoke else "results") / f"{group}-{suffix}"


def data_identity(out, reference_sha):
    return {
        "data_manifest_sha256": sha256_file(out / "data/manifest.json"),
        "data_sha256": sha256_file(out / "data/pairs.jsonl"),
        "g3_reference_manifest_sha256": reference_sha,
        "protocol": PROTOCOL,
    }


def evaluate(run, out, groups, seeds, batch_size, smoke):
    import torch

    pilot = read_json(run / "seed-42/run.json")
    device, dtype = verified_runtime(pilot)
    router = downstream_models.DownstreamModels(run, device=device, dtype=dtype)
    pairs, reference_sha = audited_pairs(run, out)
    if smoke:
        pairs = fixed_smoke(pairs)
    identity = {**data_identity(out, reference_sha), "smoke": smoke, "batch_size": batch_size,
                "scorer_sha256": sha256_file(Path(__file__)),
                "router_sha256": sha256_file(Path(downstream_models.__file__))}
    visited = set()
    for seed in seeds:
        for group in groups:
            directory = result_path(out, group, seed, smoke)
            if directory in visited:
                continue
            visited.add(directory)
            expected = {**identity, "group": group, "seed": None if group == "G0" else seed,
                        "model": router.provenance(group, seed)}
            directory.parent.mkdir(parents=True, exist_ok=True)
            with (directory.parent / f".{directory.name}.lock").open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise ValueError(f"{directory} is already running") from None
                if not directory.exists():
                    print(f"Scoring {group}, seed {expected['seed']}, {len(pairs)} pairs", flush=True)
                    model = router.load(group, seed)
                    rows = score_pairs(model, router, pairs, group, device, batch_size)
                    del model
                    torch.cuda.empty_cache()
                    with TemporaryDirectory(prefix=".downstream-", dir=directory.parent) as temporary:
                        staged = Path(temporary) / "result"
                        staged.mkdir()
                        write_csv(staged / "pairs.csv", rows)
                        write_csv(staged / "summary.csv", aggregate(rows))
                        output_sha256 = {name: sha256_file(staged / name) for name in ("pairs.csv", "summary.csv")}
                        write_json(staged / "run.json", {**expected, "output_sha256": output_sha256})
                        verified_scores(staged, pairs, expected)
                        staged.rename(directory)
                else:
                    verified_scores(directory, pairs, expected)
                print(f"Verified benchmark results: {directory}", flush=True)


def summarize(run, out, groups, seeds, baseline):
    pairs, reference_sha = audited_pairs(run, out)
    identity = {**data_identity(out, reference_sha), "smoke": False}
    tables = {}
    for group in dict.fromkeys((baseline, *groups)):
        for seed in seeds:
            directory = result_path(out, group, seed)
            rows = verified_scores(directory, pairs, {**identity, "group": group,
                                                      "seed": None if group == "G0" else seed})
            tables[group, seed] = {(row["dataset"], row["lang"], row["subset"]): row
                                   for row in aggregate(rows) if row["phenomenon"] == "all"}
    print(f"Accuracy (%) and paired percentage-point change vs {baseline} for seeds {', '.join(map(str, seeds))}")
    print("Dataset      Language  Subset             Group                   Pairs      Raw     Δ raw        Per byte   Δ per byte")
    suites = sorted(tables[baseline, seeds[0]])
    for suite in suites:
        for group in groups:
            cells = [tables[group, seed][suite] for seed in seeds]
            base = [tables[baseline, seed][suite] for seed in seeds]
            values = []
            for metric in METRICS:
                key = f"{metric}_accuracy"
                deltas = [100 * (target[key] - reference[key]) for target, reference in zip(cells, base, strict=True)]
                spread = f"±{stdev(deltas):.2f}" if len(deltas) > 1 else ""
                values.append(f"{100 * mean(cell[key] for cell in cells):7.2f} {mean(deltas):+7.2f}{spread:7}")
            print(f"{suite[0]:12} {suite[1]:9} {suite[2]:18} {group:23} {cells[0]['pairs']:5}  {' '.join(values)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "smoke", "run", "summary"))
    parser.add_argument("--run", type=Path, default=Path("runs/pilot"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--groups", nargs="+", choices=GROUPS)
    parser.add_argument("--seed", nargs="+", type=int, choices=SEEDS)
    parser.add_argument("--baseline", choices=GROUPS, default="G1")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    out = args.out or args.run / "downstream"
    seeds = list(dict.fromkeys(args.seed or (42,)))
    default = GROUPS if seeds == [42] else ("G0", "G1", "G2", "G3", "G4", "G6")
    groups = list(dict.fromkeys(args.groups or default))
    if args.action == "prepare":
        downstream_data.prepare(out / "data")
    elif args.action == "summary":
        summarize(args.run, out, groups, seeds, args.baseline)
    else:
        evaluate(args.run, out, groups, seeds, args.batch_size, smoke=args.action == "smoke")


if __name__ == "__main__":
    main()
