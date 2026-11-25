"""Train full-table and frequency-control arms against the frozen embedding pilot."""

import argparse
import csv
import fcntl
from itertools import product
from pathlib import Path
from statistics import mean, stdev
from tempfile import NamedTemporaryFile

from . import frequency_control
from .boundaries import training_stems
from .data import load_split, read_json, sha256_file, write_json
from .embedding_ablation import (
    CONTROL, FULL_GROUP, GROUPS, LANGS, SEEDS, baseline, fit_group, inputs, read_plan,
    read_scores, smoke_items, verified_group, verified_runtime, verify_matched_groups,
)
from .g3 import SPLITS, native_langmap, verified_eval_paths
from .g3_lora import budget
from .rescoring import load_segmenters

ARMS = {
    "full": (FULL_GROUP, None),
    "matched": ("frequency-matched", "matched_ids"),
    "high-frequency": ("high-frequency", "high_frequency_ids"),
}
SMOKE_STEPS = 2


def output(out):
    return out / "follow-up"


def provenance(out):
    return {
        "source_plan_sha256": sha256_file(out / "run.json"),
        "selection_code_sha256": sha256_file(Path(frequency_control.__file__)),
    }


def checked_plan(out):
    plan = read_json(output(out) / "plan.json")
    if any(plan.get(key) != value for key, value in provenance(out).items()):
        raise ValueError("Follow-up plan differs from the frozen embedding pilot or frequency selection code")
    return plan


def write_manifest(path, manifest):
    with NamedTemporaryFile(prefix=".manifest-", dir=path.parent, delete=False) as pending:
        staged = Path(pending.name)
    try:
        write_json(staged, manifest)
        staged.replace(path)
    finally:
        staged.unlink(missing_ok=True)


def plan_controls(run, out):
    boundary, _, pilot = inputs(run)
    source = read_plan(out, boundary, pilot)
    path = output(out) / "plan.json"
    path.parent.mkdir(exist_ok=True)
    with (path.parent / ".plan.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            checked_plan(out)
            print(f"Verified follow-up plan: {path}")
            return
        tokenizer = read_json(run / "langmap/model_tokenizer/tokenizer.json")
        ordinary = set(tokenizer["model"]["vocab"].values())
        ordinary.update(token["id"] for token in tokenizer["added_tokens"] if not token["special"])
        ordinary.difference_update(token["id"] for token in tokenizer["added_tokens"] if token["special"])
        labels = (doc["lang"] for doc in load_split(run / "corpus", "train.jsonl"))
        controls = frequency_control.frequency_controls(read_json(out / "training_ids.json"),
                                                       labels, ordinary, source)
        write_manifest(path, {**provenance(out), **controls})
    print(f"Frozen follow-up plan: {path}")


def arm_directory(out, arm, seed, smoke=False):
    prefix = "smoke-seed" if smoke else "seed"
    return output(out) / arm / f"{prefix}-{seed}"


def compare(run, out, arm, seed, smoke):
    boundary, selection, config = inputs(run)
    pilot = {**config, "seed": seed}
    source = read_plan(out, boundary, pilot)
    plan = checked_plan(out)
    expected = source["training_budget"]
    device, dtype = verified_runtime(pilot)
    group, key = ARMS[arm]
    draw = plan["draws"][str(seed)]
    ids = draw[key] if key is not None else source["draws"][str(seed)]["selected_ids"][:3]
    directory = arm_directory(out, arm, seed, smoke)
    directory.parent.mkdir(parents=True, exist_ok=True)
    with (directory.parent / f".{directory.name}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f"{directory} is already running") from None
        train = load_split(run / "corpus", "train.jsonl")
        train_ids = read_json(out / "training_ids.json")
        if budget(train_ids, pilot["context"], pilot["grad_accum"]) != expected:
            raise ValueError("Training token or optimization step budget differs from the pilot")
        encoded = [(doc["lang"], path) for doc, path in zip(train, train_ids, strict=True)]
        evaluation = verified_eval_paths(run, selection, load_segmenters(run), training_stems(train),
                                         native_langmap(run))
        if smoke:
            encoded = smoke_items(encoded, train, 3)
            docs, paths = evaluation["validation"]
            evaluation = {"validation": (smoke_items(docs, docs, 1),
                                         {group: smoke_items(paths[CONTROL], docs, 1)})}
            rows, baseline_sha = [], None
        else:
            rows, baseline_sha = baseline(boundary, seed, pilot, source, evaluation)
            evaluation = {split: (docs, {group: paths[CONTROL]}) for split, (docs, paths) in evaluation.items()}
        manifest = {**pilot, "group": group, "smoke": smoke, "smoke_steps": SMOKE_STEPS if smoke else None,
                    "planned_training_budget": expected, "base_plan_sha256": plan["source_plan_sha256"],
                    "followup_plan_sha256": sha256_file(output(out) / "plan.json"),
                    "g3_bpb_sha256": baseline_sha, "row_ids": ids if key is not None else None}
        directory.mkdir(exist_ok=True)
        run_manifest = directory / "run.json"
        if run_manifest.exists():
            if read_json(run_manifest) != manifest:
                raise ValueError(f"{directory}: different run settings")
        else:
            write_manifest(run_manifest, manifest)
        result = directory / "bpb.csv"
        groups = (group,) if smoke else (CONTROL, group)
        if not result.exists():
            rows.extend(fit_group(directory / group, ids, encoded, pilot, expected, evaluation,
                                  dtype, device, smoke, smoke_steps=SMOKE_STEPS))
            pending = result.with_suffix(".tmp")
            with pending.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            read_scores(pending, groups, evaluation)
            pending.replace(result)
        else:
            read_scores(result, groups, evaluation)
            verified_group(directory / group, expected, smoke, smoke_steps=SMOKE_STEPS)
        print(f"Verified results: {result}")


def summary(out, arms, seeds):
    plan = checked_plan(out)
    expected = read_json(out / "run.json")["training_budget"]
    source_scores, source_runs = [], []
    for seed in seeds:
        directory = out / f"seed-{seed}"
        manifest = read_json(directory / "run.json")
        if (manifest["plan_sha256"] != plan["source_plan_sha256"] or manifest["seed"] != seed
                or manifest["planned_training_budget"] != expected or manifest["smoke"]):
            raise ValueError(f"Unmatched G4/G6 result: {directory}")
        verify_matched_groups(directory, expected, smoke=False)
        source_scores.append(read_scores(directory / "bpb.csv", (CONTROL, *GROUPS), SPLITS))
        source_runs.append(manifest)

    for arm in arms:
        group, _ = ARMS[arm]
        results, costs = [], []
        for seed, original in zip(seeds, source_runs, strict=True):
            directory = arm_directory(out, arm, seed)
            current = read_json(directory / "run.json")
            fields = ("seed", "model_revision", "device", "dtype", "packages", "planned_training_budget",
                      "g3_bpb_sha256", "smoke")
            if (any(current[key] != original[key] for key in fields)
                    or current["group"] != group or current["base_plan_sha256"] != plan["source_plan_sha256"]
                    or current["followup_plan_sha256"] != sha256_file(output(out) / "plan.json")):
                raise ValueError(f"Unmatched follow-up result: {directory}")
            costs.append(verified_group(directory / group, expected, smoke=False))
            results.append(read_scores(directory / "bpb.csv", (CONTROL, group), SPLITS))
        print(f"\n{group} across seeds {', '.join(map(str, seeds))}. Lower BPB is better.")
        print("Split         Language    G4 BPB     Arm BPB    G4-Arm")
        for split, lang in product(SPLITS, LANGS):
            g4 = [score[split, "G4-drift", lang]["bpb"] for score in source_scores]
            other = [score[split, group, lang]["bpb"] for score in results]
            for source, candidate in zip(source_scores, results, strict=True):
                if source[split, CONTROL, lang] != candidate[split, CONTROL, lang]:
                    raise ValueError(f"Changed G3 reference for {split} {lang}")
            differences = [x - y for x, y in zip(g4, other, strict=True)]
            spread = f" ± {stdev(differences):.4f}" if len(differences) > 1 else ""
            print(f"{split:13} {lang:9} {mean(g4):10.4f} {mean(other):10.4f} "
                  f"{mean(differences):+9.4f}{spread}")
        for seed, stats in zip(seeds, costs, strict=True):
            print(f"Seed {seed}: {stats['trainable_parameters']:,} trainable parameters, "
                  f"{stats['train_seconds']:.1f}s, "
                  f"{stats['train_peak_allocated_bytes'] / 2**30:.2f} GiB peak allocated")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "smoke", "run", "summary"))
    parser.add_argument("--run", type=Path, default=Path("runs/pilot"))
    parser.add_argument("--out", type=Path, help="path to the existing G4/G6 plan")
    parser.add_argument("--arm", choices=tuple(ARMS), nargs="+")
    parser.add_argument("--seed", type=int, choices=SEEDS, nargs="+")
    args = parser.parse_args()
    out = args.out or args.run / "g3-eval" / "embedding-ablation"
    if args.action == "plan":
        plan_controls(args.run, out)
    elif args.action == "summary":
        summary(out, args.arm or ["matched", "full"], args.seed or SEEDS)
    else:
        for seed in args.seed or [SEEDS[0]]:
            for arm in args.arm or ["matched", "full"]:
                compare(args.run, out, arm, seed, smoke=args.action == "smoke")


if __name__ == "__main__":
    main()
