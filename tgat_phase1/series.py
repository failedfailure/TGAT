"""Continue and inspect a frozen Phase 1 pilot from the WSL host."""

import argparse
from collections import Counter
import csv
from datetime import datetime
import fcntl
from itertools import product
import math
from pathlib import Path
import statistics
import subprocess
import sys

from .data import LANGUAGES, load_split, read_json, sha256_file


GROUPS = ("G1-original", "G2-langmap")
POOLED_GROUP = "pooled-unigram"
STAGES = ("before_lora", "after_lora")
REPORT_LANGUAGES = (*LANGUAGES, "all")


def inspect_seed(run, seed, reference, expected_bytes, *, pooled=False):
    groups = (POOLED_GROUP,) if pooled else GROUPS
    directory = run / (f"pooled-seed-{seed}" if pooled else f"seed-{seed}")
    if not directory.exists():
        return "missing", "no output directory", None
    if not (directory / "bpb.csv").is_file():
        saved = [group for group in groups if (directory / group / "training.json").is_file()]
        return "incomplete", f"bpb.csv pending; saved groups: {', '.join(saved) or 'none'}", None

    try:
        config = read_json(directory / "run.json")
        for field, value in reference.items():
            if config.get(field) != (seed if field == "seed" else value):
                raise ValueError(f"run.json differs at {field}")
        with (directory / "bpb.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        expected = set(product(groups, STAGES, REPORT_LANGUAGES))
        scores = {}
        nlls = {}
        for row in rows:
            key = (row["group"], row["stage"], row["lang"])
            if key not in expected or key in scores:
                raise ValueError(f"unexpected or duplicate row {key}")
            byte_count = int(row["bytes"])
            token_count = int(row["tokens"])
            nll, bpb = float(row["nll_nats"]), float(row["bpb"])
            if (byte_count != expected_bytes[key[2]] or token_count < 1 or nll < 0
                    or not math.isfinite(bpb)
                    or not math.isclose(bpb, nll / (byte_count * math.log(2)), rel_tol=1e-9)):
                raise ValueError(f"invalid evaluation metrics for {key}")
            scores[key] = bpb
            nlls[key] = nll
        if scores.keys() != expected:
            raise ValueError(f"expected {len(expected)} BPB rows")
        for group, stage in product(groups, STAGES):
            if not math.isclose(nlls[group, stage, "all"],
                                sum(nlls[group, stage, lang] for lang in LANGUAGES)):
                raise ValueError(f"incorrect aggregate for {group} {stage}")
        for group in groups:
            output = directory / group
            stats = read_json(output / "training.json")
            if stats["train_tokens"] < 1 or stats["optimizer_steps"] < 1:
                raise ValueError(f"invalid training counters for {group}")
            if (output / "adapter/adapter_model.safetensors").stat().st_size == 0:
                raise ValueError(f"empty adapter for {group}")
    except (OSError, TypeError, ValueError, KeyError) as error:
        return "invalid", str(error), None
    return "verified", f"{len(groups)} LoRA group(s) and {len(expected)} BPB rows", scores


def launch(run, seed, reference, *, pooled=False, output_root=None):
    log_dir = (output_root or run) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    prefix = "pooled-seed" if pooled else "seed"
    log = log_dir / f"{prefix}-{seed}-{timestamp}.log"
    project = Path(__file__).resolve().parent.parent
    # Compose mounts the project at /workspace, so pass a mount-relative path.
    cmd = ["bash", "run.sh", "compare", "--run", str(run.relative_to(project)),
           "--seed", str(seed), "--epochs", str(reference["epochs"]),
           "--context", str(reference["context"]), "--grad-accum", str(reference["grad_accum"]),
           "--learning-rate", str(reference["learning_rate"])]
    if not reference["gradient_checkpointing"]:
        cmd.append("--no-gradient-checkpointing")
    if pooled:
        cmd.append("--pooled-control")
    if output_root is not None:
        cmd.extend(("--output-root", str(output_root.relative_to(project))))
    print(f"Starting {prefix}-{seed}. Log: {log}", flush=True)
    with log.open("x", encoding="utf-8") as stream:
        with subprocess.Popen(cmd, cwd=project, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, errors="replace") as process:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end="", flush=True)
            if process.wait() != 0:
                raise ValueError(f"seed {seed} exited {process.returncode}; see {log}")


def show_summary(results):
    print("\nSeed BPB (before -> after)")
    print("Seed  Language   G1 Original         G2 LangMAP         After G2 - G1")
    gaps = {lang: [] for lang in REPORT_LANGUAGES}
    for seed, scores in results:
        for lang in REPORT_LANGUAGES:
            a, b, c, d = [scores[group, stage, lang] for group, stage in product(GROUPS, STAGES)]
            gaps[lang].append(d - b)
            print(f"{seed:>4}  {lang:<9}  {a:.4f} -> {b:.4f}     {c:.4f} -> {d:.4f}     {d-b:+.4f}")
    if results:
        print(f"\nPaired post-LoRA G2 - G1 across {len(results)} verified seed(s)")
        for lang, values in gaps.items():
            spread = f", sample SD {statistics.stdev(values):.4f}" if len(values) > 1 else ""
            print(f"{lang:<9}  mean {statistics.mean(values):+.4f}{spread}")


def show_pooled_summary(results):
    if not results:
        return
    print("\nPaired BPB deltas (mean ± sample SD when available)")
    print("Stage         Language   Pooled - Original   LangMAP - Pooled")
    for stage, lang in product(STAGES, REPORT_LANGUAGES):
        triples = [(base[GROUPS[0], stage, lang], pooled[POOLED_GROUP, stage, lang],
                    base[GROUPS[1], stage, lang]) for pooled, base in results]
        gaps = ([b - a for a, b, _ in triples], [c - b for _, b, c in triples])
        cells = []
        for values in gaps:
            spread = f" ± {statistics.stdev(values):.4f}" if len(values) > 1 else ""
            cells.append(f"{statistics.mean(values):+.4f}{spread}")
        print(f"{stage:<13} {lang:<9} {cells[0]:>18} {cells[1]:>18}")


def show_duration_summary(results, baselines, epochs):
    if not results:
        return
    print(f"\nMean post-LoRA BPB change from 1 to {epochs} epochs across {len(results)} paired seed(s)")
    print("Language        G1 change   G2 change  G2-G1 gap change")
    for lang in REPORT_LANGUAGES:
        g1, g2 = [
            statistics.mean(scores[group, "after_lora", lang] - baselines[seed][group, "after_lora", lang]
                            for seed, scores in results)
            for group in GROUPS
        ]
        print(f"{lang:<12}  {g1:>+11.4f} {g2:>+11.4f} {g2-g1:>17.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("batch", "status", "summary", "pooled-batch", "pooled-summary"))
    parser.add_argument("--run", type=Path, default=Path("runs/pilot"))
    parser.add_argument("--epochs", type=int, help="run or inspect a separate training-duration arm")
    parser.add_argument("seeds", nargs="*", type=int)
    args = parser.parse_intermixed_args()
    run = args.run.resolve()
    seeds = list(dict.fromkeys(args.seeds or (42, 43, 44)))
    pooled = args.command.startswith("pooled-")
    if args.epochs is not None and (pooled or args.epochs < 2):
        parser.error("--epochs requires at least 2 and applies to G1/G2 runs")
    output_root = run / f"dose-{args.epochs}" if args.epochs is not None else run
    try:
        reference = read_json(run / "seed-42/run.json")
        if args.epochs is not None and reference["epochs"] != 1:
            raise ValueError("the duration arm requires a one-epoch baseline")
        for stage in ("corpus", "langmap"):
            if sha256_file(run / stage / "manifest.json") != reference[f"{stage}_manifest_sha256"]:
                raise ValueError(f"{stage} manifest differs from seed 42")
        counts = Counter()
        for row in load_split(run / "corpus", "heldout.jsonl"):
            counts[row["lang"]] += len(row["text"].encode("utf-8"))
        counts["all"] = sum(counts.values())
        expected = {**reference, "epochs": args.epochs} if args.epochs is not None else reference
        baselines = {}
        if pooled or args.epochs is not None:
            for seed in seeds:
                status, detail, scores = inspect_seed(run, seed, reference, counts)
                if status != "verified":
                    raise ValueError(f"comparison requires verified baseline seed {seed}: {detail}")
                baselines[seed] = scores
        if args.command in ("batch", "pooled-batch"):
            if inspect_seed(run, 42, reference, counts)[0] != "verified":
                raise ValueError("batch requires a verified seed 42")
            with (run / ".batch.lock").open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise ValueError("another batch is already running for this pilot") from None
                for seed in seeds:
                    status, detail, _ = inspect_seed(output_root, seed, expected, counts, pooled=pooled)
                    if status == "verified":
                        print(f"Seed {seed}: verified, skipping", flush=True)
                        continue
                    if status != "missing":
                        raise ValueError(f"seed {seed} {status}: {detail}. Existing output preserved.")
                    launch(run, seed, expected, pooled=pooled,
                           output_root=output_root if args.epochs is not None else None)
                    status, detail, _ = inspect_seed(output_root, seed, expected, counts, pooled=pooled)
                    if status != "verified":
                        raise ValueError(f"seed {seed} {status}: {detail}")
                    print(f"Seed {seed}: Verified BPB", flush=True)
        results = []
        for seed in seeds:
            status, detail, scores = inspect_seed(output_root, seed, expected, counts, pooled=pooled)
            print(f"Seed {seed}: {status} ({detail})")
            if scores is not None:
                results.append((seed, scores))
        if args.command != "status":
            if pooled:
                show_pooled_summary([(scores, baselines[seed]) for seed, scores in results])
            else:
                show_summary(results)
                if args.epochs is not None:
                    show_duration_summary(results, baselines, args.epochs)
            return int(len(results) != len(seeds))
        return 0
    except (OSError, TypeError, ValueError, KeyError) as error:
        print(f"TGAT: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
