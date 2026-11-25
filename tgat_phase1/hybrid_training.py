"""Train G4 on the fixed mixed tokenizer paths and evaluate its matched inference route."""

import argparse
import fcntl
from pathlib import Path
from tempfile import TemporaryDirectory

from . import downstream, downstream_models, hybrid_training_metrics, hybrid_training_plan
from . import routing_data, routing_metrics, routing_quality
from .data import load_split, read_json, sha256_file, write_json
from . import embedding_ablation
from .embedding_ablation import SEEDS, smoke_items, trained_group, verified_group, verified_runtime
from .g3_lora import budget

GROUP = "G4-hybrid-trained"
SMOKE_STEPS = 2


def output_dir(out, seed, smoke):
    return out / ("smoke" if smoke else "results") / f"seed-{seed}"


def reference_settings(run, seed, smoke, router):
    out = run / "routing-quality"
    rows = routing_data.load_rows(out / "data")
    if smoke:
        rows = routing_quality.smoke_rows(rows)
    settings = routing_quality.input_fingerprints(out, seed, smoke)
    root = routing_quality.result_dir(out, seed, smoke)
    sources, digests = {}, {}
    for cell in routing_quality.CELLS:
        directory = root / cell
        sources[cell] = routing_quality.verify_cell(
            directory, rows, routing_quality.cell_settings(router, cell, seed, settings),
        )
        digests[cell] = sha256_file(directory / "run.json")
    cells = {"G1-BPE": sources["G1-BPE"], "G4-hybrid": routing_metrics.hybrid_rows(sources)}
    return rows, cells, {**settings, "references_sha256": digests}


def experiment_settings(run, out, seed, smoke, plan, router, reference):
    return {
        "seed": seed,
        "smoke": smoke,
        "smoke_steps": SMOKE_STEPS if smoke else None,
        "plan_sha256": sha256_file(out / "plan.json"),
        "row_ids": plan["row_ids_by_seed"][str(seed)],
        "training_budget": plan["training_budget"],
        "training_runner_sha256": sha256_file(Path(__file__)),
        "training_code_sha256": {
            source.name: sha256_file(source)
            for source in (Path(embedding_ablation.__file__), Path(__file__).with_name("train.py"),
                           Path(__file__).with_name("tied_rows.py"))
        },
        "original_g4": router.provenance("G4", seed),
        "reference": reference,
    }


def publish_settings(path, expected):
    if path.exists():
        if read_json(path) != expected:
            raise ValueError(f"{path}: saved settings differ from this experiment")
    else:
        pending = path.with_suffix(".tmp")
        write_json(pending, expected)
        pending.replace(path)


def checkpoint_statistics(run, directory, seed, smoke, plan):
    stats = verified_group(directory / GROUP, plan["training_budget"], smoke, smoke_steps=SMOKE_STEPS)
    original = read_json(run / "g3-eval/embedding-ablation" / f"seed-{seed}/G4-drift/training.json")
    for key in ("embedding_parameters", "trainable_parameters"):
        if stats[key] != original[key]:
            raise ValueError(f"Trained hybrid differs from original G4 in {key}")
    return stats


def collect_scores(directory, rows, cells, expected, stats):
    settings = {**expected, "trained_artifacts_sha256": stats["artifacts_sha256"], "cell": GROUP}
    return {**cells, GROUP: routing_quality.verify_cell(directory / "flores", rows, settings)}


def compare(run, out, seeds, smoke):
    import torch

    plan = hybrid_training_plan.read_plan(run, out)
    pilot = read_json(run / "seed-42/run.json")
    device, dtype = verified_runtime(pilot)
    router = downstream_models.DownstreamModels(run, device=device, dtype=dtype)
    documents = load_split(run / "corpus", "train.jsonl")
    paths = read_json(out / "training_ids.json")
    expected_budget = budget(paths, pilot["context"], pilot["grad_accum"])
    if expected_budget != plan["training_budget"]:
        raise ValueError("Hybrid training paths differ from the frozen token and step budget")
    encoded = [(doc["lang"], ids) for doc, ids in zip(documents, paths, strict=True)]
    if smoke:
        encoded = smoke_items(encoded, documents, 3)
    for seed in seeds:
        reference_out = run / "routing-quality"
        reference_dir = routing_quality.result_dir(reference_out, seed, smoke)
        if any(not (reference_dir / cell / "run.json").exists() for cell in routing_quality.CELLS):
            routing_quality.compare(run, reference_out, (seed,), smoke)
        rows, controls, reference = reference_settings(run, seed, smoke, router)
        expected = experiment_settings(run, out, seed, smoke, plan, router, reference)
        directory = output_dir(out, seed, smoke)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / ".run.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError(f"{directory}: hybrid training is already running") from None
            publish_settings(directory / "run.json", expected)
            if not (directory / "flores").exists():
                model = trained_group(directory / GROUP, expected["row_ids"], encoded, {**pilot, "seed": seed},
                                      expected_budget, dtype, device, smoke, smoke_steps=SMOKE_STEPS)
                stats = checkpoint_statistics(run, directory, seed, smoke, plan)
                routes = {lang: routing_quality.CELLS[cell][1] for lang, cell in routing_metrics.POLICY.items()}
                eval_ids = [router.encode(row["text"], row["lang"], routes[row["lang"]]) for row in rows]
                settings = {**expected, "trained_artifacts_sha256": stats["artifacts_sha256"], "cell": GROUP}
                routing_quality.score_cell(directory / "flores", settings, rows, model, eval_ids, router.eos_id, device)
                del model
                torch.cuda.empty_cache()
            stats = checkpoint_statistics(run, directory, seed, smoke, plan)
            cells = collect_scores(directory, rows, controls, expected, stats)
            with TemporaryDirectory(prefix=".summary-", dir=directory) as temporary:
                report = Path(temporary) / "quality.csv"
                downstream.write_csv(report, hybrid_training_metrics.hybrid_training_summary(cells))
                report.replace(directory / "quality.csv")
        print(f"Verified hybrid training results in {directory}", flush=True)


def summary(run, out, seeds):
    plan = hybrid_training_plan.read_plan(run, out)
    router = downstream_models.DownstreamModels(run)
    for seed in seeds:
        directory = output_dir(out, seed, smoke=False)
        rows, controls, reference = reference_settings(run, seed, False, router)
        expected = experiment_settings(run, out, seed, False, plan, router, reference)
        if read_json(directory / "run.json") != expected:
            raise ValueError(f"{directory}: saved settings differ from this experiment")
        stats = checkpoint_statistics(run, directory, seed, False, plan)
        cells = collect_scores(directory, rows, controls, expected, stats)
        print(f"FLORES-200 BPB by split and language, seed {seed}")
        print("Split     Language    G1-BPE   G4 inference   G4 trained   Trained−inference   Trained−G1")
        for row in hybrid_training_metrics.hybrid_training_summary(cells):
            print(f"{row['split']:9} {row['lang']:9} {row['g1_bpe_bpb']:9.4f} "
                  f"{row['g4_hybrid_bpb']:14.4f} {row['g4_hybrid_trained_bpb']:12.4f} "
                  f"{row['trained_vs_inference_bpb']:+19.4f} {row['trained_gap_bpb']:+12.4f}")
        print(f"Trained {stats['train_tokens']:,} targets in {stats['optimizer_steps']:,} optimizer steps. "
              f"Peak allocated {stats['train_peak_allocated_bytes'] / 2**30:.2f} GiB.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "smoke", "run", "summary"))
    parser.add_argument("--run", type=Path, default=Path("runs/pilot"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int, nargs="+", choices=SEEDS)
    args = parser.parse_args()
    out = args.out or args.run / "hybrid-training"
    seeds = list(dict.fromkeys(args.seed or (42,)))
    if args.action == "plan":
        plan = hybrid_training_plan.prepare(args.run, out)
        current, original = plan["training_budget"], plan["original_g4_training_budget"]
        print(f"Frozen training paths in {out / 'plan.json'}")
        for label, setting in (("Hybrid G4", current), ("Original G4", original)):
            print(f"{label} uses {setting['train_tokens']:,} training targets and "
                  f"{setting['optimizer_steps']:,} optimizer steps")
    elif args.action == "summary":
        summary(args.run, out, seeds)
    else:
        compare(args.run, out, seeds, smoke=args.action == "smoke")


if __name__ == "__main__":
    main()
