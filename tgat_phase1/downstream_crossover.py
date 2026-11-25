"""Score G1 and G4 under both inference tokenization paths."""

import argparse
import fcntl
from pathlib import Path
from statistics import mean, stdev
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from . import downstream, downstream_models
from .crossover_metrics import CELLS, crossover_summary
from .data import read_json, sha256_file, write_json
from .embedding_ablation import SEEDS, verified_runtime

NATIVE = {"G1": "BPE", "G4": "G3"}
ROUTES = {route: group for group, route in NATIVE.items()}
BATCH_SIZE = 16
EFFECTS = (
    "tokenizer_at_g1_pp", "tokenizer_at_g4_pp", "checkpoint_at_bpe_pp",
    "checkpoint_at_g3_pp", "interaction_pp",
)


def output_dir(out, seed, smoke):
    return out / "crossover" / ("smoke" if smoke else "results") / f"seed-{seed}"


def score_settings(out, reference_sha, smoke):
    return {**downstream.data_identity(out, reference_sha), "smoke": smoke, "batch_size": BATCH_SIZE,
            "scorer_sha256": sha256_file(Path(downstream.__file__)),
            "router_sha256": sha256_file(Path(downstream_models.__file__))}


def checkpoint_settings(router, group, seed, protocol):
    return {**protocol, "group": group, "seed": seed, "model": router.provenance(group, seed)}


def crossed_settings(router, group, route, seed, protocol):
    base = checkpoint_settings(router, group, seed, protocol)
    routes = router.provenance(ROUTES[route], seed)
    tokenizer = {key: value for key, value in routes.items() if key in (
        "tokenizer", "eos_id", "model_revision", "langmap_manifest_sha256", "g3_selection_sha256",
    )}
    return {**base, "scoring_route": route, "scoring_tokenizer": tokenizer,
            "crossover_sha256": sha256_file(Path(__file__))}


def score_crossed(directory, expected, router, pairs, group, route, seed, device):
    import torch

    print(f"Scoring {group} with {route}, seed {seed}, {len(pairs)} pairs", flush=True)
    model = router.load(group, seed)
    encoder_group = ROUTES[route]
    scorer = SimpleNamespace(eos_id=router.eos_id,
                             encode=lambda text, lang, _: router.encode(text, lang, encoder_group))
    rows = downstream.score_pairs(model, scorer, pairs, group, device, BATCH_SIZE)
    del model
    torch.cuda.empty_cache()
    with TemporaryDirectory(prefix=".crossed-", dir=directory.parent) as temporary:
        staged = Path(temporary) / "result"
        staged.mkdir()
        downstream.write_csv(staged / "pairs.csv", rows)
        downstream.write_csv(staged / "summary.csv", downstream.aggregate(rows))
        outputs = {name: sha256_file(staged / name) for name in ("pairs.csv", "summary.csv")}
        write_json(staged / "run.json", {**expected, "output_sha256": outputs})
        downstream.verified_scores(staged, pairs, expected)
        staged.rename(directory)


def collect_cells(out, seed, smoke, router, pairs, protocol, device=None):
    root = output_dir(out, seed, smoke)
    cells = {}
    for group, route in CELLS:
        if route == NATIVE[group]:
            directory = downstream.result_path(out, group, seed, smoke)
            expected = checkpoint_settings(router, group, seed, protocol)
        else:
            directory = root / f"{group}-{route}"
            expected = crossed_settings(router, group, route, seed, protocol)
            if device is not None and not directory.exists():
                score_crossed(directory, expected, router, pairs, group, route, seed, device)
        cells[group, route] = downstream.verified_scores(directory, pairs, expected)
    return cells


def run_crossover(run, out, seeds, smoke):
    pilot = read_json(run / "seed-42/run.json")
    device, dtype = verified_runtime(pilot)
    router = downstream_models.DownstreamModels(run, device=device, dtype=dtype)
    pairs, reference_sha = downstream.audited_pairs(run, out)
    protocol = score_settings(out, reference_sha, smoke)
    if smoke:
        pairs = downstream.fixed_smoke(pairs)
    for seed in seeds:
        missing = [group for group in NATIVE if not downstream.result_path(out, group, seed, smoke).exists()]
        if missing:
            downstream.evaluate(run, out, missing, (seed,), BATCH_SIZE, smoke)
        # Validate both reused cells before computing either cross.
        for group in NATIVE:
            expected = checkpoint_settings(router, group, seed, protocol)
            downstream.verified_scores(downstream.result_path(out, group, seed, smoke), pairs, expected)
        root = output_dir(out, seed, smoke)
        root.mkdir(parents=True, exist_ok=True)
        with (root / ".run.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError(f"{root}: crossover is already running") from None
            cells = collect_cells(out, seed, smoke, router, pairs, protocol, device)
            pending = root / "effects.csv.tmp"
            downstream.write_csv(pending, crossover_summary(cells))
            pending.replace(root / "effects.csv")
        print(f"Verified crossover results: {root / 'effects.csv'}", flush=True)


def summary(run, out, seeds):
    router = downstream_models.DownstreamModels(run)
    pairs, reference_sha = downstream.audited_pairs(run, out)
    protocol = score_settings(out, reference_sha, smoke=False)
    reports = []
    for seed in seeds:
        cells = collect_cells(out, seed, False, router, pairs, protocol)
        keys = ("dataset", "lang", "subset", "metric")
        reports.append({tuple(row[key] for key in keys): row for row in crossover_summary(cells)})
    print(f"Crossover on seeds {', '.join(map(str, seeds))}. Scores are percentages, effects are percentage points.")
    print("Panel                                  Metric      G1 BPE   G1 G3  G4 BPE   G4 G3"
          "    G3−BPE @G1   G3−BPE @G4   G4−G1 @BPE   G4−G1 @G3  Interaction")
    for (dataset, lang, subset, metric) in reports[0]:
        records = [report[dataset, lang, subset, metric] for report in reports]
        scores = [mean(row[key] for row in records)
                  for key in ("g1_bpe_pct", "g1_g3_pct", "g4_bpe_pct", "g4_g3_pct")]
        effects = []
        for key in EFFECTS:
            values = [row[key] for row in records]
            spread = f"±{stdev(values):.2f}" if len(values) > 1 else ""
            effects.append(f"{mean(values):+.2f}{spread}")
        panel = f"{dataset}/{lang}/{subset}"
        print(f"{panel:38} {metric:6} " + " ".join(f"{x:7.2f}" for x in scores)
              + " " + " ".join(f"{effect:>12}" for effect in effects))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("smoke", "run", "summary"))
    parser.add_argument("--run", type=Path, default=Path("runs/pilot"))
    parser.add_argument("--out", type=Path, help="existing downstream dataset and results directory")
    parser.add_argument("--seed", type=int, nargs="+", choices=SEEDS)
    args = parser.parse_args()
    out = args.out or args.run / "downstream"
    seeds = list(dict.fromkeys(args.seed or (42,)))
    if args.action == "summary":
        summary(args.run, out, seeds)
    else:
        run_crossover(args.run, out, seeds, smoke=args.action == "smoke")


if __name__ == "__main__":
    main()
