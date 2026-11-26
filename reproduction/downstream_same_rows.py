"""Score the two matched 512-row checkpoints on frozen grammatical minimal pairs."""

import argparse
import fcntl
from pathlib import Path
from tempfile import TemporaryDirectory

from tgat_phase1 import downstream, downstream_models
from tgat_phase1.data import read_json, sha256_file, write_json
from tgat_phase1.downstream_crossover import BATCH_SIZE, checkpoint_settings, score_settings
from tgat_phase1.embedding_ablation import SEEDS, verified_group, verified_runtime
from tgat_phase1.g3_lora import check_budget

from .g2_same_rows import GROUP


class SameRowsModels(downstream_models.DownstreamModels):
    def __init__(self, run, same_rows_root, **kwargs):
        super().__init__(run, **kwargs)
        self.same_rows_root = same_rows_root

    def _resolve(self, group, seed):
        if group != GROUP:
            return super()._resolve(group, seed)
        if (group, seed) not in self._artifacts:
            g4, ids, g4_identity = super()._resolve("G4", seed)
            pilot, g2 = self._pilot_seed(seed)
            g3, _, _ = self._g3_run(seed)
            g2_identity = self.provenance("G2", seed)
            directory = self.same_rows_root / f"seed-{seed}"
            run_file = directory / "run.json"
            manifest = read_json(run_file)
            if len(ids) != 512 or len(set(ids)) != 512:
                raise ValueError("The paired G4 checkpoint must use 512 distinct rows")
            expected = {
                "pilot": pilot,
                "unit": "one pass over the same raw training documents",
                "selection_sha256": g4_identity["g3_selection_sha256"],
                "row_plan_sha256": sha256_file(self.embedding / "run.json"),
                "reference_scores_sha256": {
                    "g2_g3": sha256_file(g3 / "bpb.csv"),
                    "g4_g6": g4_identity["training_bpb_sha256"],
                },
                "g4_weights_sha256": g4_identity["artifacts_sha256"],
                "row_ids": ids,
                "g3_budget": self._plan["training_budget"],
                "smoke": False,
            }
            for key, value in expected.items():
                if manifest.get(key) != value:
                    raise ValueError(f"{directory}: matched-row training manifest differs at {key}")
            g2_stats = read_json(g2 / "training.json")
            budget = {key: g2_stats[key] for key in ("train_tokens", "optimizer_steps")}
            check_budget(manifest["g2_budget"], budget, GROUP)
            checkpoint = directory / GROUP
            stats = verified_group(checkpoint, budget, smoke=False)
            completed = read_json(directory / "results/run.json")
            if completed != {**manifest, "training": stats}:
                raise ValueError(f"{directory}: completed run does not match the checkpoint")
            g4_stats = read_json(g4 / "training.json")
            for key in ("trainable_parameters", "embedding_parameters"):
                if stats[key] != g4_stats[key]:
                    raise ValueError(f"{directory}: matched checkpoints differ at {key}")
            identity = {
                **g2_identity,
                "group": GROUP,
                "row_ids": ids,
                "row_plan_sha256": expected["row_plan_sha256"],
                "matched_g4_artifacts_sha256": expected["g4_weights_sha256"],
                "artifacts_sha256": stats["artifacts_sha256"],
                "training_run_sha256": sha256_file(run_file),
                "training_stats_sha256": sha256_file(checkpoint / "training.json"),
                "training_bpb_sha256": sha256_file(directory / "results/bpb.csv"),
            }
            self._artifacts[group, seed] = checkpoint, ids, identity
        return self._artifacts[group, seed]

    def encode(self, text, lang, group):
        return super().encode(text, lang, "G2" if group == GROUP else group)


def score(directory, expected, model, router, pairs, device):
    print(f"Scoring {GROUP}, seed {expected['seed']}, {len(pairs)} pairs", flush=True)
    rows = downstream.score_pairs(model, router, pairs, GROUP, device, BATCH_SIZE)
    with TemporaryDirectory(prefix=".same-rows-", dir=directory.parent) as temporary:
        staged = Path(temporary) / "result"
        staged.mkdir()
        downstream.write_csv(staged / "pairs.csv", rows)
        downstream.write_csv(staged / "summary.csv", downstream.aggregate(rows))
        checksums = {name: sha256_file(staged / name) for name in ("pairs.csv", "summary.csv")}
        write_json(staged / "run.json", {**expected, "output_sha256": checksums})
        downstream.verified_scores(staged, pairs, expected)
        staged.rename(directory)


def run(args, out, seeds):
    smoke = args.action == "smoke"
    evaluate = args.action != "summary"
    same_rows_root = args.same_rows_root or args.run / "g3-eval/review-g2-same-rows"
    router = SameRowsModels(args.run, same_rows_root)
    if evaluate:
        import torch

        router.device, router.dtype = verified_runtime(read_json(args.run / "seed-42/run.json"))
    pairs, reference_sha = downstream.audited_pairs(args.run, out)
    if smoke:
        pairs = downstream.fixed_smoke(pairs)
    settings = score_settings(out, reference_sha, smoke)
    for seed in seeds:
        directory = downstream.result_path(out, GROUP, seed, smoke)
        baseline = downstream.result_path(out, "G4", seed, smoke)
        if evaluate and not baseline.exists():
            downstream.evaluate(args.run, out, ("G4",), (seed,), BATCH_SIZE, smoke)
        downstream.verified_scores(baseline, pairs, checkpoint_settings(router, "G4", seed, settings))
        expected = {
            **checkpoint_settings(router, GROUP, seed, settings),
            "same_rows_scorer_sha256": sha256_file(Path(__file__)),
        }
        if evaluate:
            directory.parent.mkdir(parents=True, exist_ok=True)
            with (directory.parent / f".{directory.name}.lock").open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise ValueError(f"{directory} is already running") from None
                if not directory.exists():
                    model = router.load(GROUP, seed)
                    score(directory, expected, model, router, pairs, router.device)
                    del model
                    torch.cuda.empty_cache()
                downstream.verified_scores(directory, pairs, expected)
                print(f"Verified benchmark results: {directory}", flush=True)
        else:
            downstream.verified_scores(directory, pairs, expected)
    if not evaluate:
        downstream.summarize(args.run, out, (GROUP, "G4"), seeds, baseline=GROUP)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("smoke", "run", "summary"))
    parser.add_argument("--run", type=Path, default=Path("runs/pilot"))
    parser.add_argument("--out", type=Path, help="existing frozen downstream dataset and results directory")
    parser.add_argument("--same-rows-root", type=Path, help="defaults to RUN/g3-eval/review-g2-same-rows")
    parser.add_argument("--seed", type=int, nargs="+", choices=SEEDS)
    args = parser.parse_args()
    run(args, args.out or args.run / "downstream", list(dict.fromkeys(args.seed or SEEDS)))


if __name__ == "__main__":
    main()
