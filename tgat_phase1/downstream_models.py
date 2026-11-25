"""Resolve frozen Phase 1 model arms and route benchmark text to their tokenizers."""

import csv
from copy import deepcopy
from functools import cached_property
from importlib.metadata import version
from itertools import product
import math
from pathlib import Path

from .boundaries import boundary_hints, training_stems
from .data import LANGUAGES, MODEL, MODEL_REVISION, canonical_text, load_split, read_json, sha256_file
from .embedding_ablation import CONTROL, FULL_GROUP, GROUPS as SELECTIVE, SEEDS, read_plan, read_scores, verified_group
from .embedding_followup import ARMS, provenance as followup_provenance
from .g3 import GROUPS as BOUNDARY_GROUPS, SPLITS, load_selection, rescored_ids
from .g3_lora import check_budget, saved_model, verified_control
from .rescoring import load_segmenters
from .tokenization import original_tokenizer, verified_tokenizers
from .train import load_model


_NAMES = {
    "G0": "Qwen-base", "G1": "G1-original", "G2": "G2-langmap", "G3": CONTROL,
    "G4": "G4-drift", "G5": FULL_GROUP, "G6": "G6-random", "frequency-matched": "frequency-matched",
}
GROUPS = tuple(_NAMES)
_ADAPTER_CONFIG = "adapter/adapter_config.json"
_ADAPTER_WEIGHTS = "adapter/adapter_model.safetensors"


def _require(actual, expected, label):
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(f"{label} differs at {key}")


def _adapter_digests(directory):
    _require(read_json(directory / _ADAPTER_CONFIG),
             {"base_model_name_or_path": MODEL, "task_type": "CAUSAL_LM", "peft_type": "LORA"},
             f"{directory} adapter")
    weights = directory / _ADAPTER_WEIGHTS
    if weights.stat().st_size == 0:
        raise ValueError(f"Empty LoRA checkpoint: {weights}")
    return {name: sha256_file(directory / name) for name in (_ADAPTER_CONFIG, _ADAPTER_WEIGHTS)}


def _route(group):
    if group not in _NAMES:
        raise ValueError(f"Unknown downstream group: {group}")
    if group in ("G0", "G1"):
        return "original-bpe"
    return "native-langmap" if group == "G2" else "g3-boundary"


class DownstreamModels:
    """Verify checkpoints on the host and load one model at a time on the scorer's device."""

    def __init__(self, run, *, device=None, dtype=None, boundary=None, embedding=None, full=None):
        self.run = Path(run)
        g3_eval = self.run / "g3-eval"
        self.boundary = Path(boundary) if boundary is not None else g3_eval / "boundary-pilot"
        self.embedding = Path(embedding) if embedding is not None else g3_eval / "embedding-ablation"
        self.full = Path(full) if full is not None else self.embedding / "follow-up"
        self.device, self.dtype = device, dtype
        self._seed_runs = {}
        self._g3_runs = {}
        self._artifacts = {}

    @cached_property
    def _pilot(self):
        pilot = read_json(self.run / "seed-42/run.json")
        _require(pilot, {
            "model": MODEL, "model_revision": MODEL_REVISION, "epochs": 1,
            "corpus_manifest_sha256": sha256_file(self.run / "corpus/manifest.json"),
            "langmap_manifest_sha256": sha256_file(self.run / "langmap/manifest.json"),
        }, "Pilot configuration")
        return pilot

    @property
    def eos_id(self):
        return self._pilot["eos_id"]

    @cached_property
    def _selection(self):
        return load_selection(self.run, self.run / "g3-eval/stem-reference", self.boundary)

    @cached_property
    def _plan(self):
        return read_plan(self.embedding, self.boundary, self._pilot)

    @cached_property
    def _followup_plan(self):
        plan = read_json(self.full / "plan.json")
        _require(plan, followup_provenance(self.embedding), "Embedding follow-up plan")
        return plan

    def _pilot_seed(self, seed):
        if seed not in self._seed_runs:
            pilot, _, control = verified_control(self.run, seed, self._selection)
            self._seed_runs[seed] = pilot, control
        return self._seed_runs[seed]

    def _g3_run(self, seed):
        if seed not in self._g3_runs:
            pilot, g2 = self._pilot_seed(seed)
            directory = self.boundary / f"lora-seed-{seed}"
            config = read_json(directory / "run.json")
            _require(config, {
                **pilot,
                "selected_lambda": self._selection["selected_lambda"],
                "length_weight": self._selection["length_weight"],
                "selection_sha256": sha256_file(self.boundary / "run.json"),
                "base_bpb_sha256": sha256_file(self.boundary / "base-bpb/bpb.csv"),
                "g2_adapter_sha256": sha256_file(g2 / _ADAPTER_WEIGHTS),
                "g2_adapter_config_sha256": sha256_file(g2 / _ADAPTER_CONFIG),
            }, directory)
            check_budget(read_json(directory / CONTROL / "training.json"), config["training_budgets"][CONTROL], CONTROL)
            self._g3_runs[seed] = directory, config, self._g3_scores(directory / "bpb.csv")
        return self._g3_runs[seed]

    @staticmethod
    def _g3_scores(path):
        expected = set(product(("before_lora", "after_lora"), SPLITS, BOUNDARY_GROUPS, (*LANGUAGES, "all")))
        scores = {}
        with path.open(newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                key = row["stage"], row["split"], row["group"], row["lang"]
                size, tokens = int(row["bytes"]), int(row["tokens"])
                nll, bpb = float(row["nll_nats"]), float(row["bpb"])
                if (key in scores or size <= 0 or tokens <= 0 or not math.isfinite(nll) or nll <= 0
                        or not math.isclose(bpb, nll / (size * math.log(2)), rel_tol=1e-9)):
                    raise ValueError(f"Invalid G3 score in {path}: {key}")
                scores[key] = size, tokens, nll, bpb
        if scores.keys() != expected:
            raise ValueError(f"Incomplete G3 scores: {path}")
        return {(split, lang): scores["after_lora", split, CONTROL, lang]
                for split, lang in product(SPLITS, (*LANGUAGES, "all"))}

    def _embedding_artifact(self, group, seed):
        name = _NAMES[group]
        g3, g3_config, g3_scores = self._g3_run(seed)
        budget = self._plan["training_budget"]
        if budget != g3_config["training_budgets"][CONTROL]:
            raise ValueError("Embedding plan and G3 use different training budgets")
        plan_sha = sha256_file(self.embedding / "run.json")
        required = {**self._pilot, "seed": seed, "smoke": False, "planned_training_budget": budget,
                    "g3_bpb_sha256": sha256_file(g3 / "bpb.csv")}
        if name in SELECTIVE:
            directory = self.embedding / f"seed-{seed}"
            groups = (CONTROL, *SELECTIVE)
            draw = self._plan["draws"][str(seed)]
            required.update(plan_sha256=plan_sha, row_ids={g: draw[k] for g, k in SELECTIVE.items()})
            ids = required["row_ids"][name]
        else:
            arm = "full" if group == "G5" else "matched"
            directory = self.full / arm / f"seed-{seed}"
            groups = (CONTROL, name)
            key = ARMS[arm][1]
            draw = self._followup_plan["draws"][str(seed)]
            ids = draw[key] if key is not None else None
            required.update(group=name, base_plan_sha256=plan_sha, row_ids=ids,
                            followup_plan_sha256=sha256_file(self.full / "plan.json"))
        _require(read_json(directory / "run.json"), required, directory)
        scores = read_scores(directory / "bpb.csv", groups, SPLITS)
        for (split, lang), expected in g3_scores.items():
            row = scores[split, CONTROL, lang]
            if tuple(row[k] for k in ("bytes", "tokens", "nll_nats", "bpb")) != expected:
                raise ValueError(f"{directory} contains a different G3 reference at {split}/{lang}")
        stats = verified_group(directory / name, budget, smoke=False)
        return directory, ids, stats["artifacts_sha256"]

    def _resolve(self, group, seed):
        route = _route(group)
        if group != "G0" and seed not in SEEDS:
            raise ValueError(f"Unsupported downstream pilot seed: {seed}")
        key = group, None if group == "G0" else seed
        if key in self._artifacts:
            return self._artifacts[key]

        identity = {"group": group, "seed": key[1], "model": MODEL, "model_revision": MODEL_REVISION,
                    "tokenizer": route, "eos_id": self.eos_id}
        if route != "original-bpe":
            identity["langmap_manifest_sha256"] = self._pilot["langmap_manifest_sha256"]
        if route == "g3-boundary":
            identity["g3_selection_sha256"] = sha256_file(self.boundary / "run.json")
        directory, ids = None, None
        if group in ("G1", "G2"):
            _, g2 = self._pilot_seed(seed)
            directory = g2.parent
            digests = _adapter_digests(directory / _NAMES[group])
            if group == "G2":
                g3_config = read_json(self.boundary / f"lora-seed-{seed}/run.json")
                _require(g3_config, {"g2_adapter_sha256": digests[_ADAPTER_WEIGHTS],
                                     "g2_adapter_config_sha256": digests[_ADAPTER_CONFIG]}, "Recorded G2 adapter")
        elif group == "G3":
            directory, _, _ = self._g3_run(seed)
            digests = _adapter_digests(directory / CONTROL)
        elif group != "G0":
            directory, ids, digests = self._embedding_artifact(group, seed)
        adapter = None if directory is None else directory / _NAMES[group]
        if adapter is not None:
            identity.update(artifacts_sha256=digests, training_run_sha256=sha256_file(directory / "run.json"),
                            training_stats_sha256=sha256_file(adapter / "training.json"),
                            training_bpb_sha256=sha256_file(directory / "bpb.csv"))
        self._artifacts[key] = adapter, ids, identity
        return self._artifacts[key]

    def provenance(self, group, seed=42):
        """Return tokenization and artifact identity without importing PyTorch."""
        return deepcopy(self._resolve(group, seed)[2])

    def _check_packages(self, packages):
        pinned = self._pilot["packages"]
        _require({name: version(name) for name in packages}, {name: pinned[name] for name in packages},
                 "Downstream runtime")

    @cached_property
    def _original(self):
        self._check_packages(("transformers", "tokenizers"))
        original = original_tokenizer()
        if (original.eos_token_id != self.eos_id
                or len(original.backend_tokenizer.get_vocab()) != self._pilot["tokenizer_vocab_entries"]):
            raise ValueError("Original tokenizer does not match the frozen model")
        return original

    @cached_property
    def _native(self):
        return verified_tokenizers(self.run, self._original)

    @cached_property
    def _boundary_encoder(self):
        self._check_packages(("transformers", "tokenizers"))
        train = load_split(self.run / "corpus", "train.jsonl")
        return load_segmenters(self.run), training_stems(train), self._selection["selected_lambda"]

    def encode(self, text, lang, group):
        """Encode nonempty text in its supplied language without adding control tokens."""
        route = _route(group)
        text = canonical_text(text)
        if lang not in LANGUAGES or not text:
            raise ValueError("Downstream encoding requires nonempty text and an English, Japanese or Turkish label")
        if route == "g3-boundary":
            segmenters, stems, weight = self._boundary_encoder
            return list(rescored_ids({"text": text, "lang": lang}, boundary_hints(text, lang, stems), weight, segmenters))
        tokenizer = self._original.backend_tokenizer if route == "original-bpe" else self._native[lang]
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        if not ids or tokenizer.decode(ids, skip_special_tokens=False) != text:
            raise ValueError(f"Downstream tokenizer failed to round-trip {lang} text")
        return ids

    def load(self, group, seed=42):
        """Load one frozen evaluation model with its adapter and optional embedding state."""
        adapter, ids, _ = self._resolve(group, seed)
        if self.device is None or self.dtype is None:
            raise ValueError("Model loading requires an evaluation device and dtype")
        self._check_packages(self._pilot["packages"])
        model = load_model(self.device, self.dtype)
        if model.config.eos_token_id != self.eos_id:
            raise ValueError("Checkpoint EOS differs from the frozen pilot")
        if adapter is not None:
            model = saved_model(model, adapter)
            if group == "G5":
                from .full_embedding import load_full_embedding

                load_full_embedding(model, adapter / "embedding.safetensors")
            elif ids is not None:
                from .tied_rows import load_tied_rows

                load_tied_rows(model, adapter / "rows.safetensors", ids)
        return model.requires_grad_(False).eval()
