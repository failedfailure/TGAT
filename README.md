# TGAT-Adapt

TGAT-Adapt studies fixed-vocabulary retokenization and selective embedding adaptation in Qwen3-0.6B-Base for English, Japanese, and Turkish. The methods preserve pretrained token IDs and compare original BPE, LangMAP, a morphology-guided boundary reward, and different ways to update tied input and output embeddings.

## Quick start

Use Linux or WSL 2 with Bash, Python 3.11 or newer, Docker Compose v2, and an NVIDIA GPU exposed to Docker. On WSL, clone into the Linux filesystem and enable Docker's WSL integration. The container builds the pinned LangMAP tokenizer forks and uses PyTorch 2.9.1 with CUDA 12.8. The experiments were run on an RTX 4070 Ti. Building the image and preparing data require access to the upstream package, model, and dataset hosts.

From the repository root:

```bash
bash run.sh build
bash run.sh check
bash run.sh smoke
```

The smoke run downloads the base model, streams a small three-language sample, fits two tokenizers, and trains and scores two LoRA adapters. It writes to `runs/smoke/`. The normal `pilot` command writes to `runs/pilot/` and resolves the current FineWeb and FineWeb2 revisions. Use the pinned procedure below to reconstruct the archived corpus.

```bash
python3 -m unittest discover -s tests -v
# After building the image, run with its pinned Python dependencies
docker compose run --rm tgat python -m unittest discover -s tests -v
```

The host test command skips tests whose optional ML dependencies are unavailable. Run `bash run.sh --help` to see individual stages.

## Reproduce the experiments

### 1. Prepare the pinned corpus and train the baselines

The [corpus manifest](reproduction/paper-corpus-manifest.json) records the exact FineWeb and FineWeb2 revisions, split seed, and expected SHA-256 digests. The preparation helper streams those versions and checks the outputs before publishing them to the local run directory. Use a fresh `runs/pilot/` or a directory that contains only the matching paper corpus. Build the image before running these steps.

```bash
TGAT_UID="$(id -u)" TGAT_GID="$(id -g)" docker compose run --rm tgat \
    python -m reproduction.prepare_corpus
bash run.sh fit --run runs/pilot --fit-docs 1000 --em-iterations 10
bash run.sh compare --run runs/pilot --seed 42 --context 512 --grad-accum 8
bash run.sh batch
bash run.sh summary

bash run.sh pooled-batch
bash run.sh pooled-summary
bash run.sh batch --epochs 2
bash run.sh summary --epochs 2
```

G1 uses original BPE and G2 uses language-conditioned LangMAP with frozen embeddings. The pooled control uses one shared Unigram fit. The two-epoch comparison starts from the base checkpoint. `batch` reuses the completed seed-42 inputs and processes the remaining seeds.

### 2. Rebuild the morphology reference and score G3

Prepare the second FineWeb split first. The six Universal Dependencies files below are checked against the Git blob hashes in [`g3_reference.py`](tgat_phase1/g3_reference.py).

```bash
bash run.sh prepare-g3
mkdir -p data/ud-r2.16
for treebank in UD_English-EWT/en_ewt UD_Japanese-GSDLUW/ja_gsdluw UD_Turkish-Kenet/tr_kenet; do
    repo="${treebank%/*}"
    prefix="${treebank#*/}"
    for split in dev test; do
        filename="${prefix}-ud-${split}.conllu"
        curl -fLsS "https://raw.githubusercontent.com/UniversalDependencies/${repo}/r2.16/${filename}" \
            -o "data/ud-r2.16/${filename}" || exit 1
    done
done
python3 -m tgat_phase1.g3_reference data/ud-r2.16 runs/pilot/g3-eval/stem-reference
bash run.sh g3-screen
bash run.sh g3-bpb
for seed in 42 43 44; do
    bash run.sh g3-lora --seed "$seed" || exit 1
done
```

G3 selects the boundary reward on UD dev and reports stem-boundary alignment on UD test. Its likelihood runs use the separate FineWeb validation and test documents. G3 trains attention LoRA with the tied embedding table frozen.

### 3. Run the embedding controls

```bash
bash run.sh embedding-plan
bash run.sh embedding-run --seed 42 43 44
bash run.sh embedding-summary --seed 42 43 44

bash run.sh embedding-followup-plan
bash run.sh embedding-followup-run --arm matched full --seed 42
bash run.sh embedding-followup-summary --arm matched full --seed 42
```

G4 ranks rows by the change in token usage between BPE and G3. G6 samples from the same eligible pool. The frequency-matched arm preserves row counts by G3 frequency band and dominant training language. G5 updates the full tied table.

After those runs, score the saved G1 adapters on the same FineWeb test documents as G2 and G4. Then train with native LangMAP and the fixed 512 row IDs from G4 to measure the boundary reward with the trainable row set held fixed.

```bash
TGAT_UID="$(id -u)" TGAT_GID="$(id -g)" docker compose run --rm tgat-gpu \
    env HF_HUB_OFFLINE=1 python -m reproduction.score_fineweb_g1
TGAT_UID="$(id -u)" TGAT_GID="$(id -g)" docker compose run --rm tgat-gpu \
    env HF_HUB_OFFLINE=1 python -m reproduction.g2_same_rows --smoke-first --seed 42 43 44
python3 -m reproduction.g2_same_rows --summary --seed 42 43 44
```

The commands write paired BPB and run provenance to `runs/pilot/g3-eval/review-bpe/` and `runs/pilot/g3-eval/review-g2-same-rows/`. Each module accepts `--out` to select a different output directory.

### 4. Score minimal pairs and FLORES-200

```bash
bash run.sh downstream-prepare
bash run.sh downstream-run --seed 42
bash run.sh downstream-run --groups G1 G3 G4 G6 --seed 43 44
bash run.sh downstream-summary --groups G1 G3 G4 G6 --seed 42 43 44 --baseline G1
bash run.sh downstream-crossover-run --seed 42 43 44
bash run.sh downstream-crossover-summary --seed 42 43 44

bash run.sh routing-quality-prepare
bash run.sh routing-quality-run --seed 42
bash run.sh routing-quality-summary --seed 42
bash run.sh hybrid-training-plan
bash run.sh hybrid-training-run --seed 42
bash run.sh hybrid-training-summary --seed 42
```

The downstream preparation stage checks the pinned BLiMP, JBLiMP, and MultiBLiMP sources. FLORES-200 preparation verifies the original Meta archive and accepts `--archive PATH` for a previously downloaded copy. The fixed mixed route uses BPE for English and Japanese and the G3 tokenizer for Turkish. The mixed-training experiment scores both G4 checkpoints with that route.

### 5. Read the outputs

Generated corpora, manifests, weights, scores, and logs are stored in `runs/`. The model cache is `.hf-cache/`. The summary commands verify local run artifacts. BPB reports bits per UTF-8 byte under the scored token path. Aggregated BPB weights languages by their byte counts. Across-seed differences are paired by seed and reported as mean plus or minus sample standard deviation. Dataset regeneration verifies corpus hashes. Training results can vary with software and hardware.

## Repository layout

| Path | Contents |
| --- | --- |
| `tgat_phase1/` | Dataset preparation, tokenization, adaptation, and evaluation |
| `tests/` | Unit tests for segmentation, model updates, and evaluation |
| `run.sh`, `Dockerfile`, `compose.yaml`, `requirements.txt` | Experiment entry point and container setup |
| `reproduction/` | Pinned corpus preparation and supplementary comparisons |

## Upstream sources

The experiments use pinned revisions of [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base/tree/da87bfb608c14b7cf20ba1ce41287e8de496c0cd) and [LangMAP](https://github.com/cimeister/unimixlm/tree/ec6d1db61f5c1ed374717b1c88af2db54235655d).

- Training and language modeling evaluation use [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) and [FineWeb2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2).
- Stem-boundary annotations are derived from [Universal Dependencies 2.16](https://hdl.handle.net/11234/1-5901).
- Minimal-pair evaluation uses [BLiMP](https://aclanthology.org/2020.tacl-1.25/), [JBLiMP](https://aclanthology.org/2023.findings-eacl.117/), and [MultiBLiMP](https://aclanthology.org/2026.tacl-1.10/).
- Known-language routing uses the [original FLORES-200 release](https://github.com/facebookresearch/flores/blob/a6c830c6e1051fb4ac1a44b32358f00463f332bd/flores200/README.md).

Models and datasets are obtained from their respective sources and are subject to their upstream terms.
