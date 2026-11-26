#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export TGAT_UID="$(id -u)" TGAT_GID="$(id -g)"

cpu() {
    docker compose run --rm tgat "$@"
}

gpu() {
    docker compose run --rm tgat-gpu "$@"
}

prepare() {
    cpu python -m tgat_phase1 prepare "$@"
}

fit() {
    cpu env PYTHONPATH=/opt/unimixlm:/workspace python -m tgat_phase1 fit "$@"
}

compare() {
    gpu python -m tgat_phase1 compare "$@"
}

# Parse dispatch and exit together before any long-running command starts.
{
case "${1:-help}" in
    build)
        docker compose build tgat
        ;;
    check)
        gpu python -c 'import torch; assert torch.cuda.is_available(); x = torch.ones(16, device="cuda", dtype=torch.bfloat16); print(torch.cuda.get_device_name(0), x.sum().item(), "CUDA", torch.version.cuda)'
        ;;
    prepare-g3)
        shift
        cpu python -m tgat_phase1 prepare-g3 "$@"
        ;;
    g3-screen)
        shift
        python3 -m tgat_phase1.g3 screen "$@"
        ;;
    g3-parity)
        shift
        cpu python -m reproduction.check_zero_weight "$@"
        ;;
    g3-bpb)
        shift
        gpu python -m tgat_phase1.g3 bpb "$@"
        ;;
    g3-lora)
        shift
        gpu python -m tgat_phase1.g3_lora "$@"
        ;;
    fineweb-single-window)
        shift
        gpu env HF_HUB_OFFLINE=1 python -m reproduction.score_fineweb_single_window "$@"
        ;;
    embedding-plan)
        shift
        cpu env HF_HUB_OFFLINE=1 python -m tgat_phase1.embedding_ablation plan "$@"
        ;;
    embedding-smoke|embedding-run)
        action="${1#embedding-}"
        shift
        gpu env HF_HUB_OFFLINE=1 python -m tgat_phase1.embedding_ablation "$action" "$@"
        ;;
    embedding-summary)
        shift
        python3 -m tgat_phase1.embedding_ablation summary "$@"
        ;;
    embedding-followup-plan|embedding-followup-summary)
        action="${1#embedding-followup-}"
        shift
        python3 -m tgat_phase1.embedding_followup "$action" "$@"
        ;;
    embedding-followup-smoke|embedding-followup-run)
        action="${1#embedding-followup-}"
        shift
        gpu env HF_HUB_OFFLINE=1 python -m tgat_phase1.embedding_followup "$action" "$@"
        ;;
    downstream-prepare|downstream-summary)
        action="${1#downstream-}"
        shift
        python3 -m tgat_phase1.downstream "$action" "$@"
        ;;
    downstream-smoke|downstream-run)
        action="${1#downstream-}"
        shift
        gpu env HF_HUB_OFFLINE=1 python -m tgat_phase1.downstream "$action" "$@"
        ;;
    downstream-crossover-smoke|downstream-crossover-run)
        action="${1#downstream-crossover-}"
        shift
        gpu env HF_HUB_OFFLINE=1 python -m tgat_phase1.downstream_crossover "$action" "$@"
        ;;
    downstream-crossover-summary)
        shift
        python3 -m tgat_phase1.downstream_crossover summary "$@"
        ;;
    downstream-same-rows-smoke|downstream-same-rows-run)
        action="${1#downstream-same-rows-}"
        shift
        gpu env HF_HUB_OFFLINE=1 python -m reproduction.downstream_same_rows "$action" "$@"
        ;;
    downstream-same-rows-summary)
        shift
        python3 -m reproduction.downstream_same_rows summary "$@"
        ;;
    routing-quality-prepare|routing-quality-summary)
        action="${1#routing-quality-}"
        shift
        python3 -m tgat_phase1.routing_quality "$action" "$@"
        ;;
    routing-quality-smoke|routing-quality-run)
        action="${1#routing-quality-}"
        shift
        gpu env HF_HUB_OFFLINE=1 python -m tgat_phase1.routing_quality "$action" "$@"
        ;;
    hybrid-training-plan)
        shift
        cpu env HF_HUB_OFFLINE=1 python -m tgat_phase1.hybrid_training plan "$@"
        ;;
    hybrid-training-smoke|hybrid-training-run)
        action="${1#hybrid-training-}"
        shift
        gpu env HF_HUB_OFFLINE=1 python -m tgat_phase1.hybrid_training "$action" "$@"
        ;;
    hybrid-training-summary)
        shift
        python3 -m tgat_phase1.hybrid_training summary "$@"
        ;;
    smoke)
        run="${2:-runs/smoke}"
        prepare --run "$run" --train-docs 16 --eval-docs 4
        fit --run "$run" --em-iterations 2
        compare --run "$run" --context 256
        ;;
    pilot)
        run="${2:-runs/pilot}"
        prepare --run "$run" --train-docs 1500 --eval-docs 64
        fit --run "$run" --fit-docs 1000
        compare --run "$run" --context 512
        ;;
    batch|status|summary|pooled-batch|pooled-summary)
        python3 -m tgat_phase1.series "$@"
        ;;
    pooled)
        run="${2:-runs/pilot}"
        test -f "$run/seed-42/bpb.csv"
        mkdir -p "$run/logs"
        log=$(mktemp "$run/logs/pooled-seed-42-XXXXXXXX.log")
        printf 'Log: %s\n' "$log"
        compare --run "$run" --pooled-control 2>&1 | tee "$log"
        ;;
    prepare|fit|compare)
        command="$1"
        shift
        "$command" "$@"
        ;;
    help|-h|--help)
        cat <<'EOF'
Usage: bash run.sh COMMAND [ARGS]
  build                    Build the pinned LangMAP/PyTorch image
  check                    Verify a BF16 operation on the GPU
  smoke [runs/name]         Small three-language end-to-end run
  pilot [runs/name]         Single-seed G1/G2 pilot
  batch [--run PATH] [SEED ...]    Continue a completed seed-42 pilot
  status [--run PATH] [SEED ...]   Inspect artifacts without Docker
  summary [--run PATH] [SEED ...]  Report BPB across verified seeds
  pooled [runs/name]        Fit LoRA on the pooled Unigram control (seed 42)
  pooled-batch [--run PATH] [SEED ...]   Continue pooled controls for verified baseline seeds
  pooled-summary [--run PATH] [SEED ...]   Summarize paired BPB deltas across pooled seeds
  batch|status|summary --epochs N [SEED ...]   Use a separate G1/G2 duration arm (N >= 2)
  prepare-g3 [--run PATH]                 Freeze fresh validation and test documents
  g3-screen [--run PATH]                  Select a boundary weight and score stem boundaries on the host
  g3-parity [--run PATH]                  Compare zero-reward and native G2 token IDs on the pilot corpus
  g3-bpb [--run PATH]                     Verify native IDs and measure base BPB on the frozen G3 texts
  g3-lora [--run PATH] [--seed SEED]       Train one matched G3 adapter and score it against saved G2
  fineweb-single-window [--seed SEED ...] Score saved checkpoints on a common single-window FineWeb subset
  embedding-plan                          Freeze G4/G6 row IDs and G3 training paths
  embedding-smoke                         Check one optimizer step, validation scoring and adapter reload
  embedding-run [--seed SEED ...]          Train G4/G6 and compare with saved G3 on frozen splits
  embedding-summary [--seed SEED ...]      Summarize paired BPB against G3 and the random control
  embedding-followup-plan                 Freeze frequency controls using the existing G4/G6 plan
  embedding-followup-smoke [--arm ARM ...] Check two steps, gradients and reload on each selected arm
  embedding-followup-run [--arm ARM ...] [--seed SEED ...]  Run full-table or frequency-control arms
  embedding-followup-summary [--arm ARM ...] [--seed SEED ...]  Compare with saved matched G4 results
  downstream-prepare                      Freeze pinned minimal-pair benchmarks
  downstream-smoke [--groups GROUP ...]    Score a fixed two pairs per benchmark and language
  downstream-run [--groups GROUP ...] [--seed SEED ...]   Score all frozen pairs on saved checkpoints
  downstream-summary [--baseline GROUP] [--seed SEED ...]  Report accuracy and paired deltas
  downstream-crossover-smoke [--seed SEED ...]            Check G1/G4 with BPE and G3 paths on eight pairs
  downstream-crossover-run [--seed SEED ...]              Score the two crossed combinations on all frozen pairs
  downstream-crossover-summary [--seed SEED ...]          Summarize checkpoint and tokenizer effects
  downstream-same-rows-smoke [--seed SEED ...]            Check matched LangMAP and G4 rows on eight pairs
  downstream-same-rows-run [--seed SEED ...]              Score the saved 512-row LangMAP and G4 checkpoints
  downstream-same-rows-summary [--seed SEED ...]          Report paired G4 minus LangMAP accuracy
  routing-quality-prepare [--archive PATH]           Freeze aligned FLORES-200 text and audit exact overlaps
  routing-quality-smoke [--seed SEED ...]            Check G1/G4 scoring with both tokenizer routes
  routing-quality-run [--seed SEED ...]              Score frozen FLORES-200 and derive the fixed mixed route
  routing-quality-summary [--seed SEED ...]          Report language-specific and byte-weighted BPB
  hybrid-training-plan                               Freeze mixed training paths with the original G4 row set
  hybrid-training-smoke [--seed SEED ...]             Check two training steps, exact reload and FLORES scoring
  hybrid-training-run [--seed SEED ...]               Train on the mixed paths and score with matched inference
  hybrid-training-summary [--seed SEED ...]           Compare trained and inference-only hybrid BPB
  prepare --run PATH ...    Freeze the corpus
  fit     --run PATH ...    Fit LangMAP on the training split
  compare --run PATH ...    Train and score both LoRA groups
Add --help to an individual stage for its options.
EOF
        ;;
    *)
        printf 'Unknown command: %s\n' "$1" >&2
        exit 2
        ;;
esac
exit "$?"
}
