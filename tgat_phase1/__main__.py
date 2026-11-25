import argparse
from pathlib import Path

from .data import prepare, prepare_g3


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def main():
    parser = argparse.ArgumentParser(prog="python -m tgat_phase1")
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("prepare", help="freeze and split the shared corpus")
    data.add_argument("--run", type=Path, required=True)
    data.add_argument("--train-docs", type=positive_int, default=16, help="per language")
    data.add_argument("--eval-docs", type=positive_int, default=4, help="per language")
    data.add_argument("--split-seed", type=int, default=2026)
    data.add_argument("--local-data", type=Path)
    g3 = commands.add_parser("prepare-g3", help="freeze fresh G3 validation and test documents")
    g3.add_argument("--run", type=Path, default=Path("runs/pilot"))
    g3.add_argument("--validation-docs", type=positive_int, default=64, help="per language")
    g3.add_argument("--test-docs", type=positive_int, default=64, help="per language")
    g3.add_argument("--split-seed", type=int, default=2027)

    fit = commands.add_parser("fit", help="fit the shared and per-language Unigram tokenizers")
    fit.add_argument("--run", type=Path, required=True)
    fit.add_argument("--fit-docs", type=positive_int, help="optional cap per language")
    fit.add_argument("--em-iterations", type=positive_int, default=10)

    train = commands.add_parser("compare", help="run the Original/LangMAP LoRA comparison")
    train.add_argument("--run", type=Path, required=True)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--epochs", type=positive_int, default=1)
    train.add_argument("--context", type=positive_int, default=512)
    train.add_argument("--grad-accum", type=positive_int, default=8)
    train.add_argument("--learning-rate", type=float, default=2e-4)
    train.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    train.add_argument("--no-gradient-checkpointing", action="store_true")
    train.add_argument("--pooled-control", action="store_true",
                       help="train only the pooled Unigram control against the corresponding baseline seed")
    train.add_argument("--output-root", type=Path, help="write results separately from the frozen inputs")
    args = parser.parse_args()
    args.run = args.run.resolve()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "prepare-g3":
        prepare_g3(args)
    elif args.command == "fit":
        from .tokenization import fit as fit_langmap
        fit_langmap(args)
    else:
        from .train import compare
        if args.context < 2 or args.learning_rate <= 0:
            parser.error("compare requires --context >= 2 and --learning-rate > 0")
        compare(args)


if __name__ == "__main__":
    main()
