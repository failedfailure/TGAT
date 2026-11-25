"""Matched-document causal LM training and held-out BPB evaluation."""

import csv
import math
import random
from collections import Counter
from importlib.metadata import version
from itertools import islice
from time import perf_counter

from .data import LANGUAGES, MODEL, MODEL_REVISION, load_split, read_json, sha256_file, split_records, write_json
from .tokenization import encode_documents, original_tokenizer, verified_tokenizers


def tied_weights(model):
    return model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()


def load_model(device, dtype):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=MODEL_REVISION, dtype=dtype, attn_implementation="sdpa",
    ).to(device)
    model.config.use_cache = False
    if not model.config.tie_word_embeddings or not tied_weights(model):
        raise RuntimeError("Expected Qwen input embeddings and LM head to share weights")
    return model


def sequence_loss(model, ids, device):
    import torch

    seq = torch.tensor([ids], dtype=torch.long, device=device)
    return model(input_ids=seq, labels=seq, use_cache=False).loss


def evaluate(model, docs, encoded, eos, context, device):
    import torch

    model.eval()
    num_bytes = Counter()
    token_count = Counter()
    nll = Counter()
    for row in docs:
        num_bytes[row["lang"]] += len(row["text"].encode("utf-8"))
    with torch.inference_mode():
        for lang, ids in split_records(encoded, eos, context):
            targets = len(ids) - 1
            value = float(sequence_loss(model, ids, device))
            if not math.isfinite(value):
                raise RuntimeError(f"Non-finite loss in {lang} evaluation")
            nll[lang] += value * targets
            token_count[lang] += targets
    nll["all"] = sum(nll.values())
    num_bytes["all"] = sum(num_bytes.values())
    token_count["all"] = sum(token_count.values())
    return [
        {"lang": lang, "bytes": num_bytes[lang], "tokens": token_count[lang],
         "nll_nats": nll[lang], "bpb": nll[lang] / (math.log(2) * num_bytes[lang])}
        for lang in (*LANGUAGES, "all")
    ]


def train_lora(model, encoded, eos, args, device, *, prepare_trainable=None, max_steps=None):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model

    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none",
    ))
    if (not tied_weights(model) or model.get_input_embeddings().weight.requires_grad
            or model.get_output_embeddings().weight.requires_grad):
        raise RuntimeError("LoRA requires frozen, tied base embedding weights")
    if prepare_trainable is not None:
        prepare_trainable(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    model.train()
    processed, step, loss_sum = 0, 0, 0.0
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    epoch_rng = random.Random(args.seed)
    for epoch in range(args.epochs):
        order = list(range(len(encoded)))
        epoch_rng.shuffle(order)
        windows = split_records((encoded[i] for i in order), eos, args.context)
        while (max_steps is None or step < max_steps) and (batch := list(islice(windows, args.grad_accum))):
            optimizer.zero_grad(set_to_none=True)
            count = sum(len(ids) - 1 for _, ids in batch)
            step_loss = torch.zeros((), device=device)
            for _, ids in batch:
                targets = len(ids) - 1
                loss = sequence_loss(model, ids, device)
                (loss * (targets / count)).backward()
                step_loss += loss.detach() * targets
            loss_value = step_loss.item()
            if not math.isfinite(loss_value):
                raise RuntimeError(f"Non-finite training loss at step {step + 1}")
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()
            loss_sum += loss_value
            processed += count
            step += 1
            if step == 1 or step % 10 == 0:
                print(f"epoch={epoch + 1} step={step} tokens={processed} "
                      f"loss={loss_value / count:.4f}", flush=True)
        if max_steps is not None and step >= max_steps:
            break
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = perf_counter() - start
    return model, {
        "optimizer_steps": step,
        "train_tokens": processed,
        "trainable_parameters": sum(p.numel() for p in trainable),
        "train_seconds": elapsed,
        "train_tokens_per_second": processed / elapsed,
        "train_mean_nll": loss_sum / processed,
        "train_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated() if device.type == "cuda" else None
        ),
        "train_peak_reserved_bytes": (
            torch.cuda.max_memory_reserved() if device.type == "cuda" else None
        ),
    }


def hardware(args):
    import torch

    if args.device == "cpu":
        return torch.device("cpu"), torch.float32
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside the container. Run with GPU access, "
                           "or pass --device cpu to verify a small run.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This configuration requires native BF16 support for CUDA.")
    return torch.device("cuda:0"), torch.bfloat16


def compare(args):
    import torch
    from transformers import set_seed

    corpus = args.run / "corpus"
    train = load_split(corpus, "train.jsonl")
    heldout = load_split(corpus, "heldout.jsonl")
    original = original_tokenizer()
    fitted = verified_tokenizers(args.run, original)
    eos = original.eos_token_id
    vocab = original.backend_tokenizer.get_vocab()
    device, dtype = hardware(args)
    manifest = {
        "model": MODEL, "model_revision": MODEL_REVISION,
        "seed": args.seed, "epochs": args.epochs, "context": args.context,
        "microbatch": 1, "grad_accum": args.grad_accum, "learning_rate": args.learning_rate,
        "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
        "cuda": torch.version.cuda, "dtype": str(dtype),
        "gradient_checkpointing": not args.no_gradient_checkpointing,
        "packages": {p: version(p) for p in ("torch", "transformers", "peft", "tokenizers")},
        "corpus_manifest_sha256": sha256_file(corpus / "manifest.json"),
        "langmap_manifest_sha256": sha256_file(args.run / "langmap" / "manifest.json"),
        "train_documents": len(train),
        "train_bytes": sum(len(row["text"].encode("utf-8")) for row in train),
        "tokenizer_vocab_entries": len(vocab), "eos_id": eos,
    }
    groups = (("G1-original", original.backend_tokenizer), ("G2-langmap", fitted))
    reference = None
    if args.pooled_control:
        from .series import inspect_seed

        if read_json(args.run / f"seed-{args.seed}/run.json") != manifest:
            raise ValueError("The pooled control must match the baseline configuration for this seed")
        heldout_bytes = Counter()
        for row in heldout:
            heldout_bytes[row["lang"]] += len(row["text"].encode("utf-8"))
        heldout_bytes["all"] = sum(heldout_bytes.values())
        status, detail, reference = inspect_seed(args.run, args.seed, manifest, heldout_bytes)
        if status != "verified":
            raise ValueError(f"The pooled control requires a verified seed {args.seed}: {detail}")
        groups = (("pooled-unigram", fitted["base"]),)
    root = args.output_root or args.run
    root.mkdir(parents=True, exist_ok=True)
    out = root / (f"pooled-seed-{args.seed}" if args.pooled_control else f"seed-{args.seed}")
    out.mkdir(exist_ok=False)
    write_json(out / "run.json", manifest)

    rows = []
    for group, tokenizers in groups:
        print(f"Running {group}", flush=True)
        training = list(encode_documents(train, tokenizers))
        evaluation = list(encode_documents(heldout, tokenizers))
        set_seed(args.seed)
        base = load_model(device, dtype)
        if base.config.eos_token_id != eos or max(vocab.values()) >= base.config.vocab_size:
            raise RuntimeError("Original tokenizer IDs do not match the Qwen model configuration")
        baseline = evaluate(base, heldout, evaluation, eos, args.context, device)
        rows.extend({"group": group, "stage": "before_lora", **row} for row in baseline)
        set_seed(args.seed)
        model, stats = train_lora(base, training, eos, args, device)
        after = evaluate(model, heldout, evaluation, eos, args.context, device)
        rows.extend({"group": group, "stage": "after_lora", **row} for row in after)
        group_dir = out / group
        group_dir.mkdir()
        model.save_pretrained(group_dir / "adapter", save_embedding_layers=False)
        write_json(group_dir / "training.json", stats)
        write_json(group_dir / "heldout.json", {"before_lora": baseline, "after_lora": after})
        del model, base
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pending = out / "bpb.csv.tmp"
    with pending.open("w", encoding="utf-8", newline="") as stream:
        csv_writer = csv.DictWriter(
            stream, fieldnames=["group", "stage", "lang", "bytes", "tokens", "nll_nats", "bpb"],
        )
        csv_writer.writeheader()
        csv_writer.writerows(rows)
    pending.replace(out / "bpb.csv")
    print(f"Results: {out / 'bpb.csv'}")
    if reference is not None:
        print("Language   Stage         Original    Pooled   LangMAP  Pooled-Orig  LangMAP-Pooled")
        for row in rows:
            lang, stage, pooled = row["lang"], row["stage"], row["bpb"]
            original_bpb = reference["G1-original", stage, lang]
            langmap_bpb = reference["G2-langmap", stage, lang]
            print(f"{lang:<10} {stage:<12} {original_bpb:>8.4f} {pooled:>9.4f} {langmap_bpb:>9.4f}"
                  f" {pooled-original_bpb:>+12.4f} {langmap_bpb-pooled:>+15.4f}")
