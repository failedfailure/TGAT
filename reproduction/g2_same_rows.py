"""Train the native LangMAP arm with the original G4 tied rows."""

import argparse
import csv
import fcntl
from collections import Counter
from copy import copy
from itertools import product
from math import isfinite
from pathlib import Path
from statistics import mean, stdev
from tempfile import TemporaryDirectory

from tgat_phase1.data import load_split, read_json, sha256_file, write_json
from tgat_phase1.embedding_ablation import (
    CONTROL, LANGS, SEEDS, fit_group, inputs, read_plan, read_scores, smoke_items,
    verified_group, verified_runtime,
)
from tgat_phase1.g3 import SPLITS, native_langmap, path_digest, publish
from tgat_phase1.g3_lora import budget, check_budget, language_totals, verified_control
from tgat_phase1.tokenization import encode_documents, verified_langmap_directory

G2, G4 = 'G2-langmap', 'G4-drift'
GROUP = 'G2-langmap-g4-rows'


def control_scores(path, group):
    with path.open(newline='', encoding='utf-8') as stream:
        return [row for row in csv.DictReader(stream)
                if row['stage'] == 'after_lora' and row['group'] == group]


def factorial_values(g2, g3, g2_rows, g4):
    return dict(g2=g2, g3=g3, g2_with_g4_rows=g2_rows, g4=g4,
                rows_under_g2=g2_rows-g2, rows_under_g3=g4-g3, boundary_frozen=g3-g2,
                boundary_with_rows=g4-g2_rows, interaction=(g4-g2_rows)-(g3-g2))


def summarize(rows, reference, evaluation, seed):
    scores = {(row['split'], row['group'], row['lang']): row for row in (*rows, *reference)}
    output = []
    for split in SPLITS:
        documents, paths = evaluation[split]
        byte_counts = language_totals(documents, (len(row['text'].encode('utf-8')) for row in documents))
        g2_count = language_totals(documents, map(len, paths[GROUP]))
        for lang in LANGS:
            cells = [scores[split, group, lang] for group in (G2, CONTROL, GROUP, G4)]
            if any(int(row['bytes']) != byte_counts[lang] for row in cells):
                raise ValueError(f'Mismatched corpus bytes in {split} {lang}')
            if any(int(cells[i]['tokens']) != g2_count[lang] for i in (0, 2)):
                raise ValueError(f'Mismatched native LangMAP tokens in {split} {lang}')
            values = factorial_values(*(float(row['bpb']) for row in cells))
            output.append(dict(seed=seed, split=split, lang=lang, **values))
    return output


def run_seed(args, boundary, selection, plan, train, encoded, evaluation, seed):
    pilot, _, g2 = verified_control(args.run, seed, selection)
    planned = budget([ids for _, ids in encoded], pilot['context'], pilot['grad_accum'])
    check_budget(read_json(g2 / 'training.json'), planned, G2)
    ablation = args.run / f'g3-eval/embedding-ablation/seed-{seed}'
    g4_run = read_json(ablation / 'run.json')
    g4 = verified_group(ablation / G4, plan['training_budget'], smoke=False)
    from safetensors import safe_open

    with safe_open(str(ablation / G4 / 'rows.safetensors'), framework='np') as weights:
        saved_ids = weights.get_tensor('ids').tolist()
    ids = plan['draws'][str(seed)]['selected_ids']
    if (len(ids) != 512 or len(set(ids)) != 512 or ids != g4_run['row_ids'][G4] or ids != saved_ids
            or len({tuple(draw['selected_ids']) for draw in plan['draws'].values()}) != 1):
        raise ValueError('Expected the same 512 tied rows as the original G4')
    g3_path = boundary / f'lora-seed-{seed}/bpb.csv'
    g4_path = ablation / 'bpb.csv'
    g3_run = read_json(g3_path.parent / 'run.json')
    if (g3_run['g2_adapter_sha256'] != sha256_file(g2 / 'adapter/adapter_model.safetensors')
            or g3_run['g2_adapter_config_sha256'] != sha256_file(g2 / 'adapter/adapter_config.json')):
        raise ValueError('G2 reference scores differ from the frozen adapter')
    if (g4_run['plan_sha256'] != sha256_file(ablation.parent / 'run.json')
            or g4_run['g3_bpb_sha256'] != sha256_file(g3_path)
            or any(g4_run.get(key) != value for key, value in pilot.items())):
        raise ValueError('G4 and G3 reference scores differ from the frozen inputs')
    counts = Counter(token for _, row_ids in encoded for token in row_ids)
    manifest = dict(pilot=pilot, unit='one pass over the same raw training documents',
                    selection_sha256=sha256_file(boundary / 'run.json'),
                    row_plan_sha256=g4_run['plan_sha256'], reference_scores_sha256={
                        'g2_g3': sha256_file(g3_path), 'g4_g6': sha256_file(g4_path),
                    }, g4_weights_sha256=g4['artifacts_sha256'], row_ids=ids,
                    g2_budget=planned, g3_budget=plan['training_budget'],
                    g2_selected_rows_with_input_support=sum(counts[token] > 0 for token in ids),
                    g2_selected_row_input_occurrences=sum(counts[token] for token in ids),
                    smoke=args.smoke)
    print(f'Seed {seed}: {planned}, {manifest["g2_selected_rows_with_input_support"]} selected rows '
          'occur in the G2 training input', flush=True)
    if args.plan:
        return

    device, dtype = verified_runtime(pilot)
    out = args.out / (f'smoke-seed-{seed}' if args.smoke else f'seed-{seed}')
    out.mkdir(exist_ok=True)
    with (out / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checkpoint = out / GROUP
        stored_manifest = out / 'run.json'
        if stored_manifest.exists():
            if read_json(stored_manifest) != manifest:
                raise ValueError(f'Run configuration changed in {out}')
        else:
            pending = stored_manifest.with_suffix('.tmp')
            write_json(pending, manifest)
            pending.replace(stored_manifest)
        if (out / 'results').exists():
            verified_group(checkpoint, planned, args.smoke, smoke_steps=2)
            splits = ('validation',) if args.smoke else SPLITS
            read_scores(out / 'results/bpb.csv', (GROUP,), splits)
            return
        if args.smoke:
            docs, paths = evaluation['validation']
            evaluation = {'validation': (smoke_items(docs, docs, 1),
                                         {GROUP: smoke_items(paths[GROUP], docs, 1)})}
            encoded = smoke_items(encoded, train, 2)
        rows = fit_group(checkpoint, ids, encoded, pilot, planned, evaluation, dtype, device,
                         args.smoke, smoke_steps=2)
        stats = verified_group(checkpoint, planned, args.smoke, smoke_steps=2)
        if any(stats[field] != g4[field] for field in ('trainable_parameters', 'embedding_parameters')):
            raise ValueError('G2 + G4 rows and G4 have different parameter counts')
        tables = {'bpb.csv': rows}
        if not args.smoke:
            g4_scores = read_scores(g4_path, (CONTROL, G4, 'G6-random'), SPLITS)
            references = control_scores(g3_path, G2) + [row for (_, group, _), row in g4_scores.items()
                                                         if group in (CONTROL, G4)]
            tables['factorial.csv'] = summarize(rows, references, evaluation, seed)
        publish(out / 'results', {**manifest, 'training': stats}, tables)
        print(f'Results verified: {out / "results"}', flush=True)


def paired_scores(directory, seed):
    manifest = read_json(directory / 'run.json')
    if manifest['smoke'] or manifest['pilot']['seed'] != seed:
        raise ValueError(f'Expected completed full results for seed {seed}: {directory}')
    check_budget(manifest['training'], manifest['g2_budget'], GROUP)
    measured = read_scores(directory / 'bpb.csv', (GROUP,), SPLITS)
    path = directory / 'factorial.csv'
    lookup = {}
    with path.open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            split, lang = row['split'], row['lang']
            key = split, lang
            scores = [float(row[field]) for field in ('g2', 'g3', 'g2_with_g4_rows', 'g4')]
            values = factorial_values(*scores)
            if (key in lookup or (split, GROUP, lang) not in measured or int(row['seed']) != seed
                    or values['g2_with_g4_rows'] != measured[split, GROUP, lang]['bpb']
                    or any(not isfinite(value) or value != float(row[field]) for field, value in values.items())):
                raise ValueError(f'Invalid paired contrast in {path}: {key}')
            lookup[key] = values
    if lookup.keys() != set(product(SPLITS, LANGS)):
        raise ValueError(f'Incomplete paired contrasts: {path}')
    return lookup


def summarize_seeds(out, seeds):
    scores = [paired_scores(out / f'seed-{seed}/results', seed) for seed in seeds]
    summary = []
    for split, lang in product(SPLITS, LANGS):
        row = {'split': split, 'lang': lang, 'seeds': ' '.join(map(str, seeds))}
        for field in scores[0][split, lang]:
            values = [table[split, lang][field] for table in scores]
            row[f'{field}_mean'] = mean(values)
            row[f'{field}_sd'] = stdev(values) if len(values) > 1 else ''
        summary.append(row)
    with TemporaryDirectory(prefix='.summary-', dir=out) as temporary:
        staged = Path(temporary) / 'summary.csv'
        with staged.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        staged.replace(out / 'summary.csv')
    print(f'Paired summary: {out / "summary.csv"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('runs/pilot'))
    parser.add_argument('--out', type=Path, help='defaults to RUN/g3-eval/review-g2-same-rows')
    parser.add_argument('--seed', type=int, nargs='+', choices=SEEDS, default=list(SEEDS))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--plan', action='store_true', help='verify exact IDs and budgets without a GPU')
    mode.add_argument('--smoke', action='store_true', help='fit two steps and score a small validation subset')
    mode.add_argument('--smoke-first', action='store_true', help='verify a short run before training all requested seeds')
    mode.add_argument('--summary', action='store_true', help='summarize completed paired scores without a GPU')
    args = parser.parse_args()
    if len(args.seed) != len(set(args.seed)):
        parser.error('each --seed value must be unique')
    args.out = args.out or args.run / 'g3-eval/review-g2-same-rows'
    if args.summary:
        summarize_seeds(args.out, args.seed)
        return
    boundary, selection, pilot = inputs(args.run)
    plan = read_plan(args.run / 'g3-eval/embedding-ablation', boundary, pilot)
    verified_langmap_directory(args.run)
    native = native_langmap(args.run)
    train = load_split(args.run / 'corpus', 'train.jsonl')
    encoded = list(encode_documents(train, native))
    evaluation = {}
    for split in SPLITS:
        rows = load_split(args.run / 'g3-eval', f'{split}.jsonl')
        ids = [tokens for _, tokens in encode_documents(rows, native)]
        if path_digest(((row, None, None) for row in rows), ids) != selection['token_path_sha256'][split][G2]:
            raise ValueError(f'{split}: G2 evaluation IDs differ from the frozen token paths')
        evaluation[split] = (rows, {GROUP: ids})
    if not args.plan:
        args.out.mkdir(parents=True, exist_ok=True)
    if args.smoke_first:
        smoke = copy(args)
        smoke.smoke = True
        run_seed(smoke, boundary, selection, plan, train, encoded, evaluation, args.seed[0])
    for seed in args.seed:
        run_seed(args, boundary, selection, plan, train, encoded, evaluation, seed)


if __name__ == '__main__':
    main()
