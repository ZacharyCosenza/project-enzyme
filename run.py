#!/usr/bin/env python3
"""
Usage:
    python run.py --model llr_baseline --name my_run
    python run.py --model llr_baseline_esm1v --name esm1v_run
    python run.py --list
"""
import argparse
import importlib
import pkgutil
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

import src.experiments
from src import dataset, evaluate

LOGS_DIR = Path(__file__).parent / 'logs'


def _discover() -> dict:
    """Collect all CONFIGS from every module in src/experiments/."""
    registry = {}
    for _, name, _ in pkgutil.iter_modules(src.experiments.__path__):
        mod = importlib.import_module(f'src.experiments.{name}')
        for config_name, params in getattr(mod, 'CONFIGS', {}).items():
            registry[config_name] = (mod, params)
    return registry


REGISTRY = _discover()


def compute_metrics(y_true, y_pred, wt_tm):
    metrics = {}
    metrics.update(evaluate.spearman(y_true, y_pred))
    metrics.update(evaluate.kendall(y_true, y_pred))
    metrics.update(evaluate.ndcg_at_k(y_true, y_pred))
    metrics.update(evaluate.auc_stabilizing(y_true, y_pred, wt_tm))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=list(REGISTRY))
    parser.add_argument('--name',  help='Human-readable run name')
    parser.add_argument('--list',  action='store_true', help='List available models')
    args = parser.parse_args()

    if args.list:
        for name in sorted(REGISTRY):
            print(name)
        return

    if not args.model or not args.name:
        parser.error('--model and --name are required')

    exp, params = REGISTRY[args.model]
    params = dict(params)
    data  = dataset.load()

    wt_seq  = params.get('wildtype')
    wt_rows = data.test[data.test['protein_sequence'] == wt_seq] if wt_seq else []
    wt_tm   = float(wt_rows['tm'].values[0]) if len(wt_rows) else None

    print(f'\n[fit]     {args.model}')
    exp.fit(data, params)

    print(f'[predict] {args.model}')
    predictions = exp.predict(data, params)

    valid   = ~np.isnan(predictions)
    metrics = compute_metrics(
        y_true=data.test['tm'].values[valid],
        y_pred=predictions[valid],
        wt_tm=wt_tm,
    )
    print()
    evaluate.report(metrics)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir   = LOGS_DIR / f'{args.name}_{timestamp}'
    run_dir.mkdir(parents=True)

    with open(run_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    with open(run_dir / 'predictions.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['seq_id', 'protein_sequence', 'score'])
        for seq_id, seq, score in zip(
            data.test['seq_id'],
            data.test['protein_sequence'],
            predictions,
        ):
            writer.writerow([seq_id, seq, '' if np.isnan(score) else score])

    print(f'\nSaved to {run_dir}')


if __name__ == '__main__':
    main()
