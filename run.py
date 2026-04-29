#!/usr/bin/env python3
"""
Usage:
    python run.py --model esm2_mlp --name my_run
    python run.py --model esm2_mlp --trials 20          # Bayesian (TPE)
    python run.py --model esm2_mlp --trials 20 --grid   # grid search
    python run.py --list
"""
import argparse
import importlib
import pkgutil
import csv
import json
from datetime import datetime
from pathlib import Path
from copy import deepcopy

import numpy as np
from sklearn.model_selection import train_test_split

import src.experiments
from src import dataset, evaluate

LOGS_DIR = Path(__file__).parent / 'logs'


def _discover() -> dict:
    registry = {}
    for _, name, _ in pkgutil.iter_modules(src.experiments.__path__):
        mod = importlib.import_module(f'src.experiments.{name}')
        for config_name, params in getattr(mod, 'CONFIGS', {}).items():
            registry[config_name] = (mod, params)
    return registry


REGISTRY = _discover()


def _load_data():
    data = dataset.load()
    train_df, val_df = train_test_split(data.train, test_size=0.1, random_state=42)
    return dataset.Dataset(
        train=train_df.reset_index(drop=True),
        val=val_df.reset_index(drop=True),
        test=data.test,
    )


def compute_metrics(y_true, y_pred, wt_tm):
    metrics = {}
    metrics.update(evaluate.spearman(y_true, y_pred))
    metrics.update(evaluate.kendall(y_true, y_pred))
    metrics.update(evaluate.ndcg_at_k(y_true, y_pred))
    metrics.update(evaluate.auc_stabilizing(y_true, y_pred, wt_tm))
    return metrics


def _wt_tm(data, params):
    wt_seq = params.get('wildtype')
    if not wt_seq:
        return None
    rows = data.test[data.test['protein_sequence'] == wt_seq]
    return float(rows['tm'].values[0]) if len(rows) else None


def _run_once(exp, params, data, run_dir):
    run_dir.mkdir(parents=True)
    print(f'\n[fit]     params={json.dumps({k: v for k, v in params.items() if not k.startswith("_")}, default=str)}')
    exp.fit(data, params)
    predictions = exp.predict(data, params)
    if predictions is None:
        return None
    valid = ~np.isnan(predictions)
    metrics = compute_metrics(
        y_true=data.test['tm'].values[valid],
        y_pred=predictions[valid],
        wt_tm=_wt_tm(data, params),
    )
    evaluate.report(metrics)
    with open(run_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    if '_history' in params:
        with open(run_dir / 'loss_curve.json', 'w') as f:
            json.dump(params['_history'], f, indent=2)
    with open(run_dir / 'predictions.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['seq_id', 'protein_sequence', 'score'])
        for seq_id, seq, score in zip(data.test['seq_id'], data.test['protein_sequence'], predictions):
            writer.writerow([seq_id, seq, '' if np.isnan(score) else score])
    return metrics


def _suggest(trial, sweep):
    suggested = {}
    for key, spec in sweep.items():
        t = spec['type']
        if t == 'categorical':
            suggested[key] = trial.suggest_categorical(key, spec['values'])
        elif t == 'float':
            suggested[key] = trial.suggest_float(key, spec['low'], spec['high'], log=spec.get('log', False))
        elif t == 'int':
            suggested[key] = trial.suggest_int(key, spec['low'], spec['high'], log=spec.get('log', False))
    return suggested


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=list(REGISTRY))
    parser.add_argument('--name', help='Human-readable run name (required for single runs)')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--trials', type=int, help='Number of tuning trials')
    parser.add_argument('--grid', action='store_true', help='Use grid search instead of Bayesian (TPE)')
    args, extra = parser.parse_known_args()

    if args.list:
        for name in sorted(REGISTRY):
            print(name)
        return

    if not args.model:
        parser.error('--model is required')

    exp, base_params = REGISTRY[args.model]
    base_params = dict(base_params)

    i = 0
    while i < len(extra):
        if extra[i].startswith('--'):
            key = extra[i][2:]
            if i + 1 < len(extra) and not extra[i + 1].startswith('--'):
                base_params[key] = extra[i + 1]
                i += 2
            else:
                base_params[key] = True
                i += 1
        else:
            i += 1

    data = _load_data()

    if not args.name:
        parser.error('--name is required')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = LOGS_DIR / f'{args.name}_{timestamp}'

    if args.trials:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        sweep = getattr(exp, 'SWEEP', {}).get(args.model)
        if not sweep:
            parser.error(f'No SWEEP defined for {args.model}')

        if args.grid:
            from optuna.samplers import GridSampler
            grid = {k: v['values'] for k, v in sweep.items() if v['type'] == 'categorical'}
            sampler = GridSampler(grid)
        else:
            from optuna.samplers import TPESampler
            sampler = TPESampler()

        run_dir.mkdir(parents=True)
        results = []

        def objective(trial):
            params = deepcopy(base_params)
            params.update(_suggest(trial, sweep))
            trial_dir = run_dir / f'trial_{trial.number:03d}'
            metrics = _run_once(exp, params, data, trial_dir)
            if metrics is None:
                raise optuna.exceptions.TrialPruned()
            row = {'trial': trial.number, **{k: v for k, v in params.items() if not k.startswith('_')}, **metrics}
            results.append(row)
            with open(run_dir / 'results.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            return metrics['spearman_rho']

        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective, n_trials=args.trials)

        best = study.best_trial
        summary = {'best_trial': best.number, 'best_spearman_rho': study.best_value, 'best_params': best.params}
        with open(run_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f'\nBest trial: {best.number}  spearman_rho={study.best_value:.4f}')
        print(f'Best params: {best.params}')
        print(f'Results saved to {run_dir}')

    else:
        metrics = _run_once(exp, base_params, data, run_dir)
        if metrics:
            print(f'\nSaved to {run_dir}')


if __name__ == '__main__':
    main()
