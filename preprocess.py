#!/usr/bin/env python3
"""
python preprocess.py mutant_pairs
python preprocess.py esm2_embeddings_full --batch-size 4   # or esm1_embeddings_full
python preprocess.py esm2_embeddings_pairs --batch-size 4  # or esm1_embeddings_pairs
"""
import argparse
import importlib
import pkgutil

import src.preprocessing


def _discover():
    commands = {}
    for _, name, _ in pkgutil.iter_modules(src.preprocessing.__path__):
        mod = importlib.import_module(f'src.preprocessing.{name}')
        if hasattr(mod, 'COMMANDS'):
            for cmd_name, (run_fn, reg_fn) in mod.COMMANDS.items():
                commands[cmd_name] = (run_fn, reg_fn)
        else:
            commands[name] = (mod.run, getattr(mod, 'register_args', None))
    return commands


def main():
    commands = _discover()
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='step', required=True)
    for name, (run_fn, reg_fn) in commands.items():
        sp = sub.add_parser(name)
        if reg_fn:
            reg_fn(sp)
    args = parser.parse_args()
    run_fn, _ = commands[args.step]
    run_fn(args)


if __name__ == '__main__':
    main()
