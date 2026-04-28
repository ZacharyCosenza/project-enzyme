#!/usr/bin/env python3
"""
python preprocess.py mutant_pairs
python preprocess.py esm2_embeddings --model-name facebook/esm2_t33_650M_UR50D --batch-size 4
"""
import argparse
import importlib
import pkgutil

import src.preprocessing


def _discover():
    modules = {}
    for _, name, _ in pkgutil.iter_modules(src.preprocessing.__path__):
        mod = importlib.import_module(f'src.preprocessing.{name}')
        modules[name] = mod
    return modules


def main():
    modules = _discover()
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='step', required=True)
    for name, mod in modules.items():
        sp = sub.add_parser(name)
        if hasattr(mod, 'register_args'):
            mod.register_args(sp)
    args = parser.parse_args()
    modules[args.step].run(args)


if __name__ == '__main__':
    main()
