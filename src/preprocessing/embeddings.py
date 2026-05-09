"""
Embedding preprocessing clearinghouse. Add new model families here.

Saves full per-residue embeddings (seq_len, hidden_dim) to HDF5, keyed by seq_id.
Mean pooling and other reductions happen on the fly in experiments.

Output files in data/03_features/:
  {slug}_train.h5 / {slug}_val.h5 / {slug}_test.h5
  {slug}_train_mutant.h5 / {slug}_train_wildtype.h5
  {slug}_val_mutant.h5   / {slug}_val_wildtype.h5
  {slug}_test_mutant.h5  / {slug}_test_wildtype.h5

Keys within each file are str(seq_id). Wildtype files use MD5 hash of sequence
as key, except test_wildtype.h5 which uses the key 'wildtype'.

Commands:
  esm2_embeddings_full  / esm1_embeddings_full  — full train/val/test splits
  esm2_embeddings_pairs / esm1_embeddings_pairs — mutant/wildtype pairs

Usage:
  python preprocess.py esm2_embeddings_full --batch-size 4
  python preprocess.py esm1_embeddings_pairs --batch-size 4
  python preprocess.py esm2_embeddings_full --model-name facebook/esm2_t6_8M_UR50D --batch-size 4
"""
import hashlib
import h5py
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from src import dataset as ds

PROCESSED = Path(__file__).parent.parent.parent / 'data' / '02_processed'
FEATURES = Path(__file__).parent.parent.parent / 'data' / '03_features'


def _device():
    if torch.xpu.is_available():
        return torch.device('xpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def _embed_to_h5(sequences, keys, tokenizer, model, dev, batch_size, out_path):
    """Embed sequences and write per-residue arrays to HDF5 keyed by str(key)."""
    with h5py.File(out_path, 'w') as f:
        for i in tqdm(range(0, len(sequences), batch_size), desc=f'Embedding → {out_path.name}'):
            batch_seqs = sequences[i:i+batch_size]
            batch_keys = keys[i:i+batch_size]
            inputs = tokenizer(batch_seqs, return_tensors='pt', padding=True,
                               truncation=True, max_length=1024).to(dev)
            with torch.no_grad():
                hidden = model(**inputs).last_hidden_state
            seq_lens = inputs['attention_mask'].sum(dim=1) - 2  # strip BOS + EOS
            for j, (key, length) in enumerate(zip(batch_keys, seq_lens)):
                emb = hidden[j, 1:length+1].cpu().float().numpy()
                f.create_dataset(str(key), data=emb)


def _wt_hash(seq):
    return hashlib.md5(seq.encode()).hexdigest()[:10]


def _load_model(args):
    dev = _device()
    print(f'Device: {dev}')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(dev).eval()
    return tokenizer, model, dev


def run_full(args):
    data = ds.load()
    train_df, val_df = train_test_split(data.train, test_size=0.1, random_state=42)

    tokenizer, model, dev = _load_model(args)
    slug = args.model_name.split('/')[-1]
    FEATURES.mkdir(parents=True, exist_ok=True)

    _embed_to_h5(train_df['protein_sequence'].tolist(), train_df['seq_id'].tolist(),
                 tokenizer, model, dev, args.batch_size, FEATURES / f'{slug}_train.h5')
    _embed_to_h5(val_df['protein_sequence'].tolist(), val_df['seq_id'].tolist(),
                 tokenizer, model, dev, args.batch_size, FEATURES / f'{slug}_val.h5')
    _embed_to_h5(data.test['protein_sequence'].tolist(), data.test['seq_id'].tolist(),
                 tokenizer, model, dev, args.batch_size, FEATURES / f'{slug}_test.h5')

    print(f'Saved to {FEATURES}')


def run_pairs(args):
    pairs_path = PROCESSED / 'train_mutant_pairs.csv'
    if not pairs_path.exists():
        raise FileNotFoundError('train_mutant_pairs.csv not found. Run: python preprocess.py mutant_pairs')
    pairs = pd.read_csv(pairs_path)
    pairs_train, pairs_val = train_test_split(pairs, test_size=0.1, random_state=42)

    data = ds.load()
    tokenizer, model, dev = _load_model(args)
    slug = args.model_name.split('/')[-1]
    FEATURES.mkdir(parents=True, exist_ok=True)

    for split, df in [('train', pairs_train), ('val', pairs_val)]:
        _embed_to_h5(df['protein_sequence'].tolist(), df['seq_id'].tolist(),
                     tokenizer, model, dev, args.batch_size,
                     FEATURES / f'{slug}_{split}_mutant.h5')
        wt_seqs = df['wildtype_sequence'].tolist()
        _embed_to_h5(wt_seqs, [_wt_hash(s) for s in wt_seqs],
                     tokenizer, model, dev, args.batch_size,
                     FEATURES / f'{slug}_{split}_wildtype.h5')

    _embed_to_h5(data.test['protein_sequence'].tolist(), data.test['seq_id'].tolist(),
                 tokenizer, model, dev, args.batch_size,
                 FEATURES / f'{slug}_test_mutant.h5')

    wt_emb_path = FEATURES / f'{slug}_test_wildtype.h5'
    with h5py.File(wt_emb_path, 'w') as f:
        inputs = tokenizer([data.wildtype], return_tensors='pt',
                           truncation=True, max_length=1024).to(dev)
        with torch.no_grad():
            hidden = model(**inputs).last_hidden_state
        length = inputs['attention_mask'].sum() - 2
        f.create_dataset('wildtype', data=hidden[0, 1:length+1].cpu().float().numpy())
    print(f'Saved test wildtype → {wt_emb_path.name}')

    print(f'Saved to {FEATURES}')


def _register_esm2(parser):
    parser.add_argument('--model-name', default='facebook/esm2_t33_650M_UR50D')
    parser.add_argument('--batch-size', type=int, default=4)


def _register_esm1(parser):
    parser.add_argument('--model-name', default='facebook/esm1b_t33_650M_UR50S')
    parser.add_argument('--batch-size', type=int, default=4)


COMMANDS = {
    'esm2_embeddings_full':  (run_full,  _register_esm2),
    'esm2_embeddings_pairs': (run_pairs, _register_esm2),
    'esm1_embeddings_full':  (run_full,  _register_esm1),
    'esm1_embeddings_pairs': (run_pairs, _register_esm1),
}
