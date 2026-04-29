"""
Pre-compute ESM2 mean-pool embeddings.

Outputs in data/03_features/ (full train/val, for esm2_mlp):
  {slug}_train.npy
  {slug}_val.npy

Outputs in data/03_features/ (mutant-wildtype pairs):
  {slug}_train_mutant.npy   — mutant sequence embeddings
  {slug}_train_wildtype.npy — corresponding derived wildtype embeddings
  {slug}_val_mutant.npy
  {slug}_val_wildtype.npy
  {slug}_test_mutant.npy
  {slug}_test_wildtype.npy  — test wildtype embedding tiled to match test rows

The train/val split matches run.py (test_size=0.1, random_state=42).
Requires data/02_processed/train_mutant_pairs.csv from: python preprocess.py mutant_pairs

Usage:
  python preprocess.py esm2_embeddings
  python preprocess.py esm2_embeddings --model-name facebook/esm2_t33_650M_UR50D --batch-size 4
"""
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from src import dataset as ds

PROCESSED = Path(__file__).parent.parent.parent / 'data' / '02_processed'  # mutant_pairs.csv lives here
FEATURES = Path(__file__).parent.parent.parent / 'data' / '03_features'


def _device():
    if torch.xpu.is_available():
        return torch.device('xpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def _pool(hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden * mask).sum(1) / mask.sum(1)


def embed(sequences, tokenizer, model, dev, batch_size):
    embeddings = []
    for i in tqdm(range(0, len(sequences), batch_size), desc='Embedding'):
        batch = sequences[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True,
                           truncation=True, max_length=1024).to(dev)
        with torch.no_grad():
            pooled = _pool(model(**inputs).last_hidden_state, inputs['attention_mask'])
        embeddings.append(pooled.cpu().float().numpy())
    return np.vstack(embeddings)


def run(args):
    data = ds.load()
    train_df, val_df = train_test_split(data.train, test_size=0.1, random_state=42)

    pairs_path = PROCESSED / 'train_mutant_pairs.csv'
    if not pairs_path.exists():
        raise FileNotFoundError('train_mutant_pairs.csv not found. Run: python preprocess.py mutant_pairs')
    pairs = pd.read_csv(pairs_path)
    pairs_train, pairs_val = train_test_split(pairs, test_size=0.1, random_state=42)

    dev = _device()
    print(f'Device: {dev}')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(dev).eval()

    slug = args.model_name.split('/')[-1]
    FEATURES.mkdir(parents=True, exist_ok=True)

    # Full train/val embeddings for esm2_mlp
    print(f'[03_features] Embedding {len(train_df)} train sequences...')
    np.save(FEATURES / f'{slug}_train.npy', embed(train_df['protein_sequence'].tolist(), tokenizer, model, dev, args.batch_size))
    print(f'[03_features] Embedding {len(val_df)} val sequences...')
    np.save(FEATURES / f'{slug}_val.npy', embed(val_df['protein_sequence'].tolist(), tokenizer, model, dev, args.batch_size))

    # Mutant-wildtype pair embeddings for 03_features
    for split, df in [('train', pairs_train), ('val', pairs_val)]:
        print(f'[03_features] Embedding {len(df)} {split} mutant sequences...')
        np.save(FEATURES / f'{slug}_{split}_mutant.npy', embed(df['protein_sequence'].tolist(), tokenizer, model, dev, args.batch_size))
        print(f'[03_features] Embedding {len(df)} {split} wildtype sequences...')
        np.save(FEATURES / f'{slug}_{split}_wildtype.npy', embed(df['wildtype_sequence'].tolist(), tokenizer, model, dev, args.batch_size))

    # Test: all sequences share one wildtype
    test_df = data.test
    wt_seq = data.wildtype
    print(f'[03_features] Embedding {len(test_df)} test mutant sequences...')
    np.save(FEATURES / f'{slug}_test_mutant.npy', embed(test_df['protein_sequence'].tolist(), tokenizer, model, dev, args.batch_size))
    print(f'[03_features] Embedding test wildtype (tiled to {len(test_df)} rows)...')
    wt_emb = embed([wt_seq], tokenizer, model, dev, 1)
    np.save(FEATURES / f'{slug}_test_wildtype.npy', np.tile(wt_emb, (len(test_df), 1)))

    print(f'\nSaved all embeddings to {FEATURES}')


def register_args(parser):
    parser.add_argument('--model-name', default='facebook/esm2_t33_650M_UR50D')
    parser.add_argument('--batch-size', type=int, default=4)
