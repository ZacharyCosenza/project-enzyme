"""
Pre-compute ESM2 mean-pool embeddings.

Commands:
  esm2_embeddings_full  — full train/val splits for esm2_mlp
  esm2_embeddings_pairs — mutant/wildtype pairs for esm2_wt_delta_mlp

Usage:
  python preprocess.py esm2_embeddings_full --batch-size 4
  python preprocess.py esm2_embeddings_pairs --batch-size 4
"""
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


def _pool(hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden * mask).sum(1) / mask.sum(1)


def _embed(sequences, tokenizer, model, dev, batch_size):
    embeddings = []
    for i in tqdm(range(0, len(sequences), batch_size), desc='Embedding'):
        batch = sequences[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True,
                           truncation=True, max_length=1024).to(dev)
        with torch.no_grad():
            pooled = _pool(model(**inputs).last_hidden_state, inputs['attention_mask'])
        embeddings.append(pooled.cpu().float().numpy())
    return np.vstack(embeddings)


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

    print(f'Embedding {len(train_df)} train sequences...')
    np.save(FEATURES / f'{slug}_train.npy', _embed(train_df['protein_sequence'].tolist(), tokenizer, model, dev, args.batch_size))
    print(f'Embedding {len(val_df)} val sequences...')
    np.save(FEATURES / f'{slug}_val.npy', _embed(val_df['protein_sequence'].tolist(), tokenizer, model, dev, args.batch_size))
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
        print(f'Embedding {len(df)} {split} mutant sequences...')
        np.save(FEATURES / f'{slug}_{split}_mutant.npy', _embed(df['protein_sequence'].tolist(), tokenizer, model, dev, args.batch_size))
        print(f'Embedding {len(df)} {split} wildtype sequences...')
        np.save(FEATURES / f'{slug}_{split}_wildtype.npy', _embed(df['wildtype_sequence'].tolist(), tokenizer, model, dev, args.batch_size))

    test_df = data.test
    wt_seq = data.wildtype
    print(f'Embedding {len(test_df)} test mutant sequences...')
    np.save(FEATURES / f'{slug}_test_mutant.npy', _embed(test_df['protein_sequence'].tolist(), tokenizer, model, dev, args.batch_size))
    print(f'Embedding test wildtype (tiled to {len(test_df)} rows)...')
    wt_emb = _embed([wt_seq], tokenizer, model, dev, 1)
    np.save(FEATURES / f'{slug}_test_wildtype.npy', np.tile(wt_emb, (len(test_df), 1)))
    print(f'Saved to {FEATURES}')


def _register_args(parser):
    parser.add_argument('--model-name', default='facebook/esm2_t33_650M_UR50D')
    parser.add_argument('--batch-size', type=int, default=4)


COMMANDS = {
    'esm2_embeddings_full':  (run_full,  _register_args),
    'esm2_embeddings_pairs': (run_pairs, _register_args),
}
