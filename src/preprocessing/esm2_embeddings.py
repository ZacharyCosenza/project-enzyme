"""
Pre-compute ESM2 mean-pool embeddings for train and val splits.

Output: data/02_processed/{model_slug}_train.npy
        data/02_processed/{model_slug}_val.npy

The train/val split matches run.py (test_size=0.1, random_state=42).

Usage:
  python preprocess.py esm2_embeddings
  python preprocess.py esm2_embeddings --model-name facebook/esm2_t33_650M_UR50D --batch-size 4
"""
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from src import dataset as ds

PROCESSED = Path(__file__).parent.parent.parent / 'data' / '02_processed'


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
    print(f'Train: {len(train_df)}  Val: {len(val_df)}')

    dev = _device()
    print(f'Device: {dev}')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(dev).eval()

    slug = args.model_name.split('/')[-1]
    PROCESSED.mkdir(parents=True, exist_ok=True)
    train_path = PROCESSED / f'{slug}_train.npy'
    val_path = PROCESSED / f'{slug}_val.npy'

    print(f'Embedding {len(train_df)} train sequences...')
    np.save(train_path, embed(train_df['protein_sequence'].tolist(), tokenizer, model, dev, args.batch_size))
    print(f'Embedding {len(val_df)} val sequences...')
    np.save(val_path, embed(val_df['protein_sequence'].tolist(), tokenizer, model, dev, args.batch_size))
    print(f'Saved to {PROCESSED}')


def register_args(parser):
    parser.add_argument('--model-name', default='facebook/esm2_t33_650M_UR50D')
    parser.add_argument('--batch-size', type=int, default=4)
