"""
Frozen ESM2 + MLP head on delta (mutant - wildtype) embeddings.

Input: 1280-dim difference of ESM2 mean-pool embeddings from 03_features.
Wildtype per training cluster is the consensus sequence (modal AA per position).
Wildtype for test is the known Novozymes sequence.

Requires:
  python preprocess.py mutant_pairs
  python preprocess.py esm2_embeddings_pairs
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path

from src.dataset import Dataset

WILDTYPE = (
    'VPVNPEPDATSVENVALKTGSGDSQSDPIKADLEVKGQSALPFDVDCWAILCKGAPNVLQRVNEKTKNSNRDRSGANK'
    'GPFKDPQKWGIKALPPKNPSWSAQDFKSPEEYAFASSLQGGTNAILAPVNLASQNSQGGVLNGFYSANKVAQFDPSKP'
    'QQTKGTWFQITKFTGAAGPYCKALGSNDKSVCDKNKNIAGDWGFDPAKWAYQYDEKNNKFNYVGK'
)

PROCESSED = Path(__file__).parent.parent.parent / 'data' / '02_processed'
FEATURES = Path(__file__).parent.parent.parent / 'data' / '03_features'

CONFIGS = {
    'esm2_wt_delta_mlp': {
        'wildtype': WILDTYPE,
        'model_name': 'facebook/esm2_t33_650M_UR50D',
        'hidden_dims': [1024, 512, 128],
        'dropout': 0.312,
        'lr': 0.00897,
        'weight_decay': 0.00025,
        'activation': 'gelu',
        'epochs': 50,
        'patience': 5,
        'batch_size': 512,
    },
}

SWEEP = {
    'esm2_wt_delta_mlp': {
        'hidden_dims':  {'type': 'categorical', 'values': [[512, 128], [1024, 256], [1024, 512, 128], [1024, 512, 256, 128]]},
        'dropout':      {'type': 'float',       'low': 0.2,  'high': 0.5},
        'lr':           {'type': 'float',       'low': 1e-4, 'high': 1e-2, 'log': True},
        'weight_decay': {'type': 'float',       'low': 1e-5, 'high': 1e-2, 'log': True},
        'batch_size':   {'type': 'categorical', 'values': [256, 512]},
        'activation':   {'type': 'categorical', 'values': ['relu', 'gelu', 'leaky_relu']},
    },
}

_ACTIVATIONS = {'relu': nn.ReLU, 'gelu': nn.GELU, 'leaky_relu': nn.LeakyReLU}


class _MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, dropout, activation='gelu'):
        super().__init__()
        act_cls = _ACTIVATIONS[activation]
        dims = [in_dim] + list(hidden_dims) + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(act_cls())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _cache_paths(params):
    slug = params['model_name'].split('/')[-1]
    return {
        'train_mutant':    FEATURES / f'{slug}_train_mutant.npy',
        'train_wildtype':  FEATURES / f'{slug}_train_wildtype.npy',
        'val_mutant':      FEATURES / f'{slug}_val_mutant.npy',
        'val_wildtype':    FEATURES / f'{slug}_val_wildtype.npy',
        'test_mutant':     FEATURES / f'{slug}_test_mutant.npy',
        'test_wildtype':   FEATURES / f'{slug}_test_wildtype.npy',
    }


def _load_pairs_labels():
    pairs_path = PROCESSED / 'train_mutant_pairs.csv'
    if not pairs_path.exists():
        raise FileNotFoundError('train_mutant_pairs.csv not found. Run: python preprocess.py mutant_pairs')
    pairs = pd.read_csv(pairs_path)
    train_pairs, val_pairs = train_test_split(pairs, test_size=0.1, random_state=42)
    return train_pairs['tm'].values, val_pairs['tm'].values


def fit(dataset: Dataset, params: dict) -> None:
    paths = _cache_paths(params)
    missing = [k for k, p in paths.items() if not p.exists() if 'test' not in k]
    if missing:
        raise FileNotFoundError(f'Missing features: {missing}. Run: python preprocess.py esm2_embeddings_pairs')

    print('Loading cached embeddings...')
    X_train = torch.tensor(np.load(paths['train_mutant']) - np.load(paths['train_wildtype']))
    X_val = torch.tensor(np.load(paths['val_mutant']) - np.load(paths['val_wildtype']))

    tm_train, tm_val = _load_pairs_labels()
    scaler = StandardScaler()
    y_train = torch.tensor(scaler.fit_transform(tm_train.reshape(-1, 1)).flatten().astype(np.float32))
    y_val = torch.tensor(scaler.transform(tm_val.reshape(-1, 1)).flatten().astype(np.float32))
    params['_scaler'] = scaler

    hidden_dims = params['hidden_dims']
    if isinstance(hidden_dims, str):
        hidden_dims = [int(x) for x in hidden_dims.strip('[]').split(',')]

    mlp = _MLP(X_train.shape[1], hidden_dims, float(params['dropout']), params.get('activation', 'gelu'))
    opt = torch.optim.AdamW(mlp.parameters(), lr=float(params['lr']), weight_decay=float(params.get('weight_decay', 0.0)))
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=int(params['batch_size']), shuffle=True)

    best_val_loss = float('inf')
    best_weights = None
    no_improve = 0
    train_losses, val_losses = [], []

    for epoch in range(int(params['epochs'])):
        mlp.train()
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(mlp(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        train_loss = total / len(X_train)

        mlp.eval()
        with torch.no_grad():
            val_loss = loss_fn(mlp(X_val), y_val).item()

        train_losses.append(round(train_loss, 6))
        val_losses.append(round(val_loss, 6))
        print(f'  epoch {epoch+1:02d}/{params["epochs"]}  train={train_loss:.4f}  val={val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.clone() for k, v in mlp.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= int(params['patience']):
                print(f'  early stopping at epoch {epoch+1}')
                break

    mlp.load_state_dict(best_weights)
    mlp.eval()
    params['_mlp'] = mlp
    params['_history'] = {'train_loss': train_losses, 'val_loss': val_losses}


def predict(dataset: Dataset, params: dict) -> np.ndarray:
    paths = _cache_paths(params)
    X_test = torch.tensor(np.load(paths['test_mutant']) - np.load(paths['test_wildtype']))
    with torch.no_grad():
        norm_preds = params['_mlp'](X_test).numpy()
    return params['_scaler'].inverse_transform(norm_preds.reshape(-1, 1)).flatten()
