"""
Frozen ESM2 backbone + MLP head on pre-computed embeddings.

Requires embeddings to be pre-computed:
  python preprocess.py esm2_embeddings
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from tqdm import tqdm

from src.dataset import Dataset

WILDTYPE = (
    'VPVNPEPDATSVENVALKTGSGDSQSDPIKADLEVKGQSALPFDVDCWAILCKGAPNVLQRVNEKTKNSNRDRSGANK'
    'GPFKDPQKWGIKALPPKNPSWSAQDFKSPEEYAFASSLQGGTNAILAPVNLASQNSQGGVLNGFYSANKVAQFDPSKP'
    'QQTKGTWFQITKFTGAAGPYCKALGSNDKSVCDKNKNIAGDWGFDPAKWAYQYDEKNNKFNYVGK'
)

PROCESSED = Path(__file__).parent.parent.parent / 'data' / '02_processed'

CONFIGS = {
    'esm2_mlp': {
        'wildtype': WILDTYPE,
        'model_name': 'facebook/esm2_t33_650M_UR50D',
        'hidden_dims': [512, 128],
        'dropout': 0.1,
        'lr': 1e-3,
        'epochs': 50,
        'patience': 5,
        'batch_size': 512,
        'embed_batch_size': 4,
    },
}

SWEEP = {
    'esm2_mlp': {
        'hidden_dims': {'type': 'categorical', 'values': [[256, 64], [512, 128], [1024, 256], [512, 256, 64], [1024, 512, 128]]},
        'dropout':     {'type': 'float',       'low': 0.0,  'high': 0.5},
        'lr':          {'type': 'float',       'low': 1e-4, 'high': 1e-2, 'log': True},
        'batch_size':  {'type': 'categorical', 'values': [256, 512, 1024]},
    },
}


class _MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, dropout):
        super().__init__()
        dims = [in_dim] + list(hidden_dims) + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


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


def _cache_paths(params):
    slug = params['model_name'].split('/')[-1]
    return PROCESSED / f'{slug}_train.npy', PROCESSED / f'{slug}_val.npy'


def fit(dataset: Dataset, params: dict) -> None:
    train_path, val_path = _cache_paths(params)
    if not (train_path.exists() and val_path.exists()):
        raise FileNotFoundError('Cached embeddings not found. Run: python preprocess.py esm2_embeddings')

    print('Loading cached embeddings...')
    X_train = torch.tensor(np.load(train_path))
    X_val = torch.tensor(np.load(val_path))

    scaler = StandardScaler()
    y_train = torch.tensor(scaler.fit_transform(dataset.train['tm'].values.reshape(-1, 1)).flatten().astype(np.float32))
    y_val = torch.tensor(scaler.transform(dataset.val['tm'].values.reshape(-1, 1)).flatten().astype(np.float32))
    params['_scaler'] = scaler

    hidden_dims = params['hidden_dims']
    if isinstance(hidden_dims, str):
        hidden_dims = [int(x) for x in hidden_dims.strip('[]').split(',')]

    mlp = _MLP(X_train.shape[1], hidden_dims, float(params['dropout']))
    opt = torch.optim.AdamW(mlp.parameters(), lr=float(params['lr']))
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
    mlp = params['_mlp']
    scaler = params['_scaler']
    tokenizer = AutoTokenizer.from_pretrained(params['model_name'])
    model = AutoModel.from_pretrained(params['model_name']).to(_device()).eval()
    embeddings = _embed(dataset.test['protein_sequence'].tolist(), tokenizer, model, _device(), int(params['embed_batch_size']))
    with torch.no_grad():
        norm_preds = mlp(torch.tensor(embeddings)).numpy()
    return scaler.inverse_transform(norm_preds.reshape(-1, 1)).flatten()
