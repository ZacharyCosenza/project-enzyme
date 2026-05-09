"""
CNN over delta distance maps (dist_mutant - dist_wildtype) from ESMFold PDBs.

Maps are computed on-the-fly from PDB files — only pairs with both mutant and
wildtype PDB present are used. Zero-padded to MAP_SIZE x MAP_SIZE with a
sequence-length mask applied at global average pooling.

Requires:
  python preprocess.py mutant_pairs
  python preprocess.py esmfold_structures
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
from Bio.PDB import PDBParser

from src.dataset import Dataset
from src.preprocessing.structures import STRUCTURES, WILDTYPE, _wt_hash, API_MAX_LEN

PROCESSED = Path(__file__).parent.parent.parent / 'data' / '02_processed'
MAP_SIZE = API_MAX_LEN

CONFIGS = {
    'cnn_delta_map': {
        'channels': [32, 64, 128],
        'kernel_size': 5,
        'dropout': 0.3,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'epochs': 50,
        'patience': 5,
        'batch_size': 32,
    },
}

SWEEP = {
    'cnn_delta_map': {
        'channels': {'type': 'categorical', 'values': [[32, 64], [32, 64, 128], [64, 128, 256]]},
        'kernel_size': {'type': 'categorical', 'values': [3, 5, 7]},
        'dropout': {'type': 'float', 'low': 0.1, 'high': 0.5},
        'lr': {'type': 'float', 'low': 1e-4, 'high': 1e-2, 'log': True},
        'weight_decay': {'type': 'float', 'low': 1e-5, 'high': 1e-2, 'log': True},
        'batch_size': {'type': 'categorical', 'values': [16, 32, 64]},
    },
}


def _ca_coords(pdb_path: Path) -> np.ndarray:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    return np.array(
        [res['CA'].get_vector().get_array()
         for res in structure.get_residues() if res.has_id('CA')],
        dtype=np.float32,
    )


def _dist_matrix(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None] - coords[None, :]
    return np.sqrt((diff ** 2).sum(-1))


def _delta_map(mut_pdb: Path, wt_pdb: Path) -> tuple[np.ndarray, int]:
    coords_mut = _ca_coords(mut_pdb)
    coords_wt = _ca_coords(wt_pdb)
    assert len(coords_mut) == len(coords_wt)
    n = len(coords_mut)
    delta = _dist_matrix(coords_mut) - _dist_matrix(coords_wt)
    padded = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
    padded[:n, :n] = delta
    return padded, n


def _load_maps(rows: pd.DataFrame, mut_pdb_fn, wt_pdb_fn):
    maps, lengths, ids = [], [], []
    skipped = 0
    for _, row in rows.iterrows():
        mut_pdb = mut_pdb_fn(row)
        wt_pdb = wt_pdb_fn(row)
        if not (mut_pdb.exists() and wt_pdb.exists()):
            skipped += 1
            continue
        try:
            delta, n = _delta_map(mut_pdb, wt_pdb)
            maps.append(delta)
            lengths.append(n)
            ids.append(int(row['seq_id']))
        except Exception as e:
            print(f'    skipping {mut_pdb.name}: {e}')
            skipped += 1
    if skipped:
        print(f'    skipped {skipped} pairs (missing PDB)')
    return np.stack(maps), np.array(lengths, dtype=np.int32), np.array(ids, dtype=np.int64)


class _MaskedGlobalAvgPool(nn.Module):
    def __init__(self, n_pool: int):
        super().__init__()
        self.stride = 2 ** n_pool

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        eff = torch.ceil(lengths.float() / self.stride).long().clamp(max=H)
        idx = torch.arange(H, device=x.device).unsqueeze(0)
        mask = (idx < eff.unsqueeze(1)).float()
        mask2d = (mask.unsqueeze(2) * mask.unsqueeze(1)).unsqueeze(1)
        return (x * mask2d).sum(dim=(2, 3)) / mask2d.sum(dim=(2, 3)).clamp(min=1)


class _CNN(nn.Module):
    def __init__(self, channels, kernel_size, dropout):
        super().__init__()
        pad = kernel_size // 2
        layers = []
        in_ch = 1
        for out_ch in channels:
            layers += [nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad),
                       nn.ReLU(),
                       nn.MaxPool2d(2)]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.pool = _MaskedGlobalAvgPool(n_pool=len(channels))
        self.head = nn.Sequential(
            nn.Linear(in_ch, in_ch // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(in_ch // 2, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.conv(x), lengths)).squeeze(-1)


def fit(dataset: Dataset, params: dict) -> None:
    pairs_path = PROCESSED / 'train_mutant_pairs.csv'
    if not pairs_path.exists():
        raise FileNotFoundError('train_mutant_pairs.csv not found. Run: python preprocess.py mutant_pairs')
    pairs = pd.read_csv(pairs_path)
    pairs_train, pairs_val = train_test_split(pairs, test_size=0.1, random_state=42)

    print('Building train delta maps...')
    train_maps, train_lens, train_ids = _load_maps(
        pairs_train,
        mut_pdb_fn=lambda r: STRUCTURES / f'train_mutant_{r["seq_id"]}.pdb',
        wt_pdb_fn=lambda r: STRUCTURES / f'train_wildtype_{_wt_hash(r["wildtype_sequence"])}.pdb',
    )
    print('Building val delta maps...')
    val_maps, val_lens, val_ids = _load_maps(
        pairs_val,
        mut_pdb_fn=lambda r: STRUCTURES / f'train_mutant_{r["seq_id"]}.pdb',
        wt_pdb_fn=lambda r: STRUCTURES / f'train_wildtype_{_wt_hash(r["wildtype_sequence"])}.pdb',
    )

    tm_lookup = pairs.set_index('seq_id')['tm']
    scaler = StandardScaler()
    y_train = torch.tensor(scaler.fit_transform(tm_lookup.loc[train_ids].values.reshape(-1, 1)).flatten().astype(np.float32))
    y_val = torch.tensor(scaler.transform(tm_lookup.loc[val_ids].values.reshape(-1, 1)).flatten().astype(np.float32))
    params['_scaler'] = scaler

    X_train = torch.tensor(train_maps).unsqueeze(1)
    X_val = torch.tensor(val_maps).unsqueeze(1)
    L_train = torch.tensor(train_lens)
    L_val = torch.tensor(val_lens)

    channels = params['channels']
    if isinstance(channels, str):
        channels = [int(x) for x in channels.strip('[]').split(',')]

    cnn = _CNN(channels, int(params['kernel_size']), float(params['dropout']))
    opt = torch.optim.AdamW(cnn.parameters(), lr=float(params['lr']), weight_decay=float(params.get('weight_decay', 0.0)))
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(X_train, L_train, y_train), batch_size=int(params['batch_size']), shuffle=True)

    best_val_loss, best_weights, no_improve = float('inf'), None, 0
    train_losses, val_losses = [], []

    for epoch in range(int(params['epochs'])):
        cnn.train()
        total = 0.0
        for xb, lb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(cnn(xb, lb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        train_loss = total / len(X_train)

        cnn.eval()
        with torch.no_grad():
            val_loss = loss_fn(cnn(X_val, L_val), y_val).item()

        train_losses.append(round(train_loss, 6))
        val_losses.append(round(val_loss, 6))
        print(f'  epoch {epoch+1:02d}/{params["epochs"]}  train={train_loss:.4f}  val={val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.clone() for k, v in cnn.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= int(params['patience']):
                print(f'  early stopping at epoch {epoch+1}')
                break

    cnn.load_state_dict(best_weights)
    cnn.eval()
    params['_cnn'] = cnn
    params['_history'] = {'train_loss': train_losses, 'val_loss': val_losses}


def predict(dataset: Dataset, params: dict) -> np.ndarray:
    wt_pdb = STRUCTURES / 'test_wildtype.pdb'
    print('Building test delta maps...')
    test_maps, test_lens, _ = _load_maps(
        dataset.test,
        mut_pdb_fn=lambda r: STRUCTURES / f'test_mutant_{r["seq_id"]}.pdb',
        wt_pdb_fn=lambda _: wt_pdb,
    )
    X_test = torch.tensor(test_maps).unsqueeze(1)
    L_test = torch.tensor(test_lens)
    with torch.no_grad():
        norm_preds = params['_cnn'](X_test, L_test).numpy()
    return params['_scaler'].inverse_transform(norm_preds.reshape(-1, 1)).flatten()
