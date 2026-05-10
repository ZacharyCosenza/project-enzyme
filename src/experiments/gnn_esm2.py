"""
Siamese GNN over ESMFold contact graphs with per-residue ESM2 node features.

Shared GNN encoder processes mutant and wildtype graphs independently; their
graph-level embeddings are concatenated and passed through an MLP head.

Requires:
  python preprocess.py mutant_pairs
  python preprocess.py esmfold_structures
  python preprocess.py esm2_embeddings_pairs --batch-size 4
"""
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torch_geometric.data import Data, Batch
from torch_geometric.nn import SAGEConv, global_mean_pool
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
from Bio.PDB import PDBParser

from src.dataset import Dataset
from src.preprocessing.structures import STRUCTURES, _wt_hash

PROCESSED = Path(__file__).parent.parent.parent / 'data' / '02_processed'
FEATURES = Path(__file__).parent.parent.parent / 'data' / '03_features'

CONFIGS = {
    'gnn_esm2': {
        'model_name': 'facebook/esm2_t33_650M_UR50D',
        'ca_cutoff': 8.0,
        'hidden_dim': 256,
        'num_layers': 3,
        'dropout': 0.3,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'epochs': 50,
        'patience': 5,
        'batch_size': 32,
    },
}

SWEEP = {
    'gnn_esm2': {
        'hidden_dim': {'type': 'categorical', 'values': [128, 256, 512]},
        'num_layers': {'type': 'categorical', 'values': [2, 3, 4]},
        'dropout': {'type': 'float', 'low': 0.1, 'high': 0.5},
        'lr': {'type': 'float', 'low': 1e-4, 'high': 1e-2, 'log': True},
        'weight_decay': {'type': 'float', 'low': 1e-5, 'high': 1e-2, 'log': True},
        'batch_size': {'type': 'categorical', 'values': [16, 32, 64]},
    },
}


def _device():
    if torch.xpu.is_available():
        return torch.device('xpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def _ca_coords(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    return np.array(
        [res['CA'].get_vector().get_array()
         for res in structure.get_residues() if res.has_id('CA')],
        dtype=np.float32,
    )


def _build_graph(pdb_path, esm2_emb, ca_cutoff):
    coords = _ca_coords(pdb_path)
    n = min(len(coords), len(esm2_emb))
    coords = coords[:n]
    diff = coords[:, None] - coords[None, :]
    dists = np.sqrt((diff ** 2).sum(-1))
    ii, jj = np.where((dists < ca_cutoff) & (dists > 0))
    return Data(
        x=torch.tensor(esm2_emb[:n], dtype=torch.float32),
        edge_index=torch.tensor(np.stack([ii, jj]), dtype=torch.long),
    )


def _load_graphs(pairs_df, h5_mut_path, h5_wt_path, mut_pdb_fn, wt_pdb_fn, ca_cutoff):
    graphs_mut, graphs_wt, labels = [], [], []
    skipped = 0
    with h5py.File(h5_mut_path, 'r') as fm, h5py.File(h5_wt_path, 'r') as fw:
        for _, row in pairs_df.iterrows():
            sid = str(row['seq_id'])
            wt_key = _wt_hash(row['wildtype_sequence'])
            mut_pdb = mut_pdb_fn(row)
            wt_pdb = wt_pdb_fn(row)
            if not (mut_pdb.exists() and wt_pdb.exists()) or sid not in fm or wt_key not in fw:
                skipped += 1
                continue
            try:
                graphs_mut.append(_build_graph(mut_pdb, fm[sid][:], ca_cutoff))
                graphs_wt.append(_build_graph(wt_pdb, fw[wt_key][:], ca_cutoff))
                labels.append(float(row['tm']))
            except Exception as e:
                print(f'    skipping {sid}: {e}')
                skipped += 1
    if skipped:
        print(f'    skipped {skipped} pairs (missing PDB or embeddings)')
    return graphs_mut, graphs_wt, np.array(labels, dtype=np.float32)


def _load_test_graphs(test_df, h5_mut_path, h5_wt_path, ca_cutoff):
    wt_pdb = STRUCTURES / 'test_wildtype.pdb'
    with h5py.File(h5_wt_path, 'r') as fw:
        wt_emb = fw['wildtype'][:]
    wt_graph = _build_graph(wt_pdb, wt_emb, ca_cutoff)

    graphs_mut, seq_ids = [], []
    skipped = 0
    with h5py.File(h5_mut_path, 'r') as fm:
        for _, row in test_df.iterrows():
            sid = str(row['seq_id'])
            mut_pdb = STRUCTURES / f'test_mutant_{sid}.pdb'
            if not mut_pdb.exists() or sid not in fm:
                skipped += 1
                continue
            try:
                graphs_mut.append(_build_graph(mut_pdb, fm[sid][:], ca_cutoff))
                seq_ids.append(int(row['seq_id']))
            except Exception as e:
                print(f'    skipping {sid}: {e}')
                skipped += 1
    if skipped:
        print(f'    skipped {skipped} test sequences (missing PDB or embeddings)')
    graphs_wt = [wt_graph] * len(graphs_mut)
    return graphs_mut, graphs_wt, seq_ids


class _PairDataset(TorchDataset):
    def __init__(self, graphs_mut, graphs_wt, labels):
        self.graphs_mut = graphs_mut
        self.graphs_wt = graphs_wt
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.graphs_mut[idx], self.graphs_wt[idx], self.labels[idx]


def _collate(batch):
    muts, wts, labels = zip(*batch)
    return Batch.from_data_list(muts), Batch.from_data_list(wts), torch.stack(labels)


class _GNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * num_layers
        self.convs = nn.ModuleList([SAGEConv(dims[i], dims[i+1]) for i in range(num_layers)])
        self.acts = nn.ModuleList([nn.ReLU() for _ in range(num_layers)])
        self.drops = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _encode(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, act, drop in zip(self.convs, self.acts, self.drops):
            x = drop(act(conv(x, edge_index)))
        return global_mean_pool(x, batch)

    def forward(self, mut_data, wt_data):
        return self.head(
            torch.cat([self._encode(mut_data), self._encode(wt_data)], dim=-1)
        ).squeeze(-1)


def _cache_paths(params):
    slug = params['model_name'].split('/')[-1]
    return {
        'train_mutant': FEATURES / f'{slug}_train_mutant.h5',
        'train_wildtype': FEATURES / f'{slug}_train_wildtype.h5',
        'val_mutant': FEATURES / f'{slug}_val_mutant.h5',
        'val_wildtype': FEATURES / f'{slug}_val_wildtype.h5',
        'test_mutant': FEATURES / f'{slug}_test_mutant.h5',
        'test_wildtype': FEATURES / f'{slug}_test_wildtype.h5',
    }


def _load_pairs():
    pairs_path = PROCESSED / 'train_mutant_pairs.csv'
    if not pairs_path.exists():
        raise FileNotFoundError('train_mutant_pairs.csv not found. Run: python preprocess.py mutant_pairs')
    pairs = pd.read_csv(pairs_path)
    return train_test_split(pairs, test_size=0.1, random_state=42)


def fit(dataset: Dataset, params: dict) -> None:
    paths = _cache_paths(params)
    missing = [k for k, p in paths.items() if not p.exists() and 'test' not in k]
    if missing:
        raise FileNotFoundError(f'Missing features: {missing}. Run: python preprocess.py esm2_embeddings_pairs')

    train_pairs, val_pairs = _load_pairs()
    ca_cutoff = float(params['ca_cutoff'])

    print('Building train graphs...')
    g_mut_tr, g_wt_tr, y_tr = _load_graphs(
        train_pairs,
        paths['train_mutant'], paths['train_wildtype'],
        mut_pdb_fn=lambda r: STRUCTURES / f'train_mutant_{r["seq_id"]}.pdb',
        wt_pdb_fn=lambda r: STRUCTURES / f'train_wildtype_{_wt_hash(r["wildtype_sequence"])}.pdb',
        ca_cutoff=ca_cutoff,
    )
    print('Building val graphs...')
    g_mut_val, g_wt_val, y_val_raw = _load_graphs(
        val_pairs,
        paths['val_mutant'], paths['val_wildtype'],
        mut_pdb_fn=lambda r: STRUCTURES / f'train_mutant_{r["seq_id"]}.pdb',
        wt_pdb_fn=lambda r: STRUCTURES / f'train_wildtype_{_wt_hash(r["wildtype_sequence"])}.pdb',
        ca_cutoff=ca_cutoff,
    )

    scaler = StandardScaler()
    y_train = scaler.fit_transform(y_tr.reshape(-1, 1)).flatten().astype(np.float32)
    y_val = scaler.transform(y_val_raw.reshape(-1, 1)).flatten().astype(np.float32)
    params['_scaler'] = scaler

    dev = _device()
    in_dim = g_mut_tr[0].x.shape[1]
    gnn = _GNN(in_dim, int(params['hidden_dim']), int(params['num_layers']), float(params['dropout'])).to(dev)
    opt = torch.optim.AdamW(gnn.parameters(), lr=float(params['lr']), weight_decay=float(params.get('weight_decay', 0.0)))
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(
        _PairDataset(g_mut_tr, g_wt_tr, y_train),
        batch_size=int(params['batch_size']), shuffle=True, collate_fn=_collate,
    )
    val_loader = DataLoader(
        _PairDataset(g_mut_val, g_wt_val, y_val),
        batch_size=int(params['batch_size']), shuffle=False, collate_fn=_collate,
    )

    best_val_loss = float('inf')
    best_weights = None
    no_improve = 0
    train_losses, val_losses = [], []

    for epoch in range(int(params['epochs'])):
        gnn.train()
        total = 0.0
        for xm, xw, yb in train_loader:
            xm, xw, yb = xm.to(dev), xw.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = loss_fn(gnn(xm, xw), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(yb)
        train_loss = total / len(g_mut_tr)

        gnn.eval()
        val_total = 0.0
        with torch.no_grad():
            for xm, xw, yb in val_loader:
                xm, xw, yb = xm.to(dev), xw.to(dev), yb.to(dev)
                val_total += loss_fn(gnn(xm, xw), yb).item() * len(yb)
        val_loss = val_total / len(g_mut_val)

        train_losses.append(round(train_loss, 6))
        val_losses.append(round(val_loss, 6))
        print(f'  epoch {epoch+1:02d}/{params["epochs"]}  train={train_loss:.4f}  val={val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.clone() for k, v in gnn.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= int(params['patience']):
                print(f'  early stopping at epoch {epoch+1}')
                break

    gnn.load_state_dict(best_weights)
    gnn.eval()
    params['_gnn'] = gnn
    params['_history'] = {'train_loss': train_losses, 'val_loss': val_losses}


def predict(dataset: Dataset, params: dict) -> np.ndarray:
    paths = _cache_paths(params)
    ca_cutoff = float(params['ca_cutoff'])

    print('Building test graphs...')
    graphs_mut, graphs_wt, seq_ids = _load_test_graphs(
        dataset.test, paths['test_mutant'], paths['test_wildtype'], ca_cutoff,
    )

    dev = _device()
    gnn = params['_gnn'].to(dev)
    gnn.eval()

    all_preds = []
    batch_size = int(params['batch_size'])
    for i in range(0, len(graphs_mut), batch_size):
        xm = Batch.from_data_list(graphs_mut[i:i+batch_size]).to(dev)
        xw = Batch.from_data_list(graphs_wt[i:i+batch_size]).to(dev)
        with torch.no_grad():
            all_preds.append(gnn(xm, xw).cpu().numpy())
    norm_preds = np.concatenate(all_preds)
    preds = params['_scaler'].inverse_transform(norm_preds.reshape(-1, 1)).flatten()

    out = np.full(len(dataset.test), np.nan)
    sid_to_pos = {int(sid): i for i, sid in enumerate(dataset.test['seq_id'])}
    for pred, sid in zip(preds, seq_ids):
        out[sid_to_pos[sid]] = pred
    return out
