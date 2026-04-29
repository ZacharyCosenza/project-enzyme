"""
Cluster training sequences into wildtype-mutant groups via Hamming distance.
The derived wildtype per cluster is the sequence with the highest Tm.

Output: data/02_processed/train_mutant_pairs.csv
  Columns: seq_id, protein_sequence, pH, data_source, tm, wildtype_sequence

Usage:
  python preprocess.py mutant_pairs
  python preprocess.py mutant_pairs --max-mutations 5
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

RAW = Path(__file__).parent.parent.parent / 'data' / '01_raw'
PROCESSED = Path(__file__).parent.parent.parent / 'data' / '02_processed'


def run(args):
    train = pd.read_csv(RAW / 'train_raw.csv').dropna(subset=['protein_sequence', 'tm']).copy()
    train['seq_len'] = train['protein_sequence'].str.len()
    print(f'Loaded {len(train)} sequences')

    max_mut = args.max_mutations
    all_edges = []
    for length, group in train.groupby('seq_len'):
        if len(group) < 2:
            continue
        seqs = group['protein_sequence'].tolist()
        n = len(seqs)
        mat = np.frombuffer(b''.join(s.encode('ascii') for s in seqs), dtype=np.uint8).reshape(n, length)
        dists = (cdist(mat, mat, metric='hamming') * length).astype(int)
        ii, jj = np.where((dists > 0) & (dists <= max_mut))
        mask = ii < jj
        global_idx = group.index.tolist()
        for li, lj in zip(ii[mask].tolist(), jj[mask].tolist()):
            all_edges.append((global_idx[li], global_idx[lj]))

    print(f'Found {len(all_edges)} within-threshold pairs (≤{max_mut} mutations)')

    n_seqs = len(train)
    idx_map = {v: i for i, v in enumerate(train.index)}
    if all_edges:
        rows = [idx_map[e[0]] for e in all_edges]
        cols = [idx_map[e[1]] for e in all_edges]
        d = [1] * len(all_edges)
        graph = csr_matrix((d + d, (rows + cols, cols + rows)), shape=(n_seqs, n_seqs))
        _, labels = connected_components(graph, directed=False)
    else:
        labels = np.arange(n_seqs)

    train['cluster_id'] = labels
    cluster_sizes = pd.Series(labels).value_counts()
    multi_clusters = cluster_sizes[cluster_sizes > 1]
    clustered = train[train['cluster_id'].isin(multi_clusters.index)].copy()

    # Derive wildtype per cluster: sequence with highest Tm
    wt_map = (
        clustered.loc[clustered.groupby('cluster_id')['tm'].idxmax()]
        .set_index('cluster_id')['protein_sequence']
    )
    clustered['wildtype_sequence'] = clustered['cluster_id'].map(wt_map)

    out = clustered[['seq_id', 'protein_sequence', 'pH', 'data_source', 'tm', 'wildtype_sequence']].copy()

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED / 'train_mutant_pairs.csv'
    out.to_csv(out_path, index=False)
    print(f'Clusters with >1 sequence: {len(multi_clusters)}, sequences: {len(out)}')
    print(f'Saved to {out_path}')


def register_args(parser):
    parser.add_argument('--max-mutations', type=int, default=10,
                        help='Max Hamming distance to consider two sequences mutants (default: 10)')
