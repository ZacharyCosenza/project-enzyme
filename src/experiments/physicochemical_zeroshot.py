"""
Zero-shot physicochemical mutation scorer.

Sums four z-score normalised components across all mutated positions:
BLOSUM62 score, delta hydrophobicity (Kyte-Doolittle), delta charge (pH 7),
delta volume (Angstrom^3, Pontius et al. 1996).
"""
import numpy as np
from Bio.Align import substitution_matrices
from Bio.SeqUtils.ProtParamData import kd as HYDROPHOBICITY
from src.dataset import Dataset

WILDTYPE = (
    'VPVNPEPDATSVENVALKTGSGDSQSDPIKADLEVKGQSALPFDVDCWAILCKGAPNVLQRVNEKTKNSNRDRSGANK'
    'GPFKDPQKWGIKALPPKNPSWSAQDFKSPEEYAFASSLQGGTNAILAPVNLASQNSQGGVLNGFYSANKVAQFDPSKP'
    'QQTKGTWFQITKFTGAAGPYCKALGSNDKSVCDKNKNIAGDWGFDPAKWAYQYDEKNNKFNYVGK'
)

BLOSUM62 = substitution_matrices.load("BLOSUM62")

# Formal charge at pH 7 — only 5 AAs carry charge, all others 0
CHARGE = {'R': 1, 'K': 1, 'D': -1, 'E': -1, 'H': 0}

# Side-chain volume (Angstrom^3), Pontius et al. 1996
VOLUME = {
    'A': 67, 'R': 148, 'N': 96, 'D': 91, 'C': 86,
    'Q': 114, 'E': 109, 'G': 48, 'H': 118, 'I': 124,
    'L': 124, 'K': 135, 'M': 124, 'F': 135, 'P': 90,
    'S': 73, 'T': 93, 'W': 163, 'Y': 141, 'V': 105,
}

CONFIGS = {
    'physicochemical_zeroshot': {
        'wildtype': WILDTYPE,
        'weights': [0.405, 0.382, -0.091, 0.122],  # blosum, hydro, charge, volume
    },
}

SWEEP = {
    'physicochemical_zeroshot': {
        'w_blosum': {'type': 'float', 'low': 0.0, 'high': 1.0},
        'w_hydro':  {'type': 'float', 'low': 0.0, 'high': 1.0},
        'w_charge': {'type': 'float', 'low': 0.0, 'high': 1.0},
        'w_volume': {'type': 'float', 'low': 0.0, 'high': 1.0},
    },
}


def fit(dataset: Dataset, params: dict) -> None:
    pass


def predict(dataset: Dataset, params: dict) -> np.ndarray:
    wildtype = params['wildtype']
    sequences = dataset.test['protein_sequence'].tolist()
    wt_len = len(wildtype)

    raw = np.full((len(sequences), 4), np.nan)

    for idx, seq in enumerate(sequences):
        if len(seq) != wt_len:
            continue
        mutations = [(wildtype[i], seq[i]) for i in range(wt_len) if wildtype[i] != seq[i]]
        try:
            b = sum(BLOSUM62[w, m] for w, m in mutations)
        except KeyError:
            continue
        h = sum(HYDROPHOBICITY.get(m, 0) - HYDROPHOBICITY.get(w, 0) for w, m in mutations)
        c = sum(CHARGE.get(m, 0) - CHARGE.get(w, 0) for w, m in mutations)
        v = sum(VOLUME.get(m, 0) - VOLUME.get(w, 0) for w, m in mutations)
        raw[idx] = [b, h, c, v]

    valid = ~np.isnan(raw[:, 0])
    result = np.full(len(sequences), np.nan)

    components = raw[valid]
    std = components.std(axis=0)
    std[std == 0] = 1
    normalised = (components - components.mean(axis=0)) / std
    weights = np.array([
        params.get('w_blosum', params['weights'][0]),
        params.get('w_hydro',  params['weights'][1]),
        params.get('w_charge', params['weights'][2]),
        params.get('w_volume', params['weights'][3]),
    ])
    result[valid] = normalised @ weights

    return result
