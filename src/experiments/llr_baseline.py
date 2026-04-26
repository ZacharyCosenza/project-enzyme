"""
LLR Baseline — zero-shot ESM2 masked marginal log-likelihood ratio.

No training data used. For each test sequence, scores the evolutionary
plausibility of each mutation relative to the wildtype:
    score = Σ_i [ log p(mut_aa_i | context) − log p(wt_aa_i | context) ]

Negative score → evolutionarily unusual → likely destabilizing.
"""
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm

from src.dataset import Dataset

WILDTYPE = (
    'VPVNPEPDATSVENVALKTGSGDSQSDPIKADLEVKGQSALPFDVDCWAILCKGAPNVLQRVNEKTKNSNRDRSGANK'
    'GPFKDPQKWGIKALPPKNPSWSAQDFKSPEEYAFASSLQGGTNAILAPVNLASQNSQGGVLNGFYSANKVAQFDPSKP'
    'QQTKGTWFQITKFTGAAGPYCKALGSNDKSVCDKNKNIAGDWGFDPAKWAYQYDEKNNKFNYVGK'
)

PARAMS = {
    'wildtype': WILDTYPE,
    'model_name': 'facebook/esm2_t33_650M_UR50D',
}

PARAMS_ESM1V = {
    'wildtype': WILDTYPE,
    'model_name': 'facebook/esm1v_t33_650M_UR90S_1',
}

CONFIGS = {
    'llr_baseline':       PARAMS,
    'llr_baseline_esm1v': PARAMS_ESM1V,
}


def _device() -> torch.device:
    if torch.xpu.is_available():
        return torch.device('xpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def _load_model(params: dict):
    dev = _device()
    tokenizer = AutoTokenizer.from_pretrained(params['model_name'])
    model = AutoModelForMaskedLM.from_pretrained(params['model_name']).to(dev)
    model.eval()
    return tokenizer, model, dev


def fit(dataset: Dataset, params: dict) -> None:
    pass  # zero-shot: no training


def predict(dataset: Dataset, params: dict) -> np.ndarray:
    wildtype = params['wildtype']
    sequences = dataset.test['protein_sequence'].tolist()
    tokenizer, model, dev = _load_model(params)

    wt_len = len(wildtype)
    scores = np.zeros(len(sequences), dtype=np.float32)
    inputs_wt = tokenizer(wildtype, return_tensors='pt', truncation=True, max_length=1024).to(dev)

    for idx, seq in enumerate(tqdm(sequences, desc='LLR scoring')):
        if len(seq) != wt_len:
            scores[idx] = np.nan
            continue
        for pos in [i for i, (w, m) in enumerate(zip(wildtype, seq)) if w != m]:
            masked = inputs_wt['input_ids'].clone()
            masked[0, pos + 1] = tokenizer.mask_token_id
            with torch.no_grad():
                logits = model(**{**inputs_wt, 'input_ids': masked}).logits
            log_probs = F.log_softmax(logits[0, pos + 1], dim=-1)
            wt_tok  = tokenizer.convert_tokens_to_ids(wildtype[pos])
            mut_tok = tokenizer.convert_tokens_to_ids(seq[pos])
            scores[idx] += (log_probs[mut_tok] - log_probs[wt_tok]).item()

    return scores
