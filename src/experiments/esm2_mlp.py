"""
ESM2 experiments — frozen backbone, LoRA, and late-layer fine-tuning.

  esm2_mlp      frozen ESM2, MLP head on pre-computed embeddings
  esm2_lora     LoRA adapters on Q/K/V, sequences through ESM2 each step
  esm2_finetune top-N layers unfrozen, sequences through ESM2 each step

Preprocessing (frozen only):
  python run.py --model esm2_mlp --preprocess
  python run.py --model esm2_mlp --name <name>
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset as TorchDataset
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model
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
        'mode': 'frozen',
        'hidden_dims': [512, 128],
        'dropout': 0.1,
        'lr': 1e-3,
        'epochs': 50,
        'patience': 5,
        'batch_size': 512,
        'embed_batch_size': 4,
    },
    'esm2_lora': {
        'wildtype': WILDTYPE,
        'model_name': 'facebook/esm2_t33_650M_UR50D',
        'mode': 'lora',
        'lora_r': 16,
        'lora_alpha': 32,
        'lora_dropout': 0.05,
        'target_modules': ['query', 'key', 'value'],
        'hidden_dims': [512, 128],
        'dropout': 0.1,
        'lr_model': 2e-4,
        'lr_head': 1e-3,
        'epochs': 20,
        'patience': 3,
        'batch_size': 4,
        'grad_accum': 8,
    },
    'esm2_finetune': {
        'wildtype': WILDTYPE,
        'model_name': 'facebook/esm2_t33_650M_UR50D',
        'mode': 'finetune',
        'unfreeze_last_n': 6,
        'hidden_dims': [512, 128],
        'dropout': 0.1,
        'lr_model': 1e-5,
        'lr_head': 1e-3,
        'epochs': 20,
        'patience': 3,
        'batch_size': 4,
        'grad_accum': 8,
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


class _SeqDataset(TorchDataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


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


def _val_loss(model, mlp, sequences, tm_norm, tokenizer, dev, batch_size, loss_fn):
    model.eval()
    mlp.eval()
    total, n = 0.0, 0
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        y = torch.tensor(tm_norm[i:i+batch_size], dtype=torch.float32).to(dev)
        inputs = tokenizer(batch, return_tensors='pt', padding=True,
                           truncation=True, max_length=1024).to(dev)
        with torch.no_grad():
            pooled = _pool(model(**inputs).last_hidden_state, inputs['attention_mask'])
        total += loss_fn(mlp(pooled), y).item() * len(batch)
        n += len(batch)
    return total / n


def _cache_paths(params):
    slug = params['model_name'].split('/')[-1]
    return PROCESSED / f'{slug}_train.npy', PROCESSED / f'{slug}_val.npy'


def _load_esm(params):
    dev = _device()
    tokenizer = AutoTokenizer.from_pretrained(params['model_name'])
    model = AutoModel.from_pretrained(params['model_name']).to(dev).eval()
    return tokenizer, model, dev


def _prepare_model(params, dev):
    tokenizer = AutoTokenizer.from_pretrained(params['model_name'])
    model = AutoModel.from_pretrained(params['model_name'])
    mode = params['mode']

    if mode == 'lora':
        cfg = LoraConfig(
            r=params['lora_r'],
            lora_alpha=params['lora_alpha'],
            lora_dropout=params['lora_dropout'],
            target_modules=params['target_modules'],
            bias='none',
        )
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()
    elif mode == 'finetune':
        for p in model.parameters():
            p.requires_grad = False
        for layer in model.encoder.layer[-params['unfreeze_last_n']:]:
            for p in layer.parameters():
                p.requires_grad = True
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(f'  trainable ESM2 params: {n_train:,} / {n_total:,}')

    return tokenizer, model.to(dev)


def _make_optimizer(model, mlp, params):
    if params['mode'] == 'frozen':
        return torch.optim.AdamW(mlp.parameters(), lr=params['lr'])
    return torch.optim.AdamW([
        {'params': [p for p in model.parameters() if p.requires_grad], 'lr': params['lr_model']},
        {'params': mlp.parameters(), 'lr': params['lr_head']},
    ])


def fit(dataset: Dataset, params: dict) -> None:
    mode = params['mode']

    scaler = StandardScaler()
    tm_train = scaler.fit_transform(dataset.train['tm'].values.reshape(-1, 1)).flatten().astype(np.float32)
    tm_val = scaler.transform(dataset.val['tm'].values.reshape(-1, 1)).flatten().astype(np.float32)
    params['_scaler'] = scaler

    loss_fn = nn.MSELoss()
    best_val_loss = float('inf')
    best_mlp_weights = None
    no_improve = 0
    train_losses, val_losses = [], []

    if mode == 'frozen':
        train_path, val_path = _cache_paths(params)
        if not (train_path.exists() and val_path.exists()):
            raise FileNotFoundError(
                f'Cached embeddings not found. Run: python preprocess.py esm2_embeddings'
            )
        print('Loading cached embeddings...')
        X_train = torch.tensor(np.load(train_path))
        X_val = torch.tensor(np.load(val_path))

        y_train = torch.tensor(tm_train)
        y_val = torch.tensor(tm_val)
        loader = DataLoader(TensorDataset(X_train, y_train), batch_size=params['batch_size'], shuffle=True)
        mlp = _MLP(X_train.shape[1], params['hidden_dims'], params['dropout'])
        opt = _make_optimizer(None, mlp, params)

        for epoch in range(params['epochs']):
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
                best_mlp_weights = {k: v.clone() for k, v in mlp.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= params['patience']:
                    print(f'  early stopping at epoch {epoch+1}')
                    break

        mlp.load_state_dict(best_mlp_weights)
        mlp.eval()
        params['_mlp'] = mlp

    else:  # lora or finetune
        dev = _device()
        tokenizer, model = _prepare_model(params, dev)

        def collate(batch):
            seqs, labels = zip(*batch)
            inputs = tokenizer(list(seqs), return_tensors='pt', padding=True,
                               truncation=True, max_length=1024)
            return inputs['input_ids'], inputs['attention_mask'], torch.stack(list(labels))

        loader = DataLoader(
            _SeqDataset(dataset.train['protein_sequence'].tolist(), tm_train),
            batch_size=params['batch_size'], shuffle=True, collate_fn=collate,
        )
        mlp = _MLP(model.config.hidden_size, params['hidden_dims'], params['dropout']).to(dev)
        opt = _make_optimizer(model, mlp, params)
        grad_accum = params['grad_accum']
        val_seqs = dataset.val['protein_sequence'].tolist()
        best_model_weights = None

        for epoch in range(params['epochs']):
            model.train()
            mlp.train()
            opt.zero_grad()
            total = 0.0

            for step, (ids, mask, y) in enumerate(tqdm(loader, desc=f'epoch {epoch+1}', leave=False)):
                ids, mask, y = ids.to(dev), mask.to(dev), y.to(dev)
                pooled = _pool(model(input_ids=ids, attention_mask=mask).last_hidden_state, mask)
                loss = loss_fn(mlp(pooled), y) / grad_accum
                loss.backward()
                total += loss.item() * grad_accum * len(y)
                if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
                    opt.step()
                    opt.zero_grad()

            train_loss = total / len(dataset.train)
            val_loss = _val_loss(model, mlp, val_seqs, tm_val, tokenizer, dev, params['batch_size'] * 2, loss_fn)

            train_losses.append(round(train_loss, 6))
            val_losses.append(round(val_loss, 6))
            print(f'  epoch {epoch+1:02d}/{params["epochs"]}  train={train_loss:.4f}  val={val_loss:.4f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_mlp_weights = {k: v.clone() for k, v in mlp.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= params['patience']:
                    print(f'  early stopping at epoch {epoch+1}')
                    break

        model.load_state_dict(best_model_weights)
        mlp.load_state_dict(best_mlp_weights)
        model.eval()
        mlp.eval()
        params['_model'] = model
        params['_tokenizer'] = tokenizer
        params['_dev'] = dev
        params['_mlp'] = mlp

    params['_history'] = {'train_loss': train_losses, 'val_loss': val_losses}


def predict(dataset: Dataset, params: dict) -> np.ndarray:
    mlp = params['_mlp']
    scaler = params['_scaler']
    sequences = dataset.test['protein_sequence'].tolist()

    if params['mode'] == 'frozen':
        if '_esm' not in params:
            params['_esm'] = _load_esm(params)
        tokenizer, model, dev = params['_esm']
        embeddings = _embed(sequences, tokenizer, model, dev, params['embed_batch_size'])
        with torch.no_grad():
            norm_preds = mlp(torch.tensor(embeddings)).numpy()
    else:
        model = params['_model']
        tokenizer = params['_tokenizer']
        dev = params['_dev']
        embeddings = []
        for i in range(0, len(sequences), params['batch_size']):
            batch = sequences[i:i+params['batch_size']]
            inputs = tokenizer(batch, return_tensors='pt', padding=True,
                               truncation=True, max_length=1024).to(dev)
            with torch.no_grad():
                pooled = _pool(model(**inputs).last_hidden_state, inputs['attention_mask'])
            embeddings.append(pooled.cpu().float().numpy())
        with torch.no_grad():
            norm_preds = mlp(torch.tensor(np.vstack(embeddings))).numpy()

    return scaler.inverse_transform(norm_preds.reshape(-1, 1)).flatten()
