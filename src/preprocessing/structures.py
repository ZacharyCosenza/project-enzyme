"""
Fetch ESMFold PDB structures for all mutant-wildtype pairs (train + test).

Idempotent — skips sequences with an existing PDB in the output directory.
Retries on 5xx/timeout with exponential backoff.

Output: data/02_processed/structures/
  train_mutant_{seq_id}.pdb
  train_wildtype_{wt_hash}.pdb   (deduplicated by sequence)
  test_mutant_{seq_id}.pdb
  test_wildtype.pdb

Usage:
  python preprocess.py esmfold_structures
  python preprocess.py esmfold_structures --delay 1.5 --retries 5
"""
import hashlib
import time
import requests
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from src import dataset as ds

PROCESSED  = Path(__file__).parent.parent.parent / 'data' / '02_processed'
STRUCTURES = PROCESSED / 'structures'
WILDTYPE = (
    'VPVNPEPDATSVENVALKTGSGDSQSDPIKADLEVKGQSALPFDVDCWAILCKGAPNVLQRVNEKTKNSNRDRSGANK'
    'GPFKDPQKWGIKALPPKNPSWSAQDFKSPEEYAFASSLQGGTNAILAPVNLASQNSQGGVLNGFYSANKVAQFDPSKP'
    'QQTKGTWFQITKFTGAAGPYCKALGSNDKSVCDKNKNIAGDWGFDPAKWAYQYDEKNNKFNYVGK'
)


API_MAX_LEN = 400


def _fetch(sequence: str, out_path: Path, timeout: int, retries: int, delay: float) -> bool:
    """Fetch one PDB. Returns True if fetched, False if already cached."""
    if out_path.exists():
        return False
    for attempt in range(retries):
        try:
            resp = requests.post(
                'https://api.esmatlas.com/foldSequence/v1/pdb/',
                data=sequence,
                timeout=timeout,
                verify=False,
            )
            resp.raise_for_status()
            if not resp.text.startswith(('HEADER', 'ATOM', 'MODEL')):
                raise ValueError(f'Non-PDB response: {resp.text[:200]}')
            out_path.write_text(resp.text)
            return True
        except (requests.HTTPError, requests.Timeout, ValueError) as e:
            if attempt < retries - 1:
                wait = delay * (2 ** attempt) * 10
                print(f'    retry {attempt + 1}/{retries - 1} after {wait:.0f}s ({e})')
                time.sleep(wait)
            else:
                print(f'    FAILED after {retries} attempts: {out_path.name} — {e}')
                return False
    return False


def _wt_hash(seq: str) -> str:
    return hashlib.md5(seq.encode()).hexdigest()[:10]


def run(args):
    STRUCTURES.mkdir(parents=True, exist_ok=True)

    pairs_path = PROCESSED / 'train_mutant_pairs.csv'
    if not pairs_path.exists():
        raise FileNotFoundError('train_mutant_pairs.csv not found. Run: python preprocess.py mutant_pairs')
    pairs = pd.read_csv(pairs_path)
    data  = ds.load()

    tasks = []

    # Train mutants
    for _, row in pairs.iterrows():
        tasks.append((row['protein_sequence'], STRUCTURES / f'train_mutant_{row["seq_id"]}.pdb'))

    # Train wildtypes (deduplicated by sequence)
    for wt_seq in pairs['wildtype_sequence'].unique():
        tasks.append((wt_seq, STRUCTURES / f'train_wildtype_{_wt_hash(wt_seq)}.pdb'))

    # Test mutants
    for _, row in data.test.iterrows():
        tasks.append((row['protein_sequence'], STRUCTURES / f'test_mutant_{row["seq_id"]}.pdb'))

    # Test wildtype
    tasks.append((WILDTYPE, STRUCTURES / 'test_wildtype.pdb'))

    already_done = sum(1 for _, p in tasks if p.exists())
    todo = [(seq, p) for seq, p in tasks if not p.exists()]
    print(f'Structures: {len(tasks)} total, {already_done} cached, {len(todo)} to fetch')

    too_long = sum(1 for seq, _ in todo if len(seq) > API_MAX_LEN)
    if too_long:
        print(f'  ({too_long} sequences exceed {API_MAX_LEN} AA and will be skipped)')

    fetched = failed = skipped = 0
    for i, (seq, out_path) in enumerate(todo):
        print(f'  [{i + 1}/{len(todo)}] {out_path.name}')
        if len(seq) > API_MAX_LEN:
            print(f'    skipping — length {len(seq)} exceeds API limit ({API_MAX_LEN})')
            skipped += 1
            continue
        ok = _fetch(seq, out_path, timeout=180, retries=args.retries, delay=args.delay)
        if ok:
            fetched += 1
            time.sleep(args.delay)
        else:
            failed += 1

    print(f'\nDone. fetched={fetched}, skipped={skipped}, failed={failed}, cached={already_done}')


def register_args(parser):
    parser.add_argument('--delay',   type=float, default=1.0, help='seconds between requests (default: 1.0)')
    parser.add_argument('--retries', type=int,   default=3,   help='retry attempts on failure (default: 3)')


COMMANDS = {
    'esmfold_structures': (run, register_args),
}
