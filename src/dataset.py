import pandas as pd
from dataclasses import dataclass
from pathlib import Path

RAW = Path(__file__).parent.parent / 'data' / '01_raw'


@dataclass
class Dataset:
    train: pd.DataFrame
    test: pd.DataFrame

    @property
    def wildtype(self) -> str:
        return self.test.loc[self.test['protein_sequence'].str.len().idxmin(), 'protein_sequence']


def load() -> Dataset:
    train = pd.read_csv(RAW / 'train_raw.csv').dropna(subset=['protein_sequence', 'tm'])
    test = pd.read_csv(RAW / 'test.csv').merge(
        pd.read_csv(RAW / 'test_labels.csv'), on='seq_id'
    )
    return Dataset(train=train.reset_index(drop=True), test=test.reset_index(drop=True))
