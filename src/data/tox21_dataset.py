import torch
from torch.utils.data import Dataset
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

def smiles_to_ecfp6(smiles, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=n_bits)

class Tox21Dataset(Dataset):
    def __init__(self, csv_path, split="train", max_train_samples=None, seed=42):
        df = pd.read_csv(csv_path)

        # Convert SMILES to ECFP6 fingerprints
        df['ecfp6'] = df['smiles'].apply(smiles_to_ecfp6)
        df = df[df['ecfp6'].notnull()]

        # Identify label columns (all except these)
        non_label_cols = {'Unnamed: 0', 'ID', 'inchikey', 'sdftitle', 'order', 'set', 'CVfold', 'smiles', 'ecfp6'}
        y_cols = [c for c in df.columns if c not in non_label_cols]

        # Filter by split (train or test or val)
        split_map = {"train": "training", "val": "validation", "test": "test"}
        filtered = df[df['set'] == split_map[split]].copy()

        # Subsample train set if requested
        if split == "train" and max_train_samples is not None:
            filtered = filtered.sample(n=min(max_train_samples, len(filtered)), random_state=seed).reset_index(drop=True)

        # Store X (features) and y (multi-label targets)
        self.X = np.array([np.array(fp) for fp in filtered['ecfp6']])
        self.y = filtered[y_cols].fillna(0).astype(np.float32).to_numpy()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)
