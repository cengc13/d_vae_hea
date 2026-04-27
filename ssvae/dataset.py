import numpy as np
import torch
from torch.utils.data import Dataset


class HEAFeatureDataset(Dataset):
    """HEA dataset with composition vectors, engineered features, and phase labels."""

    def __init__(self, df, transform=None):
        self.data = df
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        labels = np.array(self.data.iloc[idx]["Class"], np.float32)
        data = np.array(self.data.iloc[idx]["Fe":"Sc"], np.float32)
        data_engineered = np.array(self.data.iloc[idx]["k":"delta_h_mix"], np.float32)
        if self.transform:
            data = self.transform(data)
        return (
            torch.tensor(data * 100),
            torch.tensor(data_engineered),
            torch.tensor(labels).unsqueeze(-1),
        )


class HEAFeatureDatasetUnlabelled(Dataset):
    """HEA dataset with composition vectors and engineered features only (no labels)."""

    def __init__(self, df, transform=None):
        self.data = df
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = np.array(self.data.iloc[idx]["Fe":"Sc"], np.float32)
        data_engineered = np.array(self.data.iloc[idx]["k":"delta_h_mix"], np.float32)
        if self.transform:
            data = self.transform(data)
        return torch.tensor(data * 100), torch.tensor(data_engineered)
