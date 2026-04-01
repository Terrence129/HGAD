import torch
from torch.utils.data import Dataset


class SMDTrainDataset(Dataset):
    def __init__(self, windows):
        self.windows = torch.tensor(windows, dtype=torch.float32)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx]


class SMDTestDataset(Dataset):
    def __init__(self, windows, labels):
        self.windows = torch.tensor(windows, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

        if len(self.windows) != len(self.labels):
            raise ValueError(
                f"windows and labels length mismatch: "
                f"{len(self.windows)} vs {len(self.labels)}"
            )

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx]