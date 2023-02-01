import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


class TCGADataset(Dataset):
    """Dataset class to load data from a csv file."""

    def __init__(
        self,
        dataset_path: str,
        label_path: str,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ) -> None:
        self.dataset = np.asarray(pd.read_csv(dataset_path, index_col=0))
        self.dataset = torch.tensor(self.dataset, dtype=torch.float).to(device)
        label = pd.read_csv(label_path, index_col=0)
        self.label_embedder = LabelEncoder()
        self.label_embedder.fit(label)
        self.label = torch.from_numpy(self.label_embedder.transform(label))

    def __len__(self):
        """Get length of the dataset."""
        return len(self.dataset)

    def __getitem__(self, index):
        """Gets item from the dataset at the given index."""

        if torch.is_tensor(index):
            index = index.tolist()

        return self.dataset[index, :], self.label[index]
