from torch.utils.data import Dataset

class ProteinDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform or []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.data[idx][:, :42]
        labels = self.data[idx][:, 42:]
        sample = (features, labels)
        for t in self.transform:
            sample = t(sample)
        return sample  # (inputs, lengths, mask, tags)
