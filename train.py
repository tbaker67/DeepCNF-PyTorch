import os
import pathlib
import torch
from torch.utils.data import DataLoader
from DeepCNF import DeepCNF
from dataset import get_dataset, split_with_shuffle
from protein_dataset import ProteinDataset
from transforms import CreateMask, OneHotToLabel, ToTensor

def train_model():
    data_path = pathlib.Path(os.path.abspath(__file__)).parent / 'cullpdb+profile_5926.npy'
    full_data = get_dataset(data_path)

    train_data, test_data, val_data = split_with_shuffle(full_data, seed=42)

    transform_pipeline = [CreateMask(), OneHotToLabel(), ToTensor()]

    train_dataset = ProteinDataset(train_data, transform=transform_pipeline)
    test_dataset = ProteinDataset(test_data, transform=transform_pipeline)
    val_dataset = ProteinDataset(val_data, transform=transform_pipeline)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = DeepCNF()
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1e-2)

    model.train_model(train_loader, test_loader, optimizer, num_epochs=10)

if __name__ == "__main__":
    train_model()
