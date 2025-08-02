import torch
import numpy as np

class CreateMask:
    """
    All sequences are padded to 700, so this function creates a mask so that the model only learns from valid/real residues
    """
    def __call__(self, sample):
        x, y = sample
        mask = np.any(x != 0, axis=1).astype(np.bool_)  # Shape: (seq_len,)
        return x, y, mask

class OneHotToLabel:
    def __call__(self, sample):
        x, y, mask = sample
        if y.ndim == 2:  # [seq_len, num_classes]
            y = np.argmax(y, axis=1)
        return x, y, mask

class ToTensor:
    def __call__(self, sample):
        x, y, mask = sample
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        mask_tensor = torch.tensor(mask, dtype=torch.bool)
        lengths = mask_tensor.sum(dim=0) #true length for sequence (# of real residues)
        return x_tensor, lengths, mask_tensor, y_tensor
