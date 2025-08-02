import torch
import torch.nn as nn
from torchcrf import CRF

class DeepCNF(nn.Module):
    def __init__(self, input_dim=42, conv_layers=5, conv_dims=100, output_dim=8, window_size=11, gate_function='tanh'):
        super(DeepCNF, self).__init__()
        self.layers = nn.ModuleList()
        self.gate_function = gate_function

        prev_dim = input_dim
        padding = window_size // 2
        for _ in range(conv_layers):
            self.layers.append(nn.Conv1d(in_channels=prev_dim, out_channels=conv_dims, kernel_size=window_size, padding=padding))
            prev_dim = conv_dims

        self.output_layer = nn.Conv1d(prev_dim, output_dim, kernel_size=1)
        self.crf = CRF(output_dim, batch_first=True)

    def forward(self, x, lengths, mask, tags=None):
        x = x.transpose(1, 2) #need to put in specific order for the crf layer to work correctly
        for layer in self.layers:
            x = layer(x)
            x = getattr(torch, self.gate_function)(x)
        emissions = self.output_layer(x).transpose(1, 2) #put emmission back so we can 

        if tags is not None:
            loss = -self.crf(emissions, tags, mask=mask)
            return loss
        return self.crf.decode(emissions, mask=mask)

    def evaluate(self, data_loader):
        self.eval()
        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for inputs, lengths, mask, tags in data_loader:
                outputs = self.forward(inputs, lengths, mask)
                loss = self.forward(inputs, lengths, mask, tags)
                total_loss += loss.item()

                mask = mask.to(torch.int)
                for pred_seq, true_seq, mask_seq in zip(outputs, tags, mask):
                    true_seq = true_seq[mask_seq == 1]
                    pred_seq = torch.tensor(pred_seq)
                    correct += (pred_seq == true_seq).sum().item()
                    total += len(true_seq)
        return total_loss / len(data_loader), correct / total

    def train_model(self, train_loader, test_loader, optimizer, num_epochs=10, weight_decay=50):
        for epoch in range(num_epochs):
            self.train()
            total_loss = 0
            for inputs, lengths, masks, tags in train_loader:
                print(inputs.shape)
                def closure():
                    optimizer.zero_grad()
                    loss = self.forward(inputs, lengths, masks, tags)
                    loss += weight_decay * sum(p.norm(2) ** 2 for p in self.parameters())
                    loss.backward()
                    return loss
                loss = optimizer.step(closure)
                total_loss += loss.item()
            print(f"Epoch {epoch+1}: Train Loss = {total_loss / len(train_loader):.4f}")
            test_loss, acc = self.evaluate(test_loader)
            print(f"Test Loss = {test_loss:.4f}, Accuracy = {acc:.4f}")
