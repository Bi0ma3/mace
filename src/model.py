import torch.nn as nn
import torch

class MACEModel(nn.Module):
    def __init__(self, input_size=128, hidden_size=64, output_size=1):
        super(MACEModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, dna, rna, protein):
        x = torch.cat([dna, rna, protein], dim=1)  # concatenate features
        x = self.relu(self.fc1(x))
        return self.fc2(x)
