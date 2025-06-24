"""
Training loop for MACE model.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import MACEModel
import torch

def train_model():
    # Create a dummy model and print its architecture
    model = MACEModel(input_size=128, hidden_size=64, output_size=1)
    print("Initialized MACE model:\n", model)

    # Fake data for a dry run
    batch_size = 2
    x_dna = torch.randn(batch_size, 128)
    x_rna = torch.randn(batch_size, 128)
    x_protein = torch.randn(batch_size, 128)

    output = model(x_dna, x_rna, x_protein)
    print("Model output:\n", output)

if __name__ == "__main__":
    train_model()
