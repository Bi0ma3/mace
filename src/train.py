"""
Training loop for MACE model.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import MACEModel
from data_loader import load_all_modalities
import torch

def train_model():
    print("Initialized MACE model:\n")
    
    # Load data paths
    base = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    dna_path = os.path.join(base, 'dna_sample.fasta')
    rna_path = os.path.join(base, 'rna_sample.fasta')
    protein_path = os.path.join(base, 'protein_sample.fasta')

    x_dna, x_rna, x_protein = load_all_modalities(dna_path, rna_path, protein_path)
    
    input_size = x_dna.shape[1] + x_rna.shape[1] + x_protein.shape[1]
  # model concatenates the DNA, RNA, and protein tensors before passing into the fc1 layer. 
  # So the linear layer needs to accept all of them as input.
    print("Input feature size:", input_size)
    
    model = MACEModel(input_size=input_size, hidden_size=64, output_size=1)
    print(model)

    output = model(x_dna, x_rna, x_protein)
    print("Model output:\n", output)
    
print("✅ Done running train_model()\n")

if __name__ == "__main__":
        train_model()
