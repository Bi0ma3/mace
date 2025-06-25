"""
Training loop for MACE model.
"""

import sys 
import os

# Add the root directory to the system path so local modules (like model and data_loader) can be imported,
# regardless of where this script is run from.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import MACEModel                    # Import the custom neural network model
from data_loader import load_all_modalities    # Function to load one-hot encoded DNA, RNA, and protein sequences
import torch

def train_model():
    """
    Loads data, initializes the MACE model, and performs a forward pass to check output shape and architecture.
    This is the initial scaffold for a full training loop.
    """
    print("Initialized MACE model:\n")
    
  # Load data paths to raw FASTA sequence files located in ../data/raw/
    base = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    dna_path = os.path.join(base, 'dna_sample.fasta')
    rna_path = os.path.join(base, 'rna_sample.fasta')
    protein_path = os.path.join(base, 'protein_sample.fasta')

  # Load all modalities: returns one-hot encoded tensors for each sequence type
    x_dna, x_rna, x_protein = load_all_modalities(dna_path, rna_path, protein_path)

  # Compute the total input size to the model by summing the flattened sizes of all three modalities.
  # This is necessary because the model expects a single concatenated input vector.  
    input_size = x_dna.shape[1] + x_rna.shape[1] + x_protein.shape[1]
  
  # Note: The MACEModel assumes concatenation of DNA, RNA, and protein features
  # into a single input vector for the first fully connected layer.
    print("Input feature size:", input_size)
    
  # Initialize the model with:
    # - input_size: total length of concatenated input vector
    # - hidden_size: number of neurons in the hidden layer (currently 64)
    # - output_size: model outputs a single prediction value (binary classification or regression)
    model = MACEModel(input_size=input_size, hidden_size=64, output_size=1)
    print(model)

  # Run a forward pass using the model (no training yet, just to verify dimensions and outputs)
    output = model(x_dna, x_rna, x_protein)
    print("Model output:\n", output)
    
# Print confirmation that the script ran to completion    
print("✅ Done running train_model()\n")

# Ensure the training routine only runs when this script is executed directly
# and not when imported as a module
if __name__ == "__main__":
        train_model()
