import torch.nn as nn
import torch

class MaceModel(nn.Module):
    """
    MACEModel is a simple feedforward neural network designed to take concatenated
    DNA, RNA, and protein sequence embeddings and output a prediction (classification or regression (eh, maybe?)).
    """
    def __init__(self, input_size=128, hidden_size=64, output_size=1):
        """
        Initializes the MACE model architecture.

        Parameters:
            input_size (int): The total size of the input vector after concatenating all three modalities (DNA, RNA, Protein).
            hidden_size (int): Number of neurons in the hidden layer.
            output_size (int): Dimensionality of the model output (default is 1, e.g., for binary classification).
                Output size will change depends on number of features we choose to label (promoters, silencers, lnc-DNA etc.)
        """
        super(MaceModel, self).__init__()
        # First fully connected layer: projects concatenated inputs to hidden space
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Non-linear activation function applied after first layer
        self.relu = nn.ReLU() #We could use LeakyReLU
        # Final output layer: reduces hidden activations to the final output dimension
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, dna = None, rna = None, protein = None): #None makes the arguement optional
        """
        Defines the forward pass of the model.

        Parameters:
            [B, F] feature vectors, or 
            [B, L, C] one-hot sequences (auto-flattened to [B, L*C]).

            F = L*C = How many numbers describe one sample at the moment it goes into a linear layer 
            B = Batch size (how many samples at once).
            L = Length of the Sequence (i.e. How many Positions)
            C = Channel = alphabet size. Fixed at 4 for D/RNA and protiens

    Any modality may be None.
    Returns: [B, output_size]

        Parameters:
            dna (torch.Tensor): One-hot encoded input tensor for DNA sequences.
            rna (torch.Tensor): One-hot encoded input tensor for RNA sequences.
            protein (torch.Tensor): One-hot encoded input tensor for protein sequences.

        Returns:
            torch.Tensor: Output tensor from the model.
        """

        # Empty list to collect each type of data (DNA/RNA/Protein) per sample
        feats = [] 
         # Used as a check to ensure each type of data has the same batch size 
        B = None





        # Concatenate all three modalities along the feature dimension (axis=1)
        x = torch.cat([dna, rna, protein], dim=1) 
        # Apply first linear transformation and activation
        x = self.relu(self.fc1(x))
        # Pass through final linear layer to generate output
        return self.fc2(x)
