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
        #First fully connected layer: projects concatenated inputs to hidden space
        #self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc1 = nn.LazyLinear(hidden_size)
        # Non-linear activation function applied after first layer
        self.relu = nn.ReLU() 
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

        # Create a function to turn the input into a [B,F] tensor for the one-hot encoding 
        def to_BF(t):
            if t.dim() == 2:                        # [B, F]
                return t                            # Already in the correct format 
            elif t.dim() == 3:                      # [B, L, C]
                return t.reshape(t.size(0), -1)     # Flatten [B, L, C]  to [B, L*C] == [B, F]
            else:
                raise ValueError(f"Expected [B, F] or [B, L, C], got {tuple(t.shape)}") # Error if something else 
        
        # Now Loop through each feature (DRNA, Protein)
        for t in (dna, rna, protein):
            if t is None:
                continue                        # Let's us only pass DRNA and Protein 
            tBF = to_BF(t).contiguous()         # Ensures tensor is in shape [B, F]. Contiguous ensures that tensor is packed after slicing 

            if B is None:                       # Starting condition from above
                B = tBF.size(0)                 # Remember how many modalitites we have. Values will range from 1, 2 or 3
            elif tBF.size(0) != B:
                raise ValueError("Batch sizes differ across modalities.")
            feats.append(tBF)                   # Concat Tensors 

        if not feats:
            raise ValueError("Provide at least one of DNA, RNA or Protein Sequence") # Prevents empty list from passing through 
         


        # Concatenate all three modalities along the feature dimension (axis=1)
        x = torch.cat(feats, dim=1)
        print("[DEBUG] concatenated x.shape:", tuple(x.shape))
        print("[DEBUG] fc1 expects in_features:", self.fc1.in_features) 
        # Apply first linear transformation and activation
        x = self.relu(self.fc1(x))
        # Pass through final linear layer to generate output
        return self.fc2(x)
