from Bio import SeqIO
import torch

# Define valid characters for each biological sequence type
dna_alphabet = "ACGT"
rna_alphabet = "ACGU"
protein_alphabet = "ACDEFGHIKLMNPQRSTVWY"

def one_hot_encode(seq, alphabet, max_len):
    """
    Converts a biological sequence (DNA, RNA, or protein) into a flattened one-hot encoded tensor.
    
    Parameters:
        seq (str): Input biological sequence.
        alphabet (str): Set of valid characters for the sequence type.
        max_len (int): Maximum sequence length to encode. Shorter sequences will be zero-padded.
    
    Returns:
        torch.Tensor: A 1D tensor representing the one-hot encoded sequence.
    """
    # Initialize a 2D tensor of shape (max_len x alphabet_size) filled with zeros
    tensor = torch.zeros(max_len, len(alphabet))
    # Create a lookup table mapping each character to its index in the alphabet
    char_to_index = {char: i for i, char in enumerate(alphabet)}
    # For each position in the sequence, set the corresponding one-hot value to 1
    for i, base in enumerate(seq):
        if i < max_len and base in char_to_index:
            tensor[i, char_to_index[base]] = 1
    # Flatten the 2D tensor into a 1D tensor before returning the tensor
    return tensor.flatten()

def load_fasta_as_tensor(filepath, alphabet):
    """
    Loads a FASTA file and converts all sequences to a stacked tensor of one-hot encodings.
    
    Parameters:
        filepath (str): Path to the input FASTA file.
        alphabet (str): Alphabet to use for one-hot encoding (DNA, RNA, or protein).
    
    Returns:
        torch.Tensor: A 2D tensor where each row is a flattened one-hot encoded sequence.
    """
    sequences = []
    # Read sequences from the FASTA file and convert them to uppercase strings
    raw_seqs = [str(record.seq).upper() for record in SeqIO.parse(filepath, "fasta")]
    # Determine the maximum sequence length in the file for consistent encoding with padding
    max_len = max(len(seq) for seq in raw_seqs)
    # One-hot encode each sequence and collect into a list
    for seq in raw_seqs:
        encoded = one_hot_encode(seq, alphabet, max_len)
        sequences.append(encoded)
    # Stack the list of 1D tensors into a single 2D tensor (batch_size x input_size)
    return torch.stack(sequences)

def load_all_modalities(dna_path, rna_path, protein_path):
    """
    Loads and encodes DNA, RNA, and protein sequences from FASTA files.
    
    Parameters:
        dna_path (str): File path to the DNA FASTA file.
        rna_path (str): File path to the RNA FASTA file.
        protein_path (str): File path to the protein FASTA file.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: One-hot encoded tensors for DNA, RNA, and protein sequences.
    """
    # Load each modality using the appropriate alphabet and return as a tuple
    x_dna = load_fasta_as_tensor(dna_path, dna_alphabet)
    x_rna = load_fasta_as_tensor(rna_path, rna_alphabet)
    x_protein = load_fasta_as_tensor(protein_path, protein_alphabet)
    return x_dna, x_rna, x_protein

def get_train_loader(filepath, alphabet, batch_size=4, max_len=50):
    """
    Loads a FASTA file, one-hot encodes the sequences, and returns a DataLoader.
    For now, labels are random binary values.
    """
    data_tensor = load_fasta_as_tensor(filepath, alphabet, max_len)

    # Dummy binary labels (replace with real labels when available)
    labels_tensor = torch.randint(0, 2, (data_tensor.size(0), 1)).float()

    dataset = TensorDataset(data_tensor, labels_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader