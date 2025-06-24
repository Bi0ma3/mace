from Bio import SeqIO
import torch

dna_alphabet = "ACGT"
rna_alphabet = "ACGU"
protein_alphabet = "ACDEFGHIKLMNPQRSTVWY"

def one_hot_encode(seq, alphabet, max_len):
    tensor = torch.zeros(max_len, len(alphabet))
    char_to_index = {char: i for i, char in enumerate(alphabet)}
    for i, base in enumerate(seq):
        if i < max_len and base in char_to_index:
            tensor[i, char_to_index[base]] = 1
    return tensor.flatten()

def load_fasta_as_tensor(filepath, alphabet):
    sequences = []
    raw_seqs = [str(record.seq).upper() for record in SeqIO.parse(filepath, "fasta")]
    max_len = max(len(seq) for seq in raw_seqs)
    for seq in raw_seqs:
        encoded = one_hot_encode(seq, alphabet, max_len)
        sequences.append(encoded)
    return torch.stack(sequences)

def load_all_modalities(dna_path, rna_path, protein_path):
    x_dna = load_fasta_as_tensor(dna_path, dna_alphabet)
    x_rna = load_fasta_as_tensor(rna_path, rna_alphabet)
    x_protein = load_fasta_as_tensor(protein_path, protein_alphabet)
    return x_dna, x_rna, x_protein
