import os
import sys
import torch
from torch.autograd import Variable
import torch.nn as nn

# Add src/ to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import MaceModel
from src.data_loader import get_train_loader, protein_alphabet

# === Parameters ===
fasta_path = "data/raw/protein_test_TRP53_Musmus.faa"
max_len = 50  # update if needed
batch_size = 4

# === Data ===
train_loader = get_train_loader(fasta_path, protein_alphabet, batch_size)

# === Model ===
input_size = max_len * len(protein_alphabet)
net = MaceModel(input_size)
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

# === Training ===
num_epochs = 10
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = Variable(inputs.float())
        labels = Variable(labels.float())

        output = net(inputs)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Accuracy
    output = (output > 0.5).float()
    correct = (output == labels).float().sum()
    print("Epoch {}/{} | Loss: {:.3f} | Accuracy: {:.3f}".format(
        epoch+1, num_epochs, loss.item(), correct / inputs.shape[0]))
