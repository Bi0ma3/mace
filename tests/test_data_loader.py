from src.model import MaceModel

net = MaceModel()
criterion = torch.nn.BCELoss(size_average=True)   
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

num_epochs = 100
for epoch in range(num_epochs):
    for i, (inputs,labels) in enumerate (train_loader):
        inputs = Variable(inputs.float())
        labels = Variable(labels.float())
        output = net(inputs)
        optimizer.zero_grad()
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

    #Accuracy
    output = (output>0.5).float()
    correct = (output == labels).float().sum()
    print("Epoch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}".format(epoch+1,num_epochs, loss.data[0], correct/x.shape[0]))