import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import scipy.io as sio
import numpy as np
from matplotlib import pyplot as plt
import time
from DenseNet import *

trainset = torchvision.datasets.SVHN(root='.', split='train', download=True, transform=transforms.ToTensor())

mat = sio.loadmat('./train_32x32.mat')

a3=[]
for i in range(32):
    a2=[]
    for j in range(32):
        a1=[]
        for k in range(3):
            a1.insert(k,mat['X'][i][j][k][20000])
        a2.insert(j,a1)
    a3.insert(i,a2)

plt.figure()
plt.imshow(np.array(a3))
plt.show()

# Hyperparameters
batch_size = 10
momentum = 0.9
learning_rate = 0.01
nr_classes = 10
depth = 100
nr_epochs = 10
loss_vctr = []

# Load the model on the GPU
densenet = DenseNet(depth , nr_classes)
densenet.cuda()

#Calculating no. of parameters
param_nr = 0
for p in densenet.parameters():
    s = torch.numel(p)
    param_nr += s
print('Number of parameters to be trained = %d'  % (param_nr))

# Data Loaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

start = time.time()

# Oprimization Criteria and Optimization method
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(densenet.parameters(), lr=learning_rate, momentum=momentum, nesterov = True)

# Training Loop
print('\nStart of the Optimization Process\n')
for epoch in range(nr_epochs):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        
        # get the inputs
        inputs, labels = data
        inputs = inputs.cuda() 
        labels = labels.cuda()
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = densenet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            loss_vctr.append(running_loss / 2000)
            print('Epoch number = %d, Batch number = %5d, Loss = %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
    
print('\nFinished Optimization\n')
end = time.time() # Time counted in seconds
print('Time taken to train the model = %.2f minutes' % ((end - start)/60))

x = range(1, nr_epochs * int((73257 / batch_size) / 2000) + 1)
x_epoch = [z for z in range(1, len(x) + 1) if z % ((73257 // batch_size) // 2000) == 0] 
x_ticks_labels = ['epoch ' + str(y) for y in range(1, nr_epochs+1)]

plt.figure(figsize=(16,8))
plt.plot(x , loss_vctr, color = 'k')
plt.xticks(x_epoch, x_ticks_labels)
plt.title("Training Error Plot")
plt.ylabel("Training error loss")
plt.xlabel("Number of epochs")
plt.show()