import torch
import torchvision
import torchvision.transforms as transforms
from DenseNet import *

batch_size = 10
nr_classes = 10
depth = 100

densenet = DenseNet(depth , nr_classes)
densenet.cuda()

testset = torchvision.datasets.SVHN(root='.', split='test', download=True, transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

correct = 0
total = 0

for data in testloader:
    images, labels = data
    images = images.cuda()
    labels = labels.cuda()
    outputs = densenet(images)
    _, predicted = torch.max(outputs.cuda().data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
        
print('Accuracy of the network on the test images = %d %%' % (100 * correct / total))