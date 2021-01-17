import torch.nn as nn
import torch.nn.functional as F
import math
from BottleneckBlock import *
from DenseBlock import *
from TransitionBlock import *

class DenseNet(nn.Module):
    def __init__(self, depth, num_classes, growth_rate=12, reduction=0.5):
        
        super(DenseNet, self).__init__()
        in_planes = 2 * growth_rate
        n = (depth - 4) / 3
        n = n/2
        block = BottleneckBlock
        n = int(n)

        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        
        # 1st block
        self.block1 = DenseBlock(n, in_planes, growth_rate, block)
        in_planes = int(in_planes+n*growth_rate)
        self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)))
        in_planes = int(math.floor(in_planes*reduction))
        
        # 2nd block
        self.block2 = DenseBlock(n, in_planes, growth_rate, block)
        in_planes = int(in_planes+n*growth_rate)
        self.trans2 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)))
        in_planes = int(math.floor(in_planes*reduction))
        
        # 3rd block
        self.block3 = DenseBlock(n, in_planes, growth_rate, block)
        in_planes = int(in_planes+n*growth_rate)
        
        # final classifier
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_planes, num_classes)
        self.in_planes = in_planes
        
    def forward(self, x):
        
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.in_planes)
        return self.fc(out)