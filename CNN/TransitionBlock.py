import torch.nn as nn

class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(TransitionBlock, self).__init__()
        self.relu = nn.ReLU(inplace = True)
        self.bn = nn.BatchNorm2d(num_features = out_channels)
        self.conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, bias = False)
        self.avg_pool = nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0)

    def forward(self, x):

        bn = self.bn(self.relu(self.conv(x)))
        out = self.avg_pool(bn)

        return out