from torch import nn
from torch.nn import functional as F

class SEBlock(nn.Module):
    def __init__(self, in_channel, reduction):
        super(SEBlock, self).__init__()
        self.global_pooling = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.feature = nn.Sequential(
            nn.Linear(in_channel, in_channel//reduction),
            nn.ReLU(),
            nn.Linear(in_channel//reduction, in_channel),
            nn.Sigmoid()
        )

    def forward(self, img):
        b, c, h, w = img.size()
        identity = img
        identity = self.global_pooling(identity).view(b,c)
        identity = self.feature(identity).view(b,c,1,1)
        return img*identity.expand_as(img)

class CommonBlock(nn.Module):
    def __init__(self, in_channel, out_channel, se=None):
        super(CommonBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel)
        )
        self.se = se

    def forward(self, img):
        identity = img
        out = self.conv1(img)
        out = self.conv2(out)
        if self.se != None:
            out = self.se(out)
        out += identity
        return F.relu(out)


class BlockWithDownSample(nn.Module):
    def __init__(self, in_channel, out_channel, stride, se=None):
        super(BlockWithDownSample, self).__init__()
        self.downSample = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1,
                      stride=stride, padding=0),
            nn.BatchNorm2d(out_channel)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3,
                      stride=stride, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel)
        )
        self.se = se

    def forward(self, img):
        identity = self.downSample(img)
        out = self.conv1(img)
        out = self.conv2(out)
        if self.se != None:
            out = self.se(out)
        out += identity
        return F.relu(out)
