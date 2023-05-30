from torch import nn
from torch.nn import functional as F
from model.Blocks import CommonBlock, BlockWithDownSample, SEBlock

class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1)
        )
        self.conv2 = nn.Sequential(
            CommonBlock(64, 64),
            CommonBlock(64, 64),
            CommonBlock(64, 64)
        )
        self.conv3 = nn.Sequential(
            BlockWithDownSample(64, 128, 2),
            CommonBlock(128, 128),
            CommonBlock(128, 128),
            CommonBlock(128, 128)
        )
        self.conv4 = nn.Sequential(
            BlockWithDownSample(128, 256, 2),
            CommonBlock(256, 256),
            CommonBlock(256, 256),
            CommonBlock(256, 256),
            CommonBlock(256, 256),
            CommonBlock(256, 256)
        )
        self.conv5 = nn.Sequential(
            BlockWithDownSample(256, 512, 2),
            CommonBlock(512, 512),
            CommonBlock(512, 512)
        )
        self.dense = nn.Sequential( 
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, img):
        img = self.conv1(img)
        img = self.conv2(img)     
        img = self.conv3(img)
        img = self.conv4(img)
        img = self.conv5(img)
        img = self.dense(img)           
        return img
    
    def __str__(self):
        return 'ResNet'
    
class SEResNet(nn.Module):
    def __init__(self, num_classes):
        super(SEResNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1)
        )
        self.conv2 = nn.Sequential(
            CommonBlock(64, 64, SEBlock(64, 16)),
            CommonBlock(64, 64, SEBlock(64, 16)),
            CommonBlock(64, 64, SEBlock(64, 16))
        )
        self.conv3 = nn.Sequential(
            BlockWithDownSample(64, 128, 2, SEBlock(128, 16)),
            CommonBlock(128, 128, SEBlock(128, 16)),
            CommonBlock(128, 128, SEBlock(128, 16)),
            CommonBlock(128, 128, SEBlock(128, 16))
        )
        self.conv4 = nn.Sequential(
            BlockWithDownSample(128, 256, 2, SEBlock(256, 16)),
            CommonBlock(256, 256, SEBlock(256, 16)),
            CommonBlock(256, 256, SEBlock(256, 16)),
            CommonBlock(256, 256, SEBlock(256, 16)),
            CommonBlock(256, 256, SEBlock(256, 16)),
            CommonBlock(256, 256, SEBlock(256, 16))
        )
        self.conv5 = nn.Sequential(
            BlockWithDownSample(256, 512, 2, SEBlock(512, 16)),
            CommonBlock(512, 512, SEBlock(512, 16)),
            CommonBlock(512, 512, SEBlock(512, 16))
        )
        self.dense = nn.Sequential( 
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Dropout(0.65),
            nn.Linear(512, 512),
            nn.Dropout(0.65),
            nn.Linear(512, num_classes)
        )

    def forward(self, img):
        img = self.conv1(img)
        img = self.conv2(img)     
        img = self.conv3(img)
        img = self.conv4(img)
        img = self.conv5(img)
        img = self.dense(img)           
        return img
    
    def __str__(self):
        return 'SEResNet'
    
class SEResNet50(nn.Module):
    def __init__(self, num_classes):
        super(SEResNet50, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1)
        )
        self.conv2 = nn.Sequential(
            CommonBlock(64, 64, SEBlock(64, 16)),
            CommonBlock(64, 64, SEBlock(64, 16)),
            CommonBlock(64, 64, SEBlock(64, 16))
        )
        self.conv3 = nn.Sequential(
            BlockWithDownSample(64, 128, 2, SEBlock(128, 16)),
            CommonBlock(128, 128, SEBlock(128, 16)),
            CommonBlock(128, 128, SEBlock(128, 16)),
            CommonBlock(128, 128, SEBlock(128, 16))
        )
        self.conv4 = nn.Sequential(
            BlockWithDownSample(128, 256, 2, SEBlock(256, 16)),
            CommonBlock(256, 256, SEBlock(256, 16)),
            CommonBlock(256, 256, SEBlock(256, 16)),
            CommonBlock(256, 256, SEBlock(256, 16)),
            CommonBlock(256, 256, SEBlock(256, 16)),
            CommonBlock(256, 256, SEBlock(256, 16))
        )
        self.conv5 = nn.Sequential(
            BlockWithDownSample(256, 512, 2, SEBlock(512, 16)),
            CommonBlock(512, 512, SEBlock(512, 16)),
            CommonBlock(512, 512, SEBlock(512, 16)),
            CommonBlock(512, 512, SEBlock(512, 16))
        )
        self.conv6 = nn.Sequential(
            BlockWithDownSample(512, 1024, 2, SEBlock(1024, 16)),
            CommonBlock(1024, 1024, SEBlock(1024, 16)),
            CommonBlock(1024, 1024, SEBlock(1024, 16)),
            CommonBlock(1024, 1024, SEBlock(1024, 16)),
            CommonBlock(1024, 1024, SEBlock(1024, 16)),
            CommonBlock(1024, 1024, SEBlock(1024, 16)),
        )
        self.conv7 = nn.Sequential(
            BlockWithDownSample(1024, 2048, 2, SEBlock(2048, 16)),
            CommonBlock(2048, 2048, SEBlock(2048, 16)),
            CommonBlock(2048, 2048, SEBlock(2048, 16)),
        )
        self.dense = nn.Sequential( 
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Dropout(0.65),
            nn.Linear(2048, 2048),
            nn.Dropout(0.65),
            nn.Linear(2048, num_classes)
        )

    def forward(self, img):
        img = self.conv1(img)
        img = self.conv2(img)     
        img = self.conv3(img)
        img = self.conv4(img)
        img = self.conv5(img)
        img = self.conv6(img)
        img = self.conv7(img)
        img = self.dense(img)           
        return img
    
    def __str__(self):
        return 'SEResNet50'