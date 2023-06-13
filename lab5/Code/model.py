import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        output = self.net(x)
        return output

class Discriminator(nn.Module):
    def __init__(self, isSigmoid = False):
        super(Discriminator, self).__init__()
        self.isSigmoid = isSigmoid
        self.sigmoid = nn.Sigmoid()
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
        )
    def forward(self, x):
        output = self.net(x)
        if self.isSigmoid == True:
            output = self.sigmoid(output)
        return output.view(-1)
