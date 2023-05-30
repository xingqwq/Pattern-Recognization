from train_and_test.train_and_test import Train
import torchvision.transforms as transforms
import torch.nn as nn
import torch
import random
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.models import vgg11
from model.ResNet import ResNet

# 为了复现
random.seed(6689)
np.random.seed(6689)
torch.manual_seed(6689)

if __name__ == '__main__':
    epoch = 14
    batch_size = 128
    size = 224
    transform = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])]
    )
    # 使用torchvision自带VGG模型进行测试
    train_adam = Train(
        epoch,
        batch_size,
        size,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam,
        scheduler=MultiStepLR,
        model=ResNet,
        transform=transform
    )
    # train_adam.train()
    train_adam.generate_test_csv("./save_model/ResNet_0.8400.ckpt")
    # train_adam.save()