from locale import atoi
import torch
import torch.nn as nn
import time
from tqdm import tqdm,trange
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import argparse
from PIL import Image
import random

class AlexNet(nn.Module):
    def __init__(self, labelCnt, dropout = 0.3):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),  # 96*54*54
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0), # 96*26*26

            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2), # 256*26*26
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0), # 256*12*12

            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1), # 384*12*12
            nn.ReLU(),

            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1), # 384*12*12
            nn.ReLU(),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1), # 256*12*12
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0), # 256*5*5
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256 * 5 * 5, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, labelCnt),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

class Caltech101(Dataset):
    def __init__(self, imgPath, imgClass, classCnt, id, transform=None):
        self.imgPath = imgPath
        self.imgClass = imgClass
        self.classCnt = classCnt
        self.id = id
        self.transform = transform

    def __getitem__(self, item):
        img = Image.open(self.imgPath[item]).convert("RGB")
        label = atoi(self.imgClass[item])
        if self.id == 1:
            img = transforms.RandomAffine(degrees=0,translate=(0.1,0.1),shear=(-15,15))(img)
            img = transforms.RandomHorizontalFlip(p=0.8)(img)
        if self.transform is not None:
            img = self.transform(img)
        tmp = [0 if i != label else 1 for i in range(0,self.classCnt)]
        labelTensor = torch.tensor(tmp, dtype=torch.float)
        return img, labelTensor, label

    def __len__(self):
        return len(self.imgPath)

class Trainer:
    def __init__(self, trainLoader, validLoader, testLoader, args):
        # 定义参数
        self.lr = args.lr
        self.batchSize = args.batch_size
        self.device = torch.device(args.device)
        self.labelCnt = args.label_cnt
        self.testAcc = 0
        self.trainAcc = 0
        
        self.trainLoader = DataLoader(
            dataset=trainLoader,
            num_workers=args.num_worker,
            batch_size=args.batch_size,
            shuffle=True
        )
        self.validLoader = DataLoader(
            dataset=validLoader,
            num_workers=args.num_worker,
            batch_size=1,
            shuffle=True
        )
        self.testLoader = DataLoader(
            dataset=testLoader,
            num_workers=args.num_worker,
            batch_size=1,
            shuffle=True
        )
        
        # 定义模型、优化器
        self.model = AlexNet(self.labelCnt, args.dropout).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(),lr=self.lr,weight_decay=0.001)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.scheduler = MultiStepLR(self.optim, milestones=[80, 120, 160], gamma=0.5) 
        
    def train(self, epoch=10):
        print("{}{}{}".format('*'*15,"开始训练",'*'*15))
        self.writer = SummaryWriter(log_dir='/root/tf-logs')
        for i in range(epoch):
            accCnt = 0
            totalCnt = 0
            self.model.train()
            for imgs, labelTensors, labels in tqdm(self.trainLoader):
                imgs = imgs.to(self.device)
                labelTensors = labelTensors.to(self.device)
                # 训练
                y = self.model(imgs)
                loss = self.criterion(y, labelTensors)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                # 统计正确率
                y = torch.argmax(y,dim=1)
                for j in range(len(labels)):
                    if y[j].item() == labels[j].item():
                        accCnt += 1
                    totalCnt += 1
            if accCnt/totalCnt > self.trainAcc:
                self.trainAcc = accCnt/totalCnt
            if i % 5 == 0:
                acc = self.valid()
                self.writer.add_scalar(tag='acc/validation', scalar_value=acc, global_step=i/5)
            self.scheduler.step()
            self.writer.add_scalar(tag='loss/train', scalar_value=loss, global_step=i)
            self.writer.add_scalar(tag='acc/train', scalar_value=accCnt/totalCnt, global_step=i)
            print("第{}次训练，loss={:.4f} 正确率={:.4f} 学习率={:.7f}".format(i, loss.item(), accCnt/totalCnt, self.scheduler.get_last_lr()[0]))
        
    def valid(self):
        accCnt = 0
        totalCnt = 0
        self.model.eval()
        for imgs, labelTensors, labels in self.validLoader:  
            imgs = imgs.to(self.device)
            labelTensors = labelTensors.to(self.device)
            # 统计正确率
            y = self.model(imgs)
            y = torch.argmax(y,dim=1)
            for j in range(len(labels)):
                if y[j].item() == labels[j].item():
                    accCnt += 1
                totalCnt += 1
        if self.testAcc < accCnt/totalCnt:
            self.testAcc = accCnt/totalCnt
            self.save(0)
        print("验证集数据，正确率={:.4f}".format(accCnt/totalCnt))
        return accCnt/totalCnt
    
    def test(self, ptFile):
        print("{} 开始测试 {}".format("*"*5,"*"*5))
        accCnt = 0
        totalCnt = 0
        self.model.load_state_dict(torch.load(ptFile))
        self.model.eval()
        for imgs, labelTensors, labels in self.testLoader:  
            imgs = imgs.to(self.device)
            labelTensors = labelTensors.to(self.device)
            # 统计正确率
            y = self.model(imgs)
            y = torch.argmax(y,dim=1)
            for j in range(len(labels)):
                if y[j].item() == labels[j].item():
                    accCnt += 1
                totalCnt += 1
        print("测试集数据，正确率={:.4f}".format(accCnt/totalCnt))
        return accCnt/totalCnt
    
    def save(self, id):
        if id == 0:
            torch.save(self.model.state_dict(),"./pt/{}_testACC_{:.3f}.pt".format(int(time.time()),self.testAcc))
        else:
            torch.save(self.model.state_dict(),"./pt/{}_trainACC_{:.3f}.pt".format(int(time.time()),self.trainAcc))