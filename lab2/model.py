import torch
import torch.nn as nn
import time
from tqdm import tqdm,trange
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import argparse
from PIL import Image

class AlexNet(nn.Module):
    def __init__(self, labelCnt, dropout = 0.3):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),

            nn.Conv2d(48, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),

            nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, labelCnt),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

class Caltech101(Dataset):
    def __init__(self, imgPath, imgClass, classCnt, transform=None):
        self.imgPath = imgPath
        self.imgClass = imgClass
        self.classCnt = classCnt
        self.transform = transform

    def __getitem__(self, item):
        img = Image.open(self.imgPath[item]).convert("RGB")
        label = self.imgClass[item]
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
            batch_size=args.batch_size,
            shuffle=True
        )
        self.testLoader = DataLoader(
            dataset=testLoader,
            num_workers=args.num_worker,
            batch_size=args.batch_size,
            shuffle=True
        )
        
        # 定义模型、优化器
        self.model = AlexNet(self.labelCnt, args.dropout).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(),lr=self.lr)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.scheduler = MultiStepLR(self.optim, milestones=[5, 10, 15, 20, 25], gamma=1.0) 
        
    def train(self, epoch=10):
        print("{}{}{}".format('*'*15,"开始训练",'*'*15))
        lossArray = []
        traindAccArray = []
        testAccArray = []
        for i in range(epoch):
            accCnt = 0
            totalCnt = 0
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
            self.scheduler.step()
            lossArray.append(loss.item())
            traindAccArray.append(accCnt/totalCnt)
            # testAccArray.append(self.valid())
            print("第{}次训练，loss={:.4f} 正确率={:.4f} 学习率={:.7f}".format(i, loss.item(), accCnt/totalCnt, self.scheduler.get_last_lr()[0]))

        # 绘制结果曲线
        # fig = plt.figure(3, figsize=(16, 8))
        # ax1 = fig.add_subplot(1, 3, 1)
        # ax2 = fig.add_subplot(1, 3, 2)
        # ax3 = fig.add_subplot(1, 3, 3)
        # ax1.plot(lossArray, 'r', label='trainLoss')
        # ax2.plot(traindAccArray, 'g', label='trainAccuracy')
        # ax3.plot(testAccArray, 'b', label='testAccuracy')
        # ax1.legend(loc='upper right')
        # ax2.legend(loc='upper left')
        # ax3.legend(loc='upper left')
        # ax1.set_title("Epoch = {} BatchSize={} LR={:.5f}".format(epoch, self.batchSize, self.scheduler.get_last_lr()[0]))
        # ax2.set_title("Epoch = {} BatchSize={} LR={:.5f}".format(epoch, self.batchSize, self.scheduler.get_last_lr()[0]))
        # ax3.set_title("Epoch = {} BatchSize={} LR={:.5f}".format(epoch, self.batchSize, self.scheduler.get_last_lr()[0]))
        # ax1.set_xticks([])
        # ax2.set_xticks([])
        # ax3.set_xticks([])
        # fig.savefig("./{}_{}_{}_{:.5f}.png".format(int(time.time()),epoch, self.batchSize, self.scheduler.get_last_lr()[0]))
        # fig.clear()
        
    def valid(self):
        loss = 0
        accCnt = 0
        totalCnt = 0
        for imgs, labels in self.validLoader:  
            imgs = imgs.to(self.device)
            labelTensors = labelTensors.to(self.device)
            # 统计正确率
            y = self.model(imgs)
            y = torch.argmax(y,dim=1)
            for j in range(len(labels)):
                if y[j].item() == labels[j]:
                    accCnt += 1
                totalCnt += 1
        if self.testAcc < accCnt/totalCnt:
            self.testAcc = accCnt/totalCnt
            self.save()
        # self.testAcc = accCnt/totalCnt
        print("测试数据，正确率={:.4f}".format(accCnt/totalCnt))
        return accCnt/totalCnt
        
    def save(self):
        if self.testAcc != 0:
            torch.save(self.model.state_dict(),"./{}_testACC_{:.3f}.pt".format(int(time.time()),self.testAcc))
        else:
            torch.save(self.model.state_dict(),"./{}_trainACC_{:.3f}.pt".format(int(time.time()),self.trainAcc))