import torch
import torch.nn as nn
import time
from tqdm import tqdm,trange
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

torch.manual_seed(668945)

class minstModel(nn.Module):
    def __init__(self, inputSize, labelCnt) -> None:
        super(minstModel, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(inputSize*inputSize, inputSize*4),
            nn.LeakyReLU()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(inputSize*4, inputSize*2),
            nn.LeakyReLU()
        )
        self.linear3 = nn.Sequential(
            nn.Linear(inputSize*2, labelCnt),
            nn.LeakyReLU()
        )

    def forward(self, img):
        img = self.linear1(img)
        img = self.linear2(img)
        img = self.linear3(img)
        
        return img

class trainer:
    def __init__(self, inputSize=28, labelCnt=10, lr=0.001, batchSize = 8, device='cpu'):
        # 定义参数
        self.lr = lr
        self.batchSize = batchSize
        self.device = device
        self.inputSizt = inputSize
        self.labelCnt = labelCnt
        self.testAcc = 0
        self.trainAcc = 0
        
        # 定义模型、优化器
        self.model = minstModel(inputSize, labelCnt).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(),lr=self.lr)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.scheduler = MultiStepLR(self.optim, milestones=[5, 10, 15, 20, 25], gamma=0.8) 
        
        # 加载数据
        self.trainDataset = datasets.MNIST('./data', train=True, transform=transforms.ToTensor(), download=True)
        self.testDataset =  datasets.MNIST('./data', train=False, transform=transforms.ToTensor(), download=True)
        
        self.trainDataLoader = DataLoader(
            dataset=self.trainDataset,
            batch_size=self.batchSize,
            shuffle=True,
            num_workers=8
        )
        
        self.testDataLoader = DataLoader(
            dataset=self.testDataset,
            batch_size=self.batchSize,
            shuffle=True,
            num_workers=8
        )
        
    def train(self, epoch=10):
        print("{}{}{}".format('*'*15,"开始训练",'*'*15))
        lossArray = []
        traindAccArray = []
        testAccArray = []
        for i in range(epoch):
            accCnt = 0
            totalCnt = 0
            for id, data in tqdm(enumerate(self.trainDataLoader)):
                # 处理数据
                imgs, labelsRaw = data
                imgs = torch.flatten(imgs,2,3)
                labels = []
                for j in labelsRaw:
                    tmp = [0 for _ in range(self.labelCnt)]
                    tmp[j.item()] = 1
                    labels.append(tmp)
                labels = torch.tensor(labels, dtype=torch.float).to(self.device)
                
                # 训练
                y = self.model(imgs.to(self.device))
                loss = self.criterion(y.squeeze(1), labels)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                
                # 统计正确率
                y = torch.argmax(y.squeeze(1),dim=1)
                for j in range(len(labelsRaw)):
                    if y[j] == labelsRaw[j]:
                        accCnt += 1
                    totalCnt += 1
            if accCnt/totalCnt > self.trainAcc:
                self.trainAcc = accCnt/totalCnt
            self.scheduler.step()
            lossArray.append(loss.item())
            traindAccArray.append(accCnt/totalCnt)
            testAccArray.append(self.test())
            print("第{}次训练，loss={:.4f} 正确率={:.4f} 学习率={:.7f}".format(i, loss.item(), accCnt/totalCnt, self.scheduler.get_last_lr()[0]))

        # 绘制结果曲线
        fig = plt.figure(3, figsize=(16, 8))
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)
        ax1.plot(lossArray, 'r', label='trainLoss')
        ax2.plot(traindAccArray, 'g', label='trainAccuracy')
        ax3.plot(testAccArray, 'b', label='testAccuracy')
        ax1.legend(loc='upper right')
        ax2.legend(loc='upper left')
        ax3.legend(loc='upper left')
        ax1.set_title("Epoch = {} BatchSize={} LR={:.5f}".format(epoch, self.batchSize, self.scheduler.get_last_lr()[0]))
        ax2.set_title("Epoch = {} BatchSize={} LR={:.5f}".format(epoch, self.batchSize, self.scheduler.get_last_lr()[0]))
        ax3.set_title("Epoch = {} BatchSize={} LR={:.5f}".format(epoch, self.batchSize, self.scheduler.get_last_lr()[0]))
        ax1.set_xticks([])
        ax2.set_xticks([])
        ax3.set_xticks([])
        fig.savefig("./{}_{}_{}_{:.5f}.png".format(int(time.time()),epoch, self.batchSize, self.scheduler.get_last_lr()[0]))
        fig.clear()
        
    def test(self):
        loss = 0
        accCnt = 0
        totalCnt = 0
        for id, data in enumerate(self.testDataLoader):
            # 处理数据
            imgs, labelsRaw = data
            imgs = torch.flatten(imgs,2,3)
            
            # 统计正确率
            y = self.model(imgs.to(self.device))
            y = torch.argmax(y.squeeze(1),dim=1)
            for j in range(len(labelsRaw)):
                if y[j] == labelsRaw[j]:
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

model = trainer(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), batchSize=64)
for i in range(5, 35, 5):
    model.train(i)
    model.test()
    model.save()