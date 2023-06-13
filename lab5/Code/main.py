import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
import argparse
import numpy as np
import time
import random
from model import Generator, Discriminator
from torch.utils.tensorboard import SummaryWriter
from utils import dataPoints, gradPenalty, drawResult, drawLoss
# 为了复现
random.seed(6689)
np.random.seed(6689)
torch.manual_seed(6689)

# 训练参数
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--model', choices=['GAN', 'WGAN','WGAN-GP'], default='GAN')
parser.add_argument('--device', choices=['cuda', 'cpu'], default= 'cuda')
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd', 'rmrsp'], default='adam')
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--clamp', type=float, default=0.1)
args = parser.parse_args()
device = torch.device(args.device)

def train():
    # 构建模型
    print("---开始构建模型---")
    G = Generator().to(device)
    if args.model == 'GAN':
        D = Discriminator(True).to(device)
    else:
        D = Discriminator(False).to(device)

    # 初始化优化器
    if args.optimizer == 'adam':
        optimG = torch.optim.Adam(G.parameters(), lr=1e-5)
        optimD = torch.optim.Adam(D.parameters(), lr=1e-5)
    elif args.optimizer == 'sgd':
        optimG = torch.optim.SGD(G.parameters(), lr=4e-4)
        optimD = torch.optim.SGD(D.parameters(), lr=4e-4)
    elif args.optimizer == 'rmrsp':
        optimG = torch.optim.RMSprop(G.parameters(), lr=3e-4)
        optimD = torch.optim.RMSprop(D.parameters(), lr=3e-4)
        
    # 训练过程
    print("---开始训练---")
    lossDVal = []
    lossGVal = []
    trainData = dataPoints()
    trainLoader = DataLoader(trainData, shuffle=True, batch_size=args.batch_size)
    # writer = SummaryWriter(log_dir='/root/tf-logs')
    for e in range(args.epoch):
        for data in tqdm(trainLoader):
            data = data.to(device)
            # Discriminator
            labelPred = D(data)
            z = torch.randn(data.shape[0], 10).to(device)
            # 由G生成图片去获取该图片是否真实的概率
            fakeX = G(z).detach()
            fakePred = D(fakeX)
            if args.model == "GAN":
                lossD = -(torch.log(labelPred)+torch.log(1-fakePred)).mean()
            if args.model == 'WGAN':
                # 限制判别器参数大小
                for p in D.parameters():
                    p.data.clamp_(-args.clamp, args.clamp)
                lossD = (fakePred - labelPred).mean()
            if args.model == 'WGAN-GP':
                # 限制判别器参数大小
                for p in D.parameters():
                    p.data.clamp_(-args.clamp, args.clamp)
                lossD = (fakePred - labelPred).mean() + 0.2*gradPenalty(D, data, fakeX, device)
            optimD.zero_grad()
            lossD.backward()
            optimD.step()
            
            # Generator
            z = torch.randn(args.batch_size, 10).to(device)
            fakeX = G(z)
            fakePred = D(fakeX)
            if args.model == "GAN":
                lossG = torch.log(1- fakePred).mean()
            else:
                lossG = -fakePred.mean()
            optimG.zero_grad()
            lossG.backward()
            optimG.step()
        # writer.add_scalar(tag='Discriminator Loss', scalar_value=lossD.item(), global_step=e)
        # writer.add_scalar(tag='Generator Loss', scalar_value=lossG.item(), global_step=e)
        lossDVal.append(lossD.item())
        lossGVal.append(lossG.item())
        print("第{}次训练，Discriminator Loss:{:.4f} Generator Loss:{:.4f}".format(e, lossD.item(), lossG.item()))
            
        if e%50 == 0:
            drawResult(D, G, e, args, device, trainData.data)
    # Draw Loss
    drawLoss(lossDVal, lossGVal, args)

    # save the model
    state = {"modelD": D.state_dict(), "modelG": G.state_dict()}
    torch.save(state, "./_{}_{}_{}.pt".format(int(time.time()), args.model, args.optimizer))
    
    
train()