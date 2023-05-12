import argparse
import torch
from torchvision import datasets, transforms
from model import AlexNet, Caltech101, Trainer
import os

# 为了复现
torch.manual_seed(668945)

# 训练参数
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--device', choices=['cuda', 'cpu'], default= 'cuda')
parser.add_argument('--dropout', type=float, default= 0.3)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--num_worker', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--label_cnt', type=int, default=101)
args = parser.parse_args()

imgPathTrain = []
imgLabelTrain= []
imgPathValid = []
imgLabelValid= []
imgPathTest = []
imgLabelTest= []

labelToNum = {}
# 读取数据
print("{}\n{}\n{}".format("*"*30, "正在读取数据", "*"*30))
for i in os.listdir("./101_ObjectCategories"):
    labelToNum.update({i:len(labelToNum)})
    tmp = os.listdir("./101_ObjectCategories/"+i)
    for j in range(0, len(tmp)):
        if(j<=round(0.8*len(tmp))):
            imgPathTrain.append("./101_ObjectCategories/"+i+"/"+tmp[j])
            imgLabelTrain.append(labelToNum[i])
        elif(j>round(0.8*len(tmp)) and j<= round(0.9*len(tmp))):
            imgPathValid.append("./101_ObjectCategories/"+i+"/"+tmp[j])
            imgLabelValid.append(labelToNum[i])
        else:
            imgPathTest.append("./101_ObjectCategories/"+i+"/"+tmp[j])
            imgLabelTest.append(labelToNum[i])

# 数据操作
trainTransform = transforms.Compose([
                    transforms.RandomResizedCrop(224), 
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

validTransform = transforms.Compose([
                    transforms.RandomResizedCrop(224), 
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

testTransform = transforms.Compose([
                    transforms.RandomResizedCrop(224), 
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainLoader = Caltech101(imgPathTrain, imgLabelTrain, args.label_cnt, trainTransform)
validLoader = Caltech101(imgPathValid, imgLabelValid, args.label_cnt, validTransform)
testLoader = Caltech101(imgPathTest, imgLabelTest, args.label_cnt, testTransform)

# 开始训练
print("{}\n{}\ndevice: {}\nlr: {}\ndropout: {}\nbatch_size: {}\n{}"
      .format("*"*30, "开始训练训练参数：", args.device, args.lr, args.dropout, args.batch_size,"*"*30))
model = Trainer(trainLoader, validLoader, testLoader, args)
for i in range(5, 35, 5):
    model.train(i)
#     model.test()
#     model.save()