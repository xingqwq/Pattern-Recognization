import argparse
import torch
from torchvision import datasets, transforms
from model import AlexNet, Caltech101, Trainer
import os
import random
from torchinfo import summary
import numpy as np
import pickle

# 为了复现
random.seed(6689)
np.random.seed(6689)
torch.manual_seed(6689)

# 训练参数
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument("--pt", type=str)
parser.add_argument('--device', choices=['cuda', 'cpu'], default= 'cuda')
parser.add_argument('--dropout', type=float, default= 0.4)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_worker', type=int, default=12)
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
    labelToNum.update({i:[]})
    tmp = os.listdir("./101_ObjectCategories/"+i)
    for j in range(0, len(tmp)):
        labelToNum[i].append(["./101_ObjectCategories/"+i+"/"+tmp[j], len(labelToNum)])
    labelToNum[i] = np.random.permutation(np.array(labelToNum[i]))

for id, i in enumerate(os.listdir("./101_ObjectCategories")):
    imgPathTrain += labelToNum[i][0:int(round(0.8*len(labelToNum[i]))),0].tolist()
    imgLabelTrain += labelToNum[i][0:int(round(0.8*len(labelToNum[i]))),1].tolist()
    if len(labelToNum[i])<60:
        for _ in range(2):
            imgPathTrain += labelToNum[i][0:int(round(0.8*len(labelToNum[i]))),0].tolist()
            imgLabelTrain += labelToNum[i][0:int(round(0.8*len(labelToNum[i]))),1].tolist()
    imgPathValid += labelToNum[i][int(round(0.8*len(labelToNum[i]))):int(round(0.9*len(labelToNum[i]))),0].tolist()
    imgLabelValid += labelToNum[i][int(round(0.8*len(labelToNum[i]))):int(round(0.9*len(labelToNum[i]))),1].tolist()
    imgPathTest += labelToNum[i][int(round(0.9*len(labelToNum[i]))):,0].tolist()
    imgLabelTest += labelToNum[i][int(round(0.9*len(labelToNum[i]))):,1].tolist()

# 写入
f = open("./imgPathTrain.pkl","wb")
pickle.dump(imgPathTrain,f)
f.close()

f = open("./imgLabelTrain.pkl","wb")
pickle.dump(imgLabelTrain,f)
f.close()

f = open("./imgPathValid.pkl","wb")
pickle.dump(imgPathValid,f)
f.close()

f = open("./imgLabelValid.pkl","wb")
pickle.dump(imgLabelValid,f)
f.close()

f = open("./imgPathTest.pkl","wb")
pickle.dump(imgPathTest,f)
f.close()

f = open("./imgLabelTest.pkl","wb")
pickle.dump(imgLabelTest,f)
f.close()

# 读取已划分的数据
f = open("./imgPathTrain.pkl","rb")
imgPathTrain = pickle.load(f)
f.close()

f = open("./imgLabelTrain.pkl","rb")
imgLabelTrain = pickle.load(f)
f.close()

f = open("./imgPathValid.pkl","rb")
imgPathValid = pickle.load(f)
f.close()

f = open("./imgLabelValid.pkl","rb")
imgLabelValid = pickle.load(f)
f.close()

f = open("./imgPathTest.pkl","rb")
imgPathTest = pickle.load(f)
f.close()

f = open("./imgLabelTest.pkl","rb")
imgLabelTest = pickle.load(f)
f.close()

# 数据操作
trainTransform = transforms.Compose([
                    transforms.Resize((224, 224)),  # 缩放
                    transforms.ToTensor(),
                    transforms.Normalize(0.5, 0.5)])

validTransform = transforms.Compose([
                    transforms.Resize((224, 224)),  # 缩放
                    transforms.ToTensor(),
                    transforms.Normalize(0.5, 0.5)])

testTransform = transforms.Compose([
                    transforms.Resize((224, 224)),  # 缩放
                    transforms.ToTensor(),
                    transforms.Normalize(0.5, 0.5)])

trainLoader = Caltech101(imgPathTrain, imgLabelTrain, args.label_cnt, 0, trainTransform)
validLoader = Caltech101(imgPathValid, imgLabelValid, args.label_cnt, 0, validTransform)
testLoader = Caltech101(imgPathTest, imgLabelTest, args.label_cnt, 0, testTransform)


# 输出模型信息
model = Trainer(trainLoader, validLoader, testLoader, args)
# model.model.load_state_dict(torch.load("./pt/1684055697_testACC_0.585.pt"))
# 开始训练
if args.mode == 'train':
    print("{}\n{}\ndevice: {}\nlr: {}\ndropout: {}\nbatch_size: {}\n{}"
        .format("*"*30, "开始训练训练参数：", args.device, args.lr, args.dropout, args.batch_size,"*"*30))
    model.train(300)
else:
    model.test(args.pt)
