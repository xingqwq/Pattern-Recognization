import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as DataUtil
import torch.optim as optim
from transformers import BertTokenizer,BertModel
import numpy as np
import sys
import os
import pickle as pkl
import time
from tqdm import tqdm, trange
from model import RobotGraphClassify
from dataset import splitData,TextBasedDataSet,dataSet
from util import makeData,makeGraphData,getGraphData

# 相关参数设置
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(10210619)
torch.cuda.manual_seed_all(10210619)
np.random.seed(10210619)

# 超参
MAX_LEN = 128
BATCH_SIZE = 8
LR = 1e-3
TRAINP = 0.85

# makeGraphData(device, MAX_LEN)

screen,des, tweet, profile, personal, label, edge,edgeRelation = getGraphData()
userData, trainData, validData, testData, id2index = splitData("./data/user.json", "./data/edge.json", "./data/train.csv", "./data/test.csv")
trainID = [0, len(trainData)]
validID = [len(trainData), len(trainData)+len(validData)]
testID = [len(trainData)+len(validData),len(trainData)+len(validData)+len(testData)]

# Model & Adam
print("开始初始化模型")
classcifier = RobotGraphClassify().to(device)
criterion = nn.BCELoss().to(device)
optimizer = optim.Adam(classcifier.parameters(), lr=1e-4)
classcifier.load_state_dict(torch.load("./1671322068_62.pt"))
print("模型初始化完成")

# 模型评估
def predict():
    classcifier.eval()
    with torch.no_grad():
        print("评估模型")
        tp = 0
        fp = 0
        fn = 0
        result = classcifier(screen.to(device),des.to(device),tweet.to(device),profile.to(device),personal.to(device),edge.to(device),edgeRelation.to(device))
        for i, j in zip(result[validID[0]:validID[1]],label[validID[0]:validID[1]]):
            tmpResult = 1 if i[0] > i[1] else 0
            tmpLabel = 1 if j[0] > j[1] else 0
            if tmpResult == 1 and tmpLabel == 1:
                tp += 1
            elif tmpResult == 0 and tmpLabel == 1:
                fn += 1
            elif tmpResult == 1 and tmpLabel == 0:
                fp += 1
        
        try:
            precision = tp/(tp+fp)
            recall = tp/(tp+fn)
            f = 2*precision*recall / (precision+recall)
            print("精确率：{} 召回率：{} f值：{}".format(precision,recall,f))
            return f
        except Exception as e:
            print(e)
            return 0
        
# 模型训练
def train():
    print("开始进行训练")
    max = 0.0
    for epoch in range(0,10000):
        classcifier.train()
        result = classcifier(screen.to(device),des.to(device),tweet.to(device),profile.to(device),personal.to(device),edge.to(device),edgeRelation.to(device))
        loss = criterion(result[trainID[0]:trainID[1]],label[trainID[0]:trainID[1]].to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        res = predict()
        if res * 100 > max and res > 0.5:
            max = res * 100
            torch.save(classcifier.state_dict(),"./"+str(int(time.time()))+"_"+str(int(res * 100))+".pt")
        print('epoch:%d loss:%.5f' % (epoch, loss.item()))

# 获得测试集标签
def eval():
    # 加载模型
    classcifier.eval()
    result = classcifier(screen.to(device),des.to(device),tweet.to(device),profile.to(device),personal.to(device),edge.to(device),edgeRelation.to(device))
    result = result[testID[0]:testID[1]]
    label = []
    ID = []
    cnt = 0
    for i in result:
        ID.append(str(testData[cnt][0]))
        cnt += 1
        if i[0] > i[1]:
            label.append(1)
        else:
            label.append(0)
    dataframe = pd.DataFrame({'ID':ID,'Label':label})
    dataframe.to_csv("./submission.csv",sep='\t',index=False)
    

train()