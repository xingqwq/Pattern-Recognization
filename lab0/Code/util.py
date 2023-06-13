import requests
import sys
import os
import pickle as pkl
import numpy as np
import time
import torch
from tqdm import tqdm, trange
from dataset import splitData,TextBasedDataSet,dataSet,dataSetGraph
from transformers import AutoTokenizer,AutoModel
    
def makeData(device,manLen):
    # 获取数据
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    bertModel = AutoModel.from_pretrained("bert-base-cased").to(device)
    userData, trainData, validData, testData, id2index = splitData("./data/user.json", "./data/edge.json", "./data/train.csv", "./data/test.csv")

    print("开始准备训练数据\n")
    for i in trange(0,len(trainData)):
        tmp = dataSet(trainData[i:min(i+1,len(trainData))],userData,manLen,device,tokenizer,bertModel)
        des, tweet, profile, label = tmp.getData()
        # 写入磁盘
        f = open("./data_"+str(i)+".pkl",'wb')
        pkl.dump([des,tweet,profile,label],f)
        f.close()
    
    print("开始准备验证数据\n")
    for i in trange(0,len(validData)):
        tmp = dataSet(validData[i:min(i+1,len(validData))],userData,manLen,device,tokenizer,bertModel)
        des, tweet, profile, label = tmp.getData()
        # 写入磁盘
        f = open("./data_"+str(i+len(trainData))+".pkl",'wb')
        pkl.dump([des,tweet,profile,label],f)
        f.close()
    
    print("开始准备测试数据\n")
    for i in trange(0,len(testData)):
        tmp = dataSet(testData[i:min(i+1,len(testData))],userData,manLen,device,tokenizer,bertModel)
        des, tweet, profile, label = tmp.getData()
        # 写入磁盘
        f = open("./data_"+str(testData[i][0])+".pkl",'wb')
        pkl.dump([des,tweet,profile,label],f)
        f.close()

# 生成图模型数据
def makeGraphData(device,manLen):
    # 获取数据
    timeStamp = time.time()
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    bertModel = AutoModel.from_pretrained("roberta-base").to(device)
    userData, trainData, validData, testData, id2index = splitData("./data/user.json", "./data/edge.json", "./data/train.csv", "./data/test.csv")
    pData = np.vstack((trainData, validData, testData))
    
    # 生成关系图
    print("开始准备关系数据")
    edge = []
    edgeRelation = []
    for i in trange(0,len(pData)):
        if len(userData[str(pData[i][0])]["neighbors"]) > 0:
            for j in userData[str(pData[i][0])]["neighbors"]:
                try:
                    targetID = id2index[str(j[1])]
                    edge.append([i,targetID])
                    edgeRelation.append(j[0])
                except Exception as e:
                    continue
    
    edge = torch.tensor(edge).t()
    edgeRelation = torch.tensor(edgeRelation)
    print(edge.shape,edgeRelation.shape)
    torch.save(edge, "./"+str(int(timeStamp))+"_edge.pt")
    torch.save(edgeRelation, "./"+str(int(timeStamp))+"_edgeRelation.pt")
        
    # 写入磁盘
    screen = []
    des = []
    tweet = []
    profile = []
    personalIden = []
    label = []
    IDs = []
    print("开始准备数据")
    for i in trange(0,len(pData)):
        tmp = dataSetGraph(pData[i:min(i+1,len(pData))],userData,manLen,device,tokenizer,bertModel)
        screenR,desR, tweetR, profileR, personalId, labelR = tmp.getData()
        screen.append(screenR)
        des.append(desR)
        tweet.append(tweetR)
        profile.append(profileR)
        personalIden.append(personalId)
        label.append(labelR)
        IDs.append(pData[i][0])
    torch.save(torch.tensor(screen,dtype=torch.float),"./"+str(int(timeStamp))+"_screen.pt")
    torch.save(torch.tensor(des,dtype=torch.float),"./"+str(int(timeStamp))+"_des.pt")
    torch.save(torch.tensor(tweet,dtype=torch.float),"./"+str(int(timeStamp))+"_tweet.pt")
    torch.save(torch.tensor(profile,dtype=torch.float),"./"+str(int(timeStamp))+"_profile.pt")
    torch.save(torch.tensor(personalIden,dtype=torch.float),"./"+str(int(timeStamp))+"_personalIden.pt")
    torch.save(torch.tensor(label),"./"+str(int(timeStamp))+"_label.pt")
    torch.save(torch.tensor(IDs),"./"+str(int(timeStamp))+"_IDs.pt")

# 获取图数据
def getGraphData():
    print("开始装载数据")
    timeStamp = "1671290174"
    screen = torch.load("./"+timeStamp+"_screen.pt").squeeze(1).squeeze(1)
    des = torch.load("./"+timeStamp+"_des.pt").squeeze(1).squeeze(1)
    tweet = torch.load("./"+timeStamp+"_tweet.pt").squeeze(1).squeeze(1)
    profile = torch.load("./"+timeStamp+"_profile.pt").squeeze(1)
    personal = torch.load("./"+timeStamp+"_personalIden.pt").squeeze(1)
    label = torch.load("./"+timeStamp+"_label.pt").squeeze(1)
    edge = torch.load("./"+timeStamp+"_edge.pt")
    edgeRelation = torch.load("./"+timeStamp+"_edgeRelation.pt")
    print("数据装载完成")
    return screen.float(),des.float(), tweet.float(), profile.float(), personal.float(), label, edge, edgeRelation
        