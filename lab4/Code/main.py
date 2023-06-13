import jieba
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from gensim.models.word2vec import Word2Vec
from tqdm import tqdm
from documentModel import getWord2Vec, DocumentReader, DocumentClassifier
from TModel import TData, TModel
import argparse
import numpy as np
import random
import pickle

# 为了复现
random.seed(6689)
np.random.seed(6689)
torch.manual_seed(6689)

# 训练参数
parser = argparse.ArgumentParser()
parser.add_argument('--task', choices=['TModel', 'DModel'], default='DModel')
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument("--pt", type=str)
parser.add_argument('--device', choices=['cuda', 'cpu'], default= 'cuda')
parser.add_argument('--dropout', type=float, default= 0.4)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_worker', type=int, default=8)
parser.add_argument('--input_size', type=int, default=100)
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--label_cnt', type=int, default=10)
parser.add_argument('--maxLen', type=int, default=50)
parser.add_argument('--model', type=str, choices=['RNN', 'GRU','LSTM','BiLSTM'], default="RNN")
args = parser.parse_args()

if args.task == 'DModel':
    # 文本分类
    label = ['书籍','平板','手机','水果','洗发水','热水器','蒙牛','衣服','计算机','酒店']
    label2id = {i:id for id,i in enumerate(label)}
    id2label = {label2id[i]:i for i in label}
    data = pd.read_csv("./dataset/online_shopping_10_cats.csv")
    print(label2id, id2label)
    if args.mode == 'train':
        # # 划分数据集
        # trainData = []
        # testData = []
        # validData = []

        # for i in range(0, len(data)):
        #     if i%5 == 0:
        #         testData.append([data['cat'][i], data['review'][i]])
        #     elif i%5 == 4:
        #         validData.append([data['cat'][i], data['review'][i]])
        #     else:
        #         trainData.append([data['cat'][i], data['review'][i]])
        # word2VecModel = getWord2Vec()
        # word2VecModel.train(trainData)
        # print("{}开始划分数据集{}".format("*"*15,"*"*15))
        # # 训练集
        # trainDataSplited = []
        # testDataSplited = []
        # validDataSplited = []
        # for i in tqdm(trainData):
        #     label =i[0]
        #     data = i[1]
        #     if type(data) != str:
        #         continue
        #     data = jieba.lcut(data)
        #     dataLen = (len(data)if len(data)<args.maxLen else args.maxLen) + 2
        #     if len(data) < args.maxLen:
        #         data += ["<PAD>" for _ in range(args.maxLen-len(data))]
        #     elif len(data) > args.maxLen:
        #         data = data[:args.maxLen]
        #     data = ["<s>"]+data+["</s>"]
        #     trainDataSplited.append([label, data, dataLen])
            
        # # 测试集
        # for i in tqdm(testData):
        #     label =i[0]
        #     data = i[1]
        #     if type(data) != str:
        #         continue
        #     data = jieba.lcut(data)
        #     dataLen = (len(data)if len(data)<args.maxLen else args.maxLen) + 2
        #     if len(data) < args.maxLen:
        #         data += ["<PAD>" for _ in range(args.maxLen-len(data))]
        #     elif len(data) > args.maxLen:
        #         data = data[:args.maxLen]
        #     data = ["<s>"]+data+["</s>"]
        #     testDataSplited.append([label, data, dataLen])
        # # 验证集
        # for i in tqdm(validData):
        #     label =i[0]
        #     data = i[1]
        #     if type(data) != str:
        #         continue
        #     data = jieba.lcut(data)
        #     dataLen = (len(data)if len(data)<args.maxLen else args.maxLen) + 2
        #     if len(data) < args.maxLen:
        #         data += ["<PAD>" for _ in range(args.maxLen-len(data))]
        #     elif len(data) > args.maxLen:
        #         data = data[:args.maxLen]
        #     data = ["<s>"]+data+["</s>"]
        #     validDataSplited.append([label, data, dataLen])
        # # 写入文件
        # f = open("./trainData.pkl","wb")
        # pickle.dump(trainDataSplited,f)
        # f.close()
        # f = open("./testData.pkl","wb")
        # pickle.dump(testDataSplited,f)
        # f.close()
        # f = open("./validData.pkl","wb")
        # pickle.dump(validDataSplited,f)
        # f.close()
        word2VecModel = getWord2Vec("./word2vec.model")
        # 读取数据集
        f = open("./trainData.pkl","rb")
        trainDataSplited = pickle.load(f)
        f.close()
        f = open("./testData.pkl","rb")
        testDataSplited = pickle.load(f)
        f.close()
        f = open("./validData.pkl","rb")
        validDataSplited = pickle.load(f)
        f.close()
        # 初始化模型
        model = DocumentClassifier(args, trainDataSplited, testDataSplited, validDataSplited, word2VecModel, label2id, id2label, args.maxLen, word2VecModel.stopWords)
        model.train()
    else:
        word2VecModel = getWord2Vec("./word2vec.model")
        # 读取数据集
        f = open("./trainData.pkl","rb")
        trainDataSplited = pickle.load(f)
        f.close()
        f = open("./testData.pkl","rb")
        testDataSplited = pickle.load(f)
        f.close()
        f = open("./validData.pkl","rb")
        validDataSplited = pickle.load(f)
        f.close()
        # 初始化模型 
        model = DocumentClassifier(args, trainDataSplited, testDataSplited, validDataSplited, word2VecModel, label2id, id2label, args.maxLen, word2VecModel.stopWords)
        model.model.load_state_dict(torch.load("./pt/1685254120_GRU_testACC_0.873.pt"))
        model.test()
else:
    args.input_size = 7
    args.label_cnt = 288
    args.maxLen = 720
    args.batch_size = 8
    if args.mode == "train":
        trainData, testData, TScaler = TData("./dataset/jena_climate_2009_2016.csv").read()
        model = TModel(args, trainData, testData, TScaler)
        model.train()
    else:
        trainData, testData, TScaler = TData("./dataset/jena_climate_2009_2016.csv").read()
        model = TModel(args, trainData, testData, TScaler)
        model.model.load_state_dict(torch.load(args.pt))
        model.test()
