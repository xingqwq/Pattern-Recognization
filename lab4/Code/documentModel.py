import jieba
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import Dataset, DataLoader
from gensim.models.word2vec import Word2Vec
from tqdm import tqdm
from model import RNN,GRU,LSTM

class DocumentReader(Dataset):
    def __init__(self, data, label2id, maxLen, word2vec):
        super(DocumentReader, self).__init__()
        self.data = data
        self.label2id = label2id
        self.maxLen = maxLen
        self.word2vec = word2vec

    def __getitem__(self, item):
        label, data, dataLen = self.data[item]
        data = [self.word2vec[word] for word in data]
        data = np.array(data)
        labelTensor = torch.tensor([1 if i == self.label2id[label] else 0 for i in range(0, len(self.label2id))],dtype=torch.float)
        dataTensor = torch.tensor(data, dtype=torch.float)
        return dataTensor, labelTensor, self.label2id[label], dataLen

    def __len__(self):
        return len(self.data)

class getWord2Vec():
    def __init__(self, modelPath = None):
        if modelPath != None:
            self.model = Word2Vec.load(modelPath)
        self.vectorSize = 100
        self.stopWords = []
        self.initStopWords()
        
    def initStopWords(self):
        with open("./dataset/stop_words.txt", "r") as file:
            for line in file.readlines():
                self.stopWords.append(line.strip("\n"))
        
    def train(self, data):
        cutWords = []
        for i in data:
            i = i[1]
            if type(i) != str:
                continue
            cutWords += [jieba.lcut(i)]
        corpus = []
        for line in cutWords:
            tmp = []
            for word in line:
                tmp.append(word)
                # if word not in self.stopWords:
                #     tmp.append(word)
                # else:
                #     tmp.append("<UNK>")
            corpus.append(["<s>"]+tmp+["</s>"])
        self.model = Word2Vec(sentences=corpus, vector_size=self.vectorSize, window=5, hs=1, min_count=1, workers=10)
        self.model.save("./word2vec.model")
    
    def __getitem__ (self, word):
        if word == "<PAD>":
            return [0 for _ in range(self.vectorSize)]
        if word not in self.model.wv:
            return [0 for _ in range(self.vectorSize)]
        else:
            return self.model.wv[word]
            
class DocumentClassifier(nn.Module):
    def __init__(self, args, trainData, testData, validData, word2vec, label2id, id2label, maxLen, stopWords):
        super(DocumentClassifier, self).__init__()
        # 定义参数
        self.args = args
        self.lr = args.lr
        self.batchSize = args.batch_size
        self.device = torch.device(args.device)
        self.labelCnt = args.label_cnt
        self.testF1 = 0
        self.trainF1 = 0
        self.label2id = label2id
        self.id2label = id2label
        self.maxLen = maxLen
        
        self.trainLoader = DataLoader(
            dataset=DocumentReader(trainData, label2id, maxLen, word2vec),
            num_workers=args.num_worker,
            batch_size=args.batch_size,
            shuffle=True
        )
        self.validLoader = DataLoader(
            dataset=DocumentReader(validData, label2id, maxLen, word2vec),
            num_workers=args.num_worker,
            batch_size=args.batch_size,
            shuffle=True
        )
        self.testLoader = DataLoader(
            dataset=DocumentReader(testData, label2id, maxLen, word2vec),
            num_workers=args.num_worker,
            batch_size=args.batch_size,
            shuffle=True
        )
        
        # 定义模型、优化器
        if args.model == "RNN":
            self.model = RNN(args.input_size, 10, args.hidden_size, self.device).to(self.device)
        elif args.model == "GRU":
            self.model = GRU(args.input_size, 10, args.hidden_size, self.device).to(self.device)
        elif args.model == "LSTM":
            self.model = LSTM(args.input_size, 10, args.hidden_size, self.device, isBidirectional = False).to(self.device)
        elif args.model == "BiLSTM":
            self.model = LSTM(args.input_size, 10, args.hidden_size, self.device, isBidirectional = True).to(self.device)
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.scheduler = MultiStepLR(self.optim, milestones=[80, 120, 160], gamma=1.0) 
        
    def train(self):
        print("{}开始训练{}".format("*"*15,"*"*15))
        self.writer = SummaryWriter(log_dir='/root/tf-logs')
        for e in range(self.args.epoch):
            lossVal = 0
            tp = [0 for _ in range(self.args.label_cnt)]
            fp = [0 for _ in range(self.args.label_cnt)]
            fn = [0 for _ in range(self.args.label_cnt)]
            f1 = [0 for _ in range(self.args.label_cnt)]
            self.model.train()
            for data, labelTensor, labels, dataLens in tqdm(self.trainLoader):
                data = data.to(self.args.device)
                labelTensor = labelTensor.to(self.args.device)
                # print(data)
                dataPacked = nn.utils.rnn.pack_padded_sequence(data, dataLens, batch_first=True, enforce_sorted=False)
                # print(dataPacked.data)
                y, hidden = self.model(dataPacked)
                loss = self.criterion(y, labelTensor)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                lossVal += loss.item()
                # 统计正确率、F1值
                y = torch.softmax(y, dim = 1)
                y = torch.argmax(y, dim=1)
                for j in range(len(labels)):
                    if y[j].item() == labels[j]:
                        tp[labels[j]] += 1
                    else:
                        fp[y[j].item()] += 1
                        fn[labels[j]] += 1
            # 计算F1
            f1Total = 0
            for i in range(self.args.label_cnt):
                _precision = tp[i]/(tp[i]+fp[i]+1)
                _recall = tp[i]/(tp[i]+fn[i]+1)
                if _precision == 0 and _recall == 0:
                    f1[i] = 0.0
                else:
                    f1[i] = 2*_precision*_recall/(_precision+_recall)
                f1Total += f1[i]
            if f1Total/self.args.label_cnt > self.trainF1:
                self.trainF1 = f1Total/self.args.label_cnt
            if e % 5 == 0:
                f1 = self.valid()
                self.writer.add_scalar(tag=str(self.model)+'-f1/validation', scalar_value=f1, global_step=e/5)
            self.scheduler.step()
            self.writer.add_scalar(tag=str(self.model)+'-loss/train', scalar_value=lossVal, global_step=e)
            self.writer.add_scalar(tag=str(self.model)+'-f1/train', scalar_value=f1Total/self.args.label_cnt, global_step=e)
            print("第{}次训练，loss={:.4f} F1={:.4f} 学习率={:.7f}".format(e, lossVal, f1Total/self.args.label_cnt, self.scheduler.get_last_lr()[0]))
    
    def valid(self):
        tp = [0 for _ in range(self.args.label_cnt)]
        fp = [0 for _ in range(self.args.label_cnt)]
        fn = [0 for _ in range(self.args.label_cnt)]
        f1 = [0 for _ in range(self.args.label_cnt)]
        self.model.eval()
        for data, labelTensor, labels, dataLens in tqdm(self.validLoader):
            data = data.to(self.args.device)
            labelTensor = labelTensor.to(self.args.device)
            # print(data)
            dataPacked = nn.utils.rnn.pack_padded_sequence(data, dataLens, batch_first=True, enforce_sorted=False)
            # print(dataPacked.data)
            y, hidden = self.model(dataPacked)
            # 统计正确率、F1值
            y = torch.softmax(y, dim = 1)
            y = torch.argmax(y, dim=1)
            for j in range(len(labels)):
                if y[j].item() == labels[j]:
                    tp[labels[j]] += 1
                else:
                    fp[y[j].item()] += 1
                    fn[labels[j]] += 1
        # 计算F1
        f1Total = 0
        for i in range(self.args.label_cnt):
            _precision = tp[i]/(tp[i]+fp[i]+1)
            _recall = tp[i]/(tp[i]+fn[i]+1)
            if _precision == 0 and _recall == 0:
                f1[i] = 0.0
            else:
                f1[i] = 2*_precision*_recall/(_precision+_recall)
            f1Total += f1[i]
        if f1Total/self.args.label_cnt > self.testF1:
            self.testF1 = f1Total/self.args.label_cnt
            self.save(0)
        print("验证集，F1={:.4f}".format(f1Total/self.args.label_cnt))
        return f1Total/self.args.label_cnt
    
    def test(self):
        tp = [0 for _ in range(self.args.label_cnt)]
        fp = [0 for _ in range(self.args.label_cnt)]
        fn = [0 for _ in range(self.args.label_cnt)]
        f1 = [0 for _ in range(self.args.label_cnt)]
        self.model.eval()
        for data, labelTensor, labels, dataLens in tqdm(self.testLoader):
            data = data.to(self.args.device)
            labelTensor = labelTensor.to(self.args.device)
            # print(data)
            dataPacked = nn.utils.rnn.pack_padded_sequence(data, dataLens, batch_first=True, enforce_sorted=False)
            # print(dataPacked.data)
            y, hidden = self.model(dataPacked)
            # 统计正确率、F1值
            y = torch.softmax(y, dim = 1)
            y = torch.argmax(y, dim=1)
            for j in range(len(labels)):
                if y[j].item() == labels[j]:
                    tp[labels[j]] += 1
                else:
                    fp[y[j].item()] += 1
                    fn[labels[j]] += 1
        # 计算F1
        f1Total = 0
        for i in range(self.args.label_cnt):
            _precision = tp[i]/(tp[i]+fp[i]+1)
            _recall = tp[i]/(tp[i]+fn[i]+1)
            if _precision == 0 and _recall == 0:
                f1[i] = 0.0
            else:
                f1[i] = 2*_precision*_recall/(_precision+_recall)
            f1Total += f1[i]
            print("类别：{}\t 精确率:{:.4f}\t 召回率:{:.4f} F1:{:.4}".format(self.id2label[i],_precision,_recall,f1[i]))
        if f1Total/self.args.label_cnt > self.testF1:
            self.testF1 = f1Total/self.args.label_cnt
            self.save(0)
        print("测试集，F1={:.4f}".format(f1Total/self.args.label_cnt))
        return f1Total/self.args.label_cnt
    
    def save(self, id):
        if id == 0:
            torch.save(self.model.state_dict(),"./pt/DModel_{}_{}_testACC_{:.3f}.pt".format(int(time.time()),str(self.model),self.testF1))
        else:
            torch.save(self.model.state_dict(),"./pt/DModel_{}_{}_trainACC_{:.3f}.pt".format(int(time.time()),str(self.model),self.trainF1))