import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.preprocessing import MinMaxScaler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model import RNN,GRU,LSTM

class TReader(Dataset):
    def __init__(self, data, input_size, maxLen):
        super(TReader, self).__init__()
        self.data = data
        self.input_size = input_size
        self.maxLen = maxLen
        self.input=[]
        self.label=[]
        self.divideData()
        
    def divideData(self):
        selected_col = ["T (degC)","Tpot (K)","Tdew (degC)","rh (%)","VPmax (mbar)","VPact (mbar)","VPdef (mbar)","sh (g/kg)","H2OC (mmol/mol)","rho (g/m**3)","wv (m/s)","max. wv (m/s)","wd (deg)"]
        tmpData = []
        tmpLabel = []
        for key, value in self.data:
            # if len(value[selected_col].values)>144:
            #     print(key)
            tmpData.append(value[selected_col].values)
            tmpLabel.append(value["T (degC)"].values)
            if(len(tmpData)==7):
                self.input.append(tmpData[0:5])
                self.label.append(tmpLabel[5:7])
                tmpData = tmpData[5:7]
                tmpLabel = tmpLabel[5:7]
        
    def __getitem__(self, item):
        _input = [element for l in self.input[item] for element in l]
        _label = [element for l in self.label[item] for element in l]
        dataLen = len(_input)
        if dataLen<self.maxLen:
            _input += [[0 for _ in range(self.input_size)] for _ in range(self.maxLen-dataLen)]
            _label += [0 for _ in range(288-len(_label))]
        _input = np.array(_input,dtype=np.float)
        _label = np.array(_label,dtype=np.float)
        return torch.tensor(_input, dtype=torch.float),torch.tensor(_label, dtype=torch.float),dataLen
        
    def __len__(self):
        return len(self.input) 
    
class TData:
    def __init__(self, filePath):
        self.filePath = filePath

    def read(self):
        # data = pd.read_csv(self.filePath, parse_dates=['Date Time'], index_col='Date Time', 
        #                    date_format='%d.%m.%Y %H:%M:%S')
        data = pd.read_csv(self.filePath, parse_dates=['Date Time'], index_col='Date Time', 
                           date_parser=lambda x:pd.to_datetime(x,format='%d.%m.%Y %H:%M:%S'))
        data['year'] = data.index.year
        data['hour'] = data.index.hour
        data['year_month_day'] = data.index.year.astype(str) + '_' + data.index.month.astype(str) + '_' + data.index.day.astype(str)
        selected_col = ["T (degC)","Tpot (K)","Tdew (degC)","rh (%)","VPmax (mbar)","VPact (mbar)","VPdef (mbar)","sh (g/kg)","H2OC (mmol/mol)","rho (g/m**3)","wv (m/s)","max. wv (m/s)","wd (deg)"]

        # 归一化
        for col in selected_col:
            scaler = MinMaxScaler()
            data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))
        # 划分训练集和测试集
        train = data[data['year'].isin(range(2009, 2015))]
        test = data[data['year'].isin(range(2015, 2017))]
        train_group = train.groupby('year_month_day')
        test_group = test.groupby('year_month_day')
        
        return train_group, test_group

class TModel(nn.Module):
    def __init__(self, args, trainData, testData):
        super(TModel, self).__init__()
        # 定义参数
        self.args = args
        self.lr = args.lr
        self.batchSize = args.batch_size
        self.device = torch.device(args.device)
        self.labelCnt = args.label_cnt
        self.testAvgDeVal = 0
        self.trainAvgDeVal = 0
        
        self.trainLoader = DataLoader(
            dataset=TReader(trainData, self.args.input_size, self.args.maxLen),
            num_workers=args.num_worker,
            batch_size=args.batch_size,
            shuffle=True
        )
        self.testLoader = DataLoader(
            dataset=TReader(testData, self.args.input_size, self.args.maxLen),
            num_workers=args.num_worker,
            batch_size=args.batch_size,
            shuffle=True
        )
        
        # 定义模型、优化器
        if args.model == "RNN":
            self.model = RNN(args.input_size, self.args.label_cnt, args.hidden_size, self.device).to(self.device)
        elif args.model == "GRU":
            self.model = GRU(args.input_size, self.args.label_cnt, args.hidden_size, self.device).to(self.device)
        elif args.model == "LSTM":
            self.model = LSTM(args.input_size, self.args.label_cnt, args.hidden_size, self.device, isBidirectional = False).to(self.device)
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss().to(self.device)
        self.scheduler = MultiStepLR(self.optim, milestones=[80, 120, 160], gamma=1.0) 
        
    def train(self):
        print("{}开始训练{}".format("*"*15,"*"*15))
        for i in range(self.args.epoch):
            lossVal = 0
            deVal = []
            self.model.train()
            for datas, labels, dataLens in tqdm(self.trainLoader):
                datas = datas.to(self.args.device)
                labels = labels.to(self.args.device)
                # print(data)
                dataPacked = nn.utils.rnn.pack_padded_sequence(datas, dataLens, batch_first=True, enforce_sorted=False)
                # print(dataPacked.data)
                y, hidden = self.model(dataPacked)
                loss = self.criterion(y, labels)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                lossVal += loss.item()
                for i in range(y.shape[0]):
                    for j in range(len(datas)):
                        deVal.append(abs(y[i][j].item()-labels[i][j].item()))
            deVal = np.array(deVal)
            avgDeVal = np.mean(deVal)
            medianDeVal = np.median(deVal)
            if avgDeVal > self.trainAvgDeVal:
                self.trainAvgDeVal = avgDeVal
            if i % 5 == 0:
                self.test()
            self.scheduler.step()
            print("第{}次训练，loss={:.4f} avgDeVal={:.4f} medianDeVal={}".format(i, lossVal, avgDeVal, medianDeVal))
    
    def test(self):
        self.model.eval()
        deVal = []
        for datas, labels, dataLens in tqdm(self.trainLoader):
            datas = datas.to(self.args.device)
            labels = labels.to(self.args.device)
            # print(data)
            dataPacked = nn.utils.rnn.pack_padded_sequence(datas, dataLens, batch_first=True, enforce_sorted=False)
            # print(dataPacked.data)
            y, hidden = self.model(dataPacked)
            for i in range(y.shape[0]):
                for j in range(len(datas)):
                    deVal.append(abs(y[i][j].item()-labels[i][j].item()))
        deVal = np.array(deVal)
        avgDeVal = np.mean(deVal)
        medianDeVal = np.median(deVal)
        if avgDeVal > self.testAvgDeVal:
            self.testAvgDeVal = avgDeVal
        print("测试集 avgDeVal={:.4f} medianDeVal={}".format(avgDeVal, medianDeVal))
        
    def save(self, id):
        if id == 0:
            torch.save(self.model.state_dict(),"./pt/TModel_{}_{}_testACC_{:.3f}.pt".format(int(time.time()),str(self.model),self.testAvgDeVal))
        else:
            torch.save(self.model.state_dict(),"./pt/TModel_{}_{}_trainACC_{:.3f}.pt".format(int(time.time()),str(self.model),self.trainAvgDeVal))
        