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
import matplotlib.pyplot as plt
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
        selected_col = ['hour', 'T (degC)', 'H2OC (mmol/mol)', 'rho (g/m**3)', 'sh (g/kg)', 'Tpot (K)', 'VPmax (mbar)']
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
                # tmpData = []
                # tmpLabel = []
                tmpData = tmpData[5:7]
                tmpLabel = tmpLabel[5:7]
        
    def __getitem__(self, item):
        _input = [element for l in self.input[item] for element in l]
        _label = [element for l in self.label[item] for element in l]
        dataLen = len(_input)
        if dataLen<self.maxLen:
            _input += [[0 for _ in range(self.input_size)] for _ in range(self.maxLen-dataLen)]
        if len(_label)< 288:
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
        selected_col = ['hour', 'T (degC)', 'H2OC (mmol/mol)', 'rho (g/m**3)', 'sh (g/kg)', 'Tpot (K)', 'VPmax (mbar)']

        # 归一化
        TScaler = []
        for col in selected_col:
            scaler = MinMaxScaler()
            data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))
            if col == "T (degC)":
                TScaler.append(scaler)
        # 划分训练集和测试集
        train = data[data['year'].isin(range(2009, 2015))]
        test = data[data['year'].isin(range(2015, 2017))]
        train_group = train.groupby('year_month_day')
        test_group = test.groupby('year_month_day')
        
        return train_group, test_group, TScaler

class TModel(nn.Module):
    def __init__(self, args, trainData, testData, TScaler):
        super(TModel, self).__init__()
        # 定义参数
        self.args = args
        self.lr = args.lr
        self.batchSize = args.batch_size
        self.device = torch.device(args.device)
        self.labelCnt = args.label_cnt
        self.testAvgDeVal = 100
        self.trainAvgDeVal = 100
        self.TScaler = TScaler
        
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
        elif args.model == "BiLSTM":
            self.model = LSTM(args.input_size, self.args.label_cnt, args.hidden_size, self.device, isBidirectional = True).to(self.device)
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss().to(self.device)
        self.scheduler = MultiStepLR(self.optim, milestones=[30, 60, 90], gamma=0.5) 
        
    def train(self):
        print("{}开始训练{}".format("*"*15,"*"*15))
        for e in range(self.args.epoch):
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
                loss = self.criterion(labels, y)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                lossVal += loss.item()
                for i in range(y.shape[0]):
                    for j in range(288):
                        deVal.append(abs(y[i][j].item()-labels[i][j].item()))
            deVal = np.array(deVal)
            avgDeVal = np.mean(deVal)
            medianDeVal = np.median(deVal)
            if avgDeVal < self.trainAvgDeVal:
                self.trainAvgDeVal = avgDeVal
            if e %10 == 0 and e!=0:
                self.test()
            self.scheduler.step()
            print("第{}次训练，loss={:.4f} avgDeVal={:.4f} medianDeVal={}".format(e, lossVal, avgDeVal, medianDeVal))
    
    def test(self, isDraw = True):
        self.model.eval()
        deVal = []
        truthVal = []
        predVal = []
        with torch.no_grad():
            for datas, labels, dataLens in tqdm(self.testLoader):
                datas = datas.to(self.args.device)
                labels = labels.to(self.args.device)
                # print(data)
                dataPacked = nn.utils.rnn.pack_padded_sequence(datas, dataLens, batch_first=True, enforce_sorted=False)
                # print(dataPacked.data)
                y, hidden = self.model(dataPacked)
                y = y.cpu()
                labels = labels.cpu()
                for i in range(y.shape[0]):
                    for j in range(288):
                        deVal.append(abs(y[i][j].item()-labels[i][j].item()))
                        truthVal.append(labels[i][j].item())
                        predVal.append(y[i][j].item())
            if isDraw==True:
                plt.plot(truthVal[0:288], 'r', label="Truth")
                plt.plot(predVal[0:288], 'g', label="Predict")
                plt.legend()
                plt.show()
            # truthVal = np.array(truthVal).reshape(-1, 1)
            # predVal = np.array(predVal).reshape(-1, 1)
            truthVal = self.TScaler[0].inverse_transform(np.array(truthVal).reshape(-1, 1))
            predVal = self.TScaler[0].inverse_transform(np.array(predVal).reshape(-1, 1))
            deVal = np.array(deVal)
            avgDeVal = np.mean(deVal)
            medianDeVal = np.median(deVal)
            if avgDeVal < self.testAvgDeVal:
                self.testAvgDeVal = avgDeVal
                self.save(0)
            if isDraw==True:
                plt.plot(truthVal[0:288], 'r', label="Truth")
                plt.plot(predVal[0:288], 'g', label="Predict")
                plt.legend()
                plt.show()
                # plt.savefig("./{}_{}.png".format(str(self.model), int(time.time())),dpi=500)
            print("测试集 avgDeVal={:.4f} medianDeVal={}".format(avgDeVal, medianDeVal))
        
    def save(self, id):
        if id == 0:
            torch.save(self.model.state_dict(),"./pt/TModel_{}_{}_testACC_{:.3f}.pt".format(int(time.time()),str(self.model),self.testAvgDeVal))
        else:
            torch.save(self.model.state_dict(),"./pt/TModel_{}_{}_trainACC_{:.3f}.pt".format(int(time.time()),str(self.model),self.trainAvgDeVal))
        