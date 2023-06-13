import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import GCNConv

class Select(nn.Module):
    def __init__(self):
        super(Select, self).__init__()
    
    def forward(self,x):
        out,(h_n,c_n) = x
        return out[:,0,:]
    
class RobotClassify(nn.Module):
    def __init__(self):
        super(RobotClassify, self).__init__()
        self.descriptionFeature = nn.Sequential(
                                    nn.LSTM(768, 64, 2,bidirectional=True),
                                    Select(),
                                    nn.Dropout(0.3),
                                    nn.Linear(128,8),
                                    nn.LeakyReLU())
        self.textLinear = nn.Sequential(
                            nn.Dropout(0.2),
                            nn.Linear(768,8),
                            nn.LeakyReLU())
        self.proLinear = nn.Sequential(
                            nn.Linear(10,8),
                            nn.LeakyReLU())
        self.outLinear = nn.Sequential(
                            nn.Linear(24,2),
                            nn.Sigmoid())
    
    def avrTweet(self,data):
        for i,eachResult in enumerate(data):
            if i == 0:
                continue
            else:
                data[0].add(eachResult)
        return data[0]/data.shape[0]
    
    def forward(self,des,tweet,profile):
        # 获取description信息
        desR = []
        tweetR = []
        for i,eachDes in enumerate(des):
            desT = self.descriptionFeature(eachDes)
            desR.append(desT[0,:])
        desR = torch.stack(desR)
        # 获取Tweet信息
        tweetR = self.textLinear(tweet)
        # 获取Profile信息
        profile = self.proLinear(profile)
        # 信息融合
        x = torch.cat((desR,tweetR,profile),dim=1)
        x = self.outLinear(x)
        return x

class RobotGraphClassify(nn.Module):
    def __init__(self):
        super(RobotGraphClassify, self).__init__()
        self.screenFeature = nn.Sequential(
                                    nn.Linear(768,16),
                                    nn.LeakyReLU())
        self.descriptionFeature = nn.Sequential(
                                    nn.Linear(768,16),
                                    nn.LeakyReLU()
        )
        self.textLinear = nn.Sequential(
                            nn.Linear(768,16),
                            nn.LeakyReLU()
        )
        self.proLinear = nn.Sequential(
                            nn.Linear(5,8),
                            nn.LeakyReLU()
        )
        self.personalLinear = nn.Sequential(
                            nn.Linear(7,8),
                            nn.LeakyReLU()
        )
        self.linear = nn.Sequential(
                            nn.Linear(64,64),
                            nn.LeakyReLU()
        )
        self.gcn1 = GCNConv(64 , 64)
        self.gcn2 = GCNConv(64 , 64)
        self.gcn3 = GCNConv(64 , 64)
        self.dropout = nn.Dropout(0.15)
        self.outLinear = nn.Sequential(
                            nn.Linear(64,64),
                            nn.LeakyReLU(),
                            nn.Linear(64,2),
                            nn.Sigmoid())
    
    def forward(self,screen,des,tweet,profile,personal,edge,edgeRelation):
        # 获取description信息
        screen = self.screenFeature(screen)
        screen = self.dropout(screen)
        des = self.descriptionFeature(des)
        des = self.dropout(des)
        # 获取Tweet信息
        tweet = self.textLinear(tweet)
        tweet = self.dropout(tweet)
        # 获取Profile信息
        profile = self.proLinear(profile)
        personal = self.personalLinear(personal)
        # 信息融合
        x1 = torch.cat((screen,des,tweet,profile,personal), dim=1)
        x = self.dropout(x1)
        x = self.linear(x)
        x = self.gcn1(x,edge)
        x = x + x1
        x = self.dropout(x)
        x = self.gcn2(x,edge)
        x = x + x1
        x = self.dropout(x)
        x = self.gcn3(x,edge)
        x = x + x1
        x = self.dropout(x)
        x = self.outLinear(x)
        return x
