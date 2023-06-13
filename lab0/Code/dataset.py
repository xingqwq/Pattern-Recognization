import torch
import torch.utils.data as DataUtil
import numpy as np
import json
import pickle as pkl
import pandas as pd
import os
from tqdm import tqdm, trange
import time

def splitData(userPath, edgePath, trainPath, testPath):
    # 读取用户数据
    with open(userPath,'r') as file:
        tmpData = json.load(file)
        userData = {}
        for i in tmpData:
            if "profile" in i:
                userData[i['ID']] = i
                userData[i['ID']].update({"neighbors":[]})
    # 读取边缘数据
    with open(edgePath,'r') as file:
        tmpData = json.load(file)
        for i in tmpData:
            if i["relation"] == "followers":
                userData[i["seed_user_id"]]["neighbors"].append([0, i["relation_user_id"]])
            else:
                userData[i["seed_user_id"]]["neighbors"].append([1, i["relation_user_id"]])
    
    # 读取训练数据
    data = pd.read_csv(trainPath,sep='\t')
    robotCnt = []
    normalCnt = [] 
    for i in data.iloc():
        if not str(i['ID']) in userData:
            continue
        if i['Label'] == 0 :
            normalCnt.append([i['ID'], i['Label']])
        else:
            robotCnt.append([i['ID'], i['Label']])
    
    # 打乱数据
    robotCnt = np.random.permutation(np.array(robotCnt))
    normalCnt = np.random.permutation(np.array(normalCnt))
    trainData = np.random.permutation(np.vstack((robotCnt[0:int(len(robotCnt)*0.85)],normalCnt[0:int(len(normalCnt)*0.65)])))
    validData = np.random.permutation((np.vstack((robotCnt[int(len(robotCnt)*0.85):],normalCnt[int(len(normalCnt)*0.65):]))))

    # 读取测试数据
    data = pd.read_csv(testPath,sep='\t')
    testData = []
    for i in data.iloc():
        testData.append([i['ID'], 0])
        
    # 生成id2index
    pData = np.vstack((trainData, validData, testData))
    id2index = {}
    for i in range(0,len(pData)):
        id2index.update({str(pData[i][0]):i})
    
    # 生成nolabel数据
    # noLabel = []
    # ck = len(id2index)
    # for i in userData:
    #     if i not in id2index:
    #         noLabel.append([int(i), 0])
    #         id2index.update({str(i):ck})
    #         ck += 1
    
    return userData, trainData, validData, testData, id2index

class TextBasedDataSet(DataUtil.Dataset):
    def __init__(self,path):
        self.path = path
        self.file_list = os.listdir(self.path)
        
    def __getitem__(self, index): 
        f = open(self.path+"data_"+str(index)+".pkl","rb")
        data = pkl.load(f)
        f.close()
        
        des = torch.tensor(data[0],dtype = torch.float).squeeze(0)
        tweet = torch.tensor(data[1],dtype = torch.float)
        profile = torch.tensor(data[2],dtype = torch.float)
        label = torch.tensor(data[3],dtype = torch.float)

        return (des, tweet, profile, label)
    
    def __len__(self):
        return len(self.file_list)-1

class dataSet:
    def __init__(self,data,userData,maxLen,device,tokenizer,bertModel):
        self.data = data
        self.userData = userData
        self.maxLen = maxLen
        self.device = device
        self.tokenizer = tokenizer
        self.bertModel = bertModel
        self.des = []
        self.tweet = []
        self.profile = []
        self.label = []
        self.padding = [0 for _ in range(0,768)]
        self.makeData()
    
    def makeData(self):
        with torch.no_grad():
            for i in range(0,len(self.data)):
                self.getItemInfo(i)
    
    def getData(self):
        # self.des = torch.tensor(self.des)
        # self.tweet = torch.tensor(self.tweet)
        # self.profile = torch.tensor(self.profile,dtype=torch.float)
        # self.label = torch.tensor(self.label)
        return self.des, self.tweet, self.profile, self.label
    
    def getItemInfo(self, index): 
        self.des.append(self.getDes(int(self.data[index][0])))
        self.tweet.append(self.getTweet(int(self.data[index][0])))
        self.profile.append(self.getProfile(int(self.data[index][0])))
        if self.data[index][1] == 1:
            labelT = [1.,0.]
        else:
            labelT = [0.,1.]
        self.label.append(labelT)
    
    def getToken(self,data):
        token = self.tokenizer.tokenize(data)
        token = token[0:((self.maxLen-2) if len(token) > (self.maxLen-2) else len(token))]
        return ['[CLS]'] + token + ['[SEP]'] + ['[PAD]' for _ in range(0,max(0,self.maxLen-len(token)-2))]
    
    def getMask(self,token):
        return [1 if i != '[PAD]' else 0 for i in token]

    def getDes(self,id):
        data = "The description is not available!"
        try:
            data = self.userData[str(id)]["profile"]["description"]
        except Exception as e:
            print("出现异常:{}".format(e))
            data = "The description is not available!"
        if data is None or len(data) == 0:
            data = "The description is not available!"
        token = self.getToken(data)
        mask = torch.tensor(self.getMask(token)).unsqueeze(0).to(self.device)
        token = torch.tensor(self.tokenizer.convert_tokens_to_ids(token)).unsqueeze(0).to(self.device)
        output = self.bertModel(token,mask)
        return output[0].to("cpu").numpy().tolist()
        
    def getTweet(self,id):
        tokenIdData = []
        attentionMaskData = []
        if self.userData[str(id)]['tweet'] != None:
            for i in self.userData[str(id)]['tweet']:
                token = self.getToken(i)
                attentionMaskData.append(self.getMask(token))
                tokenIdData.append(self.tokenizer.convert_tokens_to_ids(token))
        else:
            token = self.getToken("The tweet is not available!")
            attentionMaskData.append(self.getMask(token))
            tokenIdData.append(self.tokenizer.convert_tokens_to_ids(token))
        tokenIdData = torch.tensor(tokenIdData).to(self.device)
        attentionMaskData = torch.tensor(attentionMaskData).to(self.device)
        output = self.bertModel(tokenIdData,attentionMaskData)
        output = output[0].to("cpu")
        # 计算每个推文特征
        result = []
        for i, eachTweet in enumerate(output):
            for j, eachWord in enumerate(eachTweet):
                if j == 0:
                    data = eachWord
                else:
                    data += eachWord
            result.append((data/self.maxLen).numpy().tolist())
        if len(result) >self.maxLen:
            result = result[0:self.maxLen]
        else:
            ck = len(result)
            for i in range(0,self.maxLen-ck):
                result.append(self.padding)
        return result
    
    def getProfile(self,id):
        feature = ["followers_count","friends_count","listed_count","favourites_count","statuses_count"]
        if self.userData[str(id)]["profile"] is None:
            return [0 for _ in range(0,5)]
        output = []
        for i in feature:
            if self.userData[str(id)]["profile"][i] is None:
                output.append(0)
            elif self.userData[str(id)]["profile"][i] == 'true':
                output.append(1)
            elif self.userData[str(id)]["profile"][i] == 'false':
                output.append(0)
            else:
                output.append(int(self.userData[str(id)]["profile"][i]))
        
        return output
    
# 加载图模型数据
class dataSetGraph:
    def __init__(self,data,userData,maxLen,device,tokenizer,bertModel):
        self.data = data
        self.userData = userData
        self.maxLen = maxLen
        self.device = device
        self.tokenizer = tokenizer
        self.bertModel = bertModel
        self.screen = []
        self.des = []
        self.tweet = []
        self.profile = []
        self.personalIden = []
        self.label = []
        self.padding = [0 for _ in range(0,768)]
        self.makeData()
    
    def makeData(self):
        with torch.no_grad():
            for i in range(0,len(self.data)):
                self.getItemInfo(i)
    
    def getData(self):
        # self.des = torch.tensor(self.des)
        # self.tweet = torch.tensor(self.tweet)
        # self.profile = torch.tensor(self.profile,dtype=torch.float)
        # self.label = torch.tensor(self.label)
        return self.screen,self.des, self.tweet, self.profile, self.personalIden, self.label
    
    def getItemInfo(self, index): 
        self.screen.append(self.getScreen(int(self.data[index][0])))
        self.des.append(self.getDes(int(self.data[index][0])))
        self.tweet.append(self.getTweet(int(self.data[index][0])))
        self.profile.append(self.getProfile(int(self.data[index][0])))
        self.personalIden.append(self.getPersonIden(int(self.data[index][0])))
        if self.data[index][1] == 1:
            labelT = [1.,0.]
        else:
            labelT = [0.,1.]
        self.label.append(labelT)
    
    def getToken(self,data):
        token = self.tokenizer.tokenize(data)
        token = token[0:((self.maxLen-2) if len(token) > (self.maxLen-2) else len(token))]
        return ['[CLS]'] + token + ['[SEP]'] + ['[PAD]' for _ in range(0,max(0,self.maxLen-len(token)-2))]
    
    def getMask(self,token):
        return [1 if i != '[PAD]' else 0 for i in token]

    def getScreen(self,id):
        data = "The screen_name is not available!"
        try:
            data = self.userData[str(id)]["profile"]["screen_name"]
        except Exception as e:
            print("出现异常:{}".format(e))
            data = "The screen_name is not available!"
        if data is None or len(data) == 0:
            data = "The screen_name is not available!"
        token = self.getToken(data)
        mask = torch.tensor(self.getMask(token)).unsqueeze(0).to(self.device)
        token = torch.tensor(self.tokenizer.convert_tokens_to_ids(token)).unsqueeze(0).to(self.device)
        output = self.bertModel(token,mask)
        output = output[0].to("cpu")
        output = torch.mean(output, dim = 1)
        return output.numpy().tolist()

    def getDes(self,id):
        data = "The description is not available!"
        try:
            data = self.userData[str(id)]["profile"]["description"]
        except Exception as e:
            print("出现异常:{}".format(e))
            data = "The description is not available!"
        if data is None or len(data) == 0:
            data = "The description is not available!"
        token = self.getToken(data)
        mask = torch.tensor(self.getMask(token)).unsqueeze(0).to(self.device)
        token = torch.tensor(self.tokenizer.convert_tokens_to_ids(token)).unsqueeze(0).to(self.device)
        output = self.bertModel(token,mask)
        output = output[0].to("cpu")
        output = torch.mean(output, dim = 1)
        return output.numpy().tolist()
        
    def getTweet(self,id):
        tokenIdData = []
        attentionMaskData = []
        if self.userData[str(id)]['tweet'] != None:
            for i in self.userData[str(id)]['tweet']:
                token = self.getToken(i)
                attentionMaskData.append(self.getMask(token))
                tokenIdData.append(self.tokenizer.convert_tokens_to_ids(token))
        else:
            token = self.getToken("The tweet is not available!")
            attentionMaskData.append(self.getMask(token))
            tokenIdData.append(self.tokenizer.convert_tokens_to_ids(token))
        tokenIdData = torch.tensor(tokenIdData).to(self.device)
        attentionMaskData = torch.tensor(attentionMaskData).to(self.device)
        output = self.bertModel(tokenIdData,attentionMaskData)
        output = output[0].to("cpu")
        # 计算每个推文特征
        result = []
        for i, eachTweet in enumerate(output):
            for j, eachWord in enumerate(eachTweet):
                if j == 0:
                    data = eachWord
                else:
                    data += eachWord
            result.append((data/self.maxLen))
        result = torch.stack(result)
        return torch.mean(result,dim = 0).unsqueeze(0).numpy().tolist()
    
    def getProfile(self,id):
        feature = ["followers_count","friends_count","listed_count","favourites_count","statuses_count"]
        if self.userData[str(id)]["profile"] is None:
            return [0 for _ in range(0,5)]
        output = []
        for i in feature:
            if self.userData[str(id)]["profile"][i] is None:
                output.append(0)
            else:
                output.append(int(self.userData[str(id)]["profile"][i]))
        return output

    def getPersonIden(self,id):
        feature = ["protected","geo_enabled","verified","contributors_enabled",
                   "is_translator","is_translation_enabled","default_profile_image"]
        if self.userData[str(id)]["profile"] is None:
            return [0 for _ in range(0,7)]
        output = []
        for i in feature:
            if self.userData[str(id)]["profile"][i] is None:
                output.append(0)
            elif self.userData[str(id)]["profile"][i] == True:
                output.append(1)
            elif self.userData[str(id)]["profile"][i] == False:
                output.append(0)
            else:
                output.append(0)
        return output