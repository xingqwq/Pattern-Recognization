from queue import Queue
import time
from scipy.optimize import linear_sum_assignment
import numpy as np
import utils

class faceItem:
    def __init__(self, x, y, w, h, maskStatus, score):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.maskStatus = maskStatus
        self.score = score

class track:
    def __init__(self):
        self.faceQueue = []
        self.maskStatus = [0, 0]
        self.isMask = -1
        self.status = 0
        self.isUpdate = 0
        self.updateTime = time.time()
    
    def new(self, face:faceItem, updateTime):
        self.status = 1
        self.updateTime = updateTime
        self.faceQueue.append(face)
        self.maskStatus[face.maskStatus] += 1
        self.isUpdate = 1

    def clear(self):
        self.faceQueue = []
        self.maskStatus = [0, 0]
        self.status = 0
        self.isUpdate = 0
        self.isMask = -1
        self.updateTime = time.time()
        
class tracker:
    def __init__(self, tracksNum):
        self.tracksNum = tracksNum
        self.tracks:list[track] = [track() for _ in range(self.tracksNum)]
    
    def apply(self, faceList):
        updateTime = time.time()
        if len(faceList) != 0:
            costMat = ([[0 for _ in range(len(faceList))]for _ in range(self.tracksNum)])
            for i in range(self.tracksNum):
                for j in range(0, len(faceList)):
                    if self.tracks[i].status != 0:
                        tmpDist = utils.calDist(self.tracks[i].faceQueue[-1], faceList[j])
                        if tmpDist <= 300:
                            costMat[i][j] = tmpDist
                        else:
                            costMat[i][j] = 999
                    else:
                        costMat[i][j] = 999
            row_ind, col_ind = linear_sum_assignment(costMat)
            self.match(col_ind, faceList, costMat, updateTime)
        self.checkTracks(updateTime)
        
    def match(self, result, face, costMat, updateTime):
        newList = []
        for i in range(len(result)):
            if costMat[i][result[i]] == 999:
                newList.append(face[result[i]])
            else:
                self.tracks[i].faceQueue.append(face[result[i]])
                self.tracks[i].maskStatus[face[result[i]].maskStatus] += 1
                self.tracks[i].updateTime = updateTime
                self.tracks[i].isUpdate = 1
        # 给未匹配的创建新轨迹
        for i in newList:
            for j in range(self.tracksNum):
                if self.tracks[j].status == 0:
                    self.tracks[j].new(i, updateTime)
        # 判断当前轨迹的状态
        for i in range(self.tracksNum):
            if self.tracks[i].status == 0:
                continue
            if self.tracks[i].maskStatus[0] > self.tracks[i].maskStatus[1] and sum(self.tracks[i].maskStatus) >= 5:
                self.tracks[i].isMask = 0
            elif self.tracks[i].maskStatus[0] < self.tracks[i].maskStatus[1] and sum(self.tracks[i].maskStatus) >= 5:
                self.tracks[i].isMask = 1
    
    def checkTracks(self, updateTime):
        for i in range(0, self.tracksNum):
            if updateTime - self.tracks[i].updateTime >= 1 and self.tracks[i].status == 1:
                self.tracks[i].clear()
            # 每两秒清除一次状态
            if sum(self.tracks[i].maskStatus) >= 40:
                self.tracks[i].maskStatus = [0, 0]

                