import numpy as np
import math
from tracker import faceItem

def calDist(face1, face2):
    return math.sqrt(((face1.x+face1.w/2)-(face2.x+face2.w/2))**2+((face1.y+face1.h/2)-(face2.y+face2.h/2))**2)

def id2color(id):
    if id % 10 == 0:
        #白
        return (255, 255, 255)
    elif id % 10 == 1:
        # 巧克力
        return (30, 105, 210)
    elif id % 10 == 2:
        # 绿
        return (0, 255, 0)
    elif id % 10 == 3:
        # 天蓝
        return (230, 216, 173)
    elif id % 10 == 4:
        # 什青
        return (139, 139, 0)
    elif id % 10 == 5:
        # 橄榄
        return (0, 128, 128)
    elif id % 10 == 6:
        # 紫
        return (128, 0, 128)
    elif id % 10 == 7:
        # 紫罗兰
        return (238, 130, 238)
    elif id % 10 == 8:
        # 深灰
        return (0, 69, 255)
    elif id % 10 == 9:
        # 藏青
        return (130, 0, 75)