import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from collections import defaultdict
import cv2
import csv
import os
from tqdm import tqdm
import numpy as np

# 全局标签名-数值的映射，由于它只会在train_data_read函数之中被初始化，所以一定要先执行train_data_read函数
name2label_map = defaultdict(int)
# 全局标签数值-名的映射，由于它只会在train_data_read函数之中被初始化，所以一定要先执行train_data_read函数
label2name_map = defaultdict(str)
# 当前图片序号-图片文件名的映射，用于结果产出
num2name_map = defaultdict(str)


class MyDataset(Dataset):

    def __init__(self, data, label, transform):
        self.data = data
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.transform(self.data[item][:]), self.label[item]


def train_data_read(size):
    """
    :param size: 将图像resize为size*size
    :return: data 和 label
    """
    path = "./data/train"
    image_dir_paths = next(os.walk(path))[1]  # 获取所有子目录路径
    data = []
    # 读取数据
    for i, image_dir_path in tqdm(enumerate(image_dir_paths)):
        name2label_map[image_dir_path] = i
        label2name_map[str(i)] = image_dir_path
        image_names = next(os.walk(path + '/' + image_dir_path))[2]
        for image_name in image_names:
            image = cv2.imread(path + '/' + image_dir_path + '/' + image_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
            data.append((image, i))

    return data

def shuffle_train_data(data):
    train_img = []
    train_label = []
    valid_img = []
    valid_label = []
    
    data = np.random.permutation(np.array(data, dtype=object))
    train_img = data[0:int(round(len(data)*0.9)),0].tolist()
    train_label = data[0:int(round(len(data)*0.9)),1].tolist()
    valid_img = data[int(round(len(data)*0.9)):,0].tolist()
    valid_label = data[int(round(len(data)*0.9)):,1].tolist()
    return train_img, train_label, valid_img, valid_label

def test_data_read(size):
    """
    :param size: 将图像resize为size*size
    :return: data 和 label
    """
    path = './data/test'
    data = []
    labels = []
    image_label_map = defaultdict(str)
    test_number = 0
    with open('./data/sample_submission.csv') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        next(spamreader)
        for row in spamreader:
            image_label_map[row[0]] = row[1]

    images_names = next(os.walk(path))[2]
    for i, image_name in enumerate(images_names):
        test_number += 1
        num2name_map[str(i)] = image_name
        image = cv2.imread(path + '/' + image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        data.append(image)
        labels.append(image_name)
    return data, labels, test_number


def data_process(batch_size, size, transform):
    """
    如果要读取数据，只需要调用这个函数即可
    :param batch_size: 含义就像他的名字一样
    :param size: 图像resize之后的大小为size*size
    :param transform: 图像变换
    :return: 两个torch.utils.data.DataLoader对象的实例，分别意味着train_data_loader 和test_data_loader
    注：transform部分我不太明白什么样的比较好，于是我采用了之前实验所使用的学长的transform ————yyz
    """
    my_transform = transform
    train_data = train_data_read(size)
    train_img, train_label, valid_img, valid_label = shuffle_train_data(train_data)
    test_data, test_labels, test_number = test_data_read(size)
    train_dataset = MyDataset(train_img, train_label, my_transform)
    valid_dataset = MyDataset(valid_img, valid_label, my_transform)
    test_dataset = MyDataset(test_data, test_labels, my_transform)
    return label2name_map, torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=8,
        shuffle=True
    ),torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=8,
        shuffle=True
    ), torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=8,
        shuffle=True
    ), test_number, num2name_map
