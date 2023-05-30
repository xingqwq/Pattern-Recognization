import torch
from train_and_test.data_process import data_process
import time
import pandas as pd
from tensorboardX import SummaryWriter
from tqdm import tqdm


class Train:

    def __init__(self, epoch, batch_size, size, criterion, optimizer, scheduler, model, transform):
        """
        :param epoch: epoch
        :param batch_size: batch_size
        :param size: 图像被resize为size*size
        :param criterion: 损失函数
        :param optimizer: 优化器
        :param scheduler: 学习率调度程序
        :param model: 模型，直接传入类名即可
        """
        self.epoch = epoch
        self.batch_size = batch_size
        self.size = size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model(12).to(self.device)
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer(self.model.parameters(), lr=0.001)
        self.scheduler = scheduler(self.optimizer, [30, 50], 0.1)
        self.writer = SummaryWriter('./result')
        self.validAcc = 0
        print("{}{}{}".format("*"*15,"准备开始读取数据","*"*15))
        self.label2name_dict, self.train_loader, self.valid_loader, self.test_loader, self.test_number, self.num2name_dict = \
            data_process(self.batch_size, self.size, transform)
        print("{}{}{}".format("*"*15,"已完成初始化，准备开始训练","*"*15))

    def train(self):
        """
        对self.model使用self.train_loader进行训练
        :return:
        """
        start = time.time()
        now_iter = 0
        for epoch in range(self.epoch):
            correct = 0
            total = 0
            total_loss = 0
            self.model.train()
            for i, (images, labels) in tqdm(enumerate(self.train_loader)):
                now_iter += 1
                images = images.to(self.device)
                labels = labels.to(self.device)
                # 前向传播
                outputs = self.model(images).to(self.device)
                loss = self.criterion(outputs, labels).to(self.device)
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # 统计准确率和最新的loss
                _, predicted = torch.max(outputs, 1)  # max函数返回最大值和索引的元组，我们仅需用到索引值作为标签
                correct += (predicted == labels).sum().item()
                total += predicted.size(0)
                total_loss += float(loss.item())
            if(epoch % 5 == 0):
                self.valid()
            now = time.time()
            self.writer.add_scalar(str(self.model) + '_loss', total_loss, epoch + 1)
            self.writer.add_scalar(str(self.model) + '_accuracy', correct / total, epoch + 1)
            print('epoch ' + str(epoch) + ': Accuracy on train dataset '
                  + str(correct / total) + ' total loss: ' + str(total_loss))
            print('lasts:' + str(now - start) + ' s')
        self.writer.close()

    def valid(self):
        self.model.eval()
        correct = 0
        total = 0
        for images, labels in self.valid_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            # 前向传播
            outputs = self.model(images).to(self.device)
            # 统计准确率和最新的loss
            _, predicted = torch.max(outputs, 1)  # max函数返回最大值和索引的元组，我们仅需用到索引值作为标签
            correct += (predicted == labels).sum().item()
            total += predicted.size(0)
        if(correct/total > self.validAcc):
            self.validAcc = correct/total
            self.save()
        print("{}验证集正确率为：{}{}".format("*"*15,correct/total,"*"*15))
        
    
    def generate_test_csv(self, path):
        """
        用当前已经生成好的模型来生成结果csv
        :return:
        """
        test_model = self.model
        ans = list()
        column = list(['file', 'species'])
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        with torch.no_grad():
            for images, label in self.test_loader:
                images = images.to(self.device)
                outputs = test_model(images)
                _, predicted = torch.max(outputs.data, 1)
                # print(images)
                # print(label)
                # print(predicted)
                for j in range(len(label)):
                    predict = self.label2name_dict[str(predicted[j].item())]
                    image_name = label[j]
                    ans.append([str(image_name), str(predict)])
        write_ans = pd.DataFrame(columns=column, data=ans)
        write_ans.to_csv('./result/'+str(self.model)+'_ans.csv')

    def save(self):
        """
        保存当前检查点到save_model目录之下
        :return:
        """
        torch.save(self.model.state_dict(), './save_model/{}_{:.4f}.ckpt'.format(str(self.model), self.validAcc))
