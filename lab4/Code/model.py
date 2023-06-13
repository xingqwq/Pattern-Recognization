import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, device, layer_num = 1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.device = device
        
        # 定义内部参数
        # self.all_weight = []
        # for _ in range(self.layer_num):
        self.u = nn.Linear(input_size, hidden_size)
        self.w = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size//2)
        self.classifier = nn.Linear(hidden_size//2, output_size)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        hidden = torch.zeros((x.sorted_indices.shape[0], self.hidden_size)).to(self.device)
        output = torch.zeros((x.sorted_indices.shape[0], self.hidden_size//2)).to(self.device)
        index = 0
        i = 0
        
        while i < x.data.shape[0]: 
            input = x.data[i:i+x.batch_sizes[index]]
            id = x.sorted_indices[:x.batch_sizes[index]]
            hidden[id] = self.u(input) + self.w(hidden[id])
            hidden[id] = self.tanh(hidden[id])
            output[id] = self.sigmoid(self.v(hidden[id]))
            i += x.batch_sizes[index]
            index += 1
        y = self.classifier(output)
        
        return y, hidden
    
    def __str__(self) -> str:
        return "RNN"
    
class GRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, device, layer_num = 1):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.device = device
        
        # 定义内部参数
        # self.all_weight = []
        # for _ in range(self.layer_num):
        self.reset_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.update_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.h_gate = nn.Linear(input_size  + hidden_size, hidden_size)
        self.output_gate = nn.Linear(hidden_size, hidden_size//2)
        self.classifier = nn.Linear(hidden_size//2, output_size)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        hidden = torch.zeros((x.sorted_indices.shape[0], self.hidden_size)).to(self.device)
        output = torch.zeros((x.sorted_indices.shape[0], self.hidden_size//2)).to(self.device)
        ones = torch.ones((x.sorted_indices.shape[0], self.hidden_size)).to(self.device)
        index = 0
        i = 0
        
        while i < x.data.shape[0]: 
            input = x.data[i:i+x.batch_sizes[index]]
            id = x.sorted_indices[:x.batch_sizes[index]]
            compined = torch.cat((input,hidden[id]),dim=1)
            z = self.sigmoid(self.update_gate(compined))
            r = self.sigmoid(self.reset_gate(compined))
            compined_ = torch.cat((input,r*hidden[id]),dim=1)
            h_hat = self.tanh(self.h_gate(compined_))
            hidden[id] = (ones[id]-z)*hidden[id] + z*h_hat
            output[id] = self.sigmoid(self.output_gate(hidden[id]))
            i += x.batch_sizes[index]
            index += 1
        y = self.classifier(output)
        
        return y, hidden
    
    def __str__(self) -> str:
        return "GRU"
    
class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, device, layer_num = 1, isBidirectional = False):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.device = device
        self.isBidirectional = isBidirectional
        
        # 定义内部参数
        # self.all_weight = []
        # for _ in range(self.layer_num):
        self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.c_gate = nn.Linear(input_size  + hidden_size, hidden_size)
        self.h_gate = nn.Linear(input_size  + hidden_size, hidden_size)
        self.output_gate = nn.Linear(hidden_size+input_size, hidden_size)
        if self.isBidirectional == True:
            self.classifier = nn.Linear(hidden_size*2, output_size)
        else:
            self.classifier = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.isBidirectional == False:
            hidden = torch.zeros((x.sorted_indices.shape[0], self.hidden_size)).to(self.device)
            c = torch.zeros((x.sorted_indices.shape[0], self.hidden_size)).to(self.device)
            output = torch.zeros((x.sorted_indices.shape[0], self.hidden_size)).to(self.device)
            index = 0
            i = 0
            
            while i < x.data.shape[0]: 
                input = x.data[i:i+x.batch_sizes[index]]
                id = x.sorted_indices[:x.batch_sizes[index]]
                combined = torch.cat((input,hidden[id]),dim=1)
                f_t = self.sigmoid(self.forget_gate(combined))
                i_t = self.sigmoid(self.input_gate(combined))
                c_hat = self.tanh(self.c_gate(combined))
                c[id] = f_t*c[id]+i_t*c_hat
                output[id] = self.sigmoid(self.output_gate(combined))
                hidden[id] = output[id]*self.tanh(c[id])
                i += x.batch_sizes[index]
                index += 1
            y = self.classifier(hidden)
        
            return y, hidden

        else:
            # 正向
            hidden = torch.zeros((x.sorted_indices.shape[0], self.hidden_size)).to(self.device)
            c = torch.zeros((x.sorted_indices.shape[0], self.hidden_size)).to(self.device)
            output = torch.zeros((x.sorted_indices.shape[0], self.hidden_size)).to(self.device)
            index = 0
            i = 0
            
            while i < x.data.shape[0]: 
                input = x.data[i:i+x.batch_sizes[index]]
                id = x.sorted_indices[:x.batch_sizes[index]]
                combined = torch.cat((input,hidden[id]),dim=1)
                f_t = self.sigmoid(self.forget_gate(combined))
                i_t = self.sigmoid(self.input_gate(combined))
                c_hat = self.tanh(self.c_gate(combined))
                c[id] = f_t*c[id]+i_t*c_hat
                output[id] = self.sigmoid(self.output_gate(combined))
                hidden[id] = output[id]*self.tanh(c[id])
                i += x.batch_sizes[index]
                index += 1
                
            # 反向
            hidden_ = torch.zeros((x.sorted_indices.shape[0], self.hidden_size)).to(self.device)
            c_ = torch.zeros((x.sorted_indices.shape[0], self.hidden_size)).to(self.device)
            output_ = torch.zeros((x.sorted_indices.shape[0], self.hidden_size)).to(self.device)
            index = len(x.batch_sizes)-1
            i = len(x.data)
            
            while i >0: 
                input = x.data[i-x.batch_sizes[index]:i]
                id = x.sorted_indices[:x.batch_sizes[index]]
                combined = torch.cat((input,hidden_[id]),dim=1)
                f_t = self.sigmoid(self.forget_gate(combined))
                i_t = self.sigmoid(self.input_gate(combined))
                c_hat = self.tanh(self.c_gate(combined))
                c_[id] = f_t*c_[id]+i_t*c_hat
                output_[id] = self.sigmoid(self.output_gate(combined))
                hidden_[id] = output_[id]*self.tanh(c_[id])
                i -= x.batch_sizes[index]
                index -= 1
            o = torch.cat((hidden,hidden_),dim=1)
            y = self.classifier(o)
        
            return y, o
    
    def __str__(self) -> str:
        if self.isBidirectional==True:
            return "BiLSTM"
        else:
            return "LSTM"