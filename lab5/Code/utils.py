import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import mat4py
import os

class dataPoints(Dataset):
    def __init__(self):
        self.data = mat4py.loadmat("./points.mat")['xx']

    def __getitem__(self, idx):
        xy = torch.tensor(np.array(self.data[idx]),dtype=torch.float)
        return xy

    def __len__(self):
        return len(self.data)

def gradPenalty(D, x, fake, device):
    alpha = torch.rand(x.shape[0], 1).to(device)
    alpha = alpha.expand_as(x)
    mid = alpha*x+(1-alpha)*fake
    mid.requires_grad_()
    pred = D(mid)
    grad = torch.autograd.grad(pred, mid, grad_outputs=torch.ones_like(pred),
                               create_graph=True, retain_graph=True,
                               only_inputs=True)[0]
    gp = torch.pow(grad.norm(2, dim=1)-1, 2).mean()
    return gp

def drawResult(D, G, e, args, device, data):
    data = np.array(data)
    # Plot
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.title('Result_{}_{}_{}'.format(e, args.model, args.optimizer))
    plt.xlabel("X", fontsize=18)
    plt.ylabel("Y", fontsize=18)
    
    # Generate
    x = torch.randn(1000, 10).to(device)
    y = G(x)
    y = y.to('cpu').detach()
    xy = np.array(y)
    
    # Draw Background
    x_min = -0.5
    y_min = -0.5
    x_max = y_max = 1.5
    i = x_min
    background = []
    while i <= x_max - 0.01:
        j = y_min
        while j <= y_max - 0.01:
            background.append([i, j])
            j += 0.01
        background.append([i, y_max])
        i += 0.01
    j = y_min
    while j <= y_max - 0.01:
        background.append([i, j])
        j += 0.01
        background.append([i, y_max])
    background.append([x_max, y_max])
    color = D(torch.Tensor(background).to(device))
    background = np.array(background)
    cm = plt.cm.get_cmap('gray')
    sc = plt.scatter(background[:, 0], background[:, 1], c=np.squeeze(color.cpu().data), cmap=cm)
    plt.colorbar(sc)
    
    # Draw Real
    plt.scatter(data[:, 0], data[:, 1], c='green', s=10)
    
    # Draw Pred
    plt.scatter(xy[:,0], xy[:,1], c='red', s=10)
    
    # Save Pig
    path = "./png/{}_{}".format(args.model,args.optimizer)
    if os.path.exists(path) == False:
        os.mkdir(path)
    plt.savefig("./{}/epoch{}_{}.jpg".format(path, str(e).zfill(4), args.model), dpi=600)
    plt.close()
    # plt.show()


def drawLoss(lossDVal, lossGVal, args):
    x = np.arange(len(lossDVal))
    fig = plt.figure(2, figsize=(16, 16), dpi=150)
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    ax1.plot(x, lossDVal, 'r', label='lossD')
    ax2.plot(x, lossGVal, 'g', label='lossG')
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')
    # Save fig
    plt.savefig("./loss_png/{}_{}.png".format(args.model,args.optimizer), dpi=600)
    plt.close()