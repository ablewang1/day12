import torch
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
import adabound as AD

x = torch.arange(-1,1,0.1)
y = x.pow(3)+torch.rand(20)/5

x = torch.unsqueeze(x,dim=1)

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # self.w1 = torch.nn.Parameter(torch.randn(1,2))
        # self.b1 = torch.nn.Parameter(torch.zeros(2))
        # self.w2 = torch.nn.Parameter(torch.randn(2,8))
        # self.b2 = torch.nn.Parameter(torch.zeros(8))
        # self.w3 = torch.nn.Parameter(torch.randn(8,16))
        # self.b3 = torch.nn.Parameter(torch.zeros(16))
        # self.w4 = torch.nn.Parameter(torch.randn(16,4))
        # self.b4 = torch.nn.Parameter(torch.zeros(4))
        # self.w5 = torch.nn.Parameter(torch.randn(4,1))
        # self.b5 = torch.nn.Parameter(torch.zeros(1))

        self.layer1 = nn.Linear(1, 4)
        self.layer2 = nn.Linear(4, 8)
        self.layer3 = nn.Linear(8, 16)
        self.layer4 = nn.Linear(16, 8)
        self.layer5 = nn.Linear(8, 1)

    def forward(self,x):

        fd1 = self.layer1(x)
        fd2 = self.layer2(fd1)
        fd3 = self.layer3(fd2)
        fd4 = self.layer4(fd3)
        fd5 = self.layer5(fd4)
        return fd5

if __name__ == '__main__':
    net = Net()
    opt = torch.optim.Adam(net.parameters(),lr=0.0001)
    # opt = AD.AdaBound(net.parameters(),lr=0.01,final_lr=0.0001)
    loss1 = torch.nn.MSELoss()

    plt.ion()
    for i in range(501):
        out = net(x.float())
        loss = loss1(out,y.float())

        opt.zero_grad()
        loss.backward()
        opt.step()

        print(loss)
        if i%5==0:
            plt.cla()
            plt.scatter(x,y)
            plt.title(loss.item())
            plt.plot(x.reshape(20),out.detach().reshape(20),'r')
            plt.pause(0.1)

    plt.ioff()
    plt.show()
