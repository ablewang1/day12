import torch
from matplotlib import pyplot as plt
from torch import nn
import random
from torch.nn import functional as F
import adabound as AD

x = torch.arange(-1,1,0.1)
x11 = torch.arange(-1,0,0.05) #预测
x11 = torch.unsqueeze(x11,dim=1)
# y = x**3+x**2+x+4+torch.randn(20)
y = x.pow(3) + torch.rand(20)/10
# plt.plot(x,y,'.')
# plt.show()
# x = torch.unsqueeze(x,dim=0)  #w*x
x = torch.unsqueeze(x,dim=1)  #x*w
# print(x.shape)
x = x.float()
y = y.float()
# exit()
def sin(x):
    return torch.log(x)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        #6层
        self.w1 = torch.nn.Parameter(torch.randn(1,32))
        self.b1 = torch.nn.Parameter(torch.zeros(32))
        self.w2 = torch.nn.Parameter(torch.randn(32,64))
        self.b2 = torch.nn.Parameter(torch.zeros(64))
        self.w3 = torch.nn.Parameter(torch.randn(64,128))
        self.b3 = torch.nn.Parameter(torch.zeros(128))
        self.w4 = torch.nn.Parameter(torch.randn(128,64))
        self.b4 = torch.nn.Parameter(torch.zeros(64))
        self.w5 = torch.nn.Parameter(torch.randn(64,32))
        self.b5 = torch.nn.Parameter(torch.zeros(32))
        self.w6 = torch.nn.Parameter(torch.randn(32,1))
        self.b6 = torch.nn.Parameter(torch.zeros(1))

        ## self.layer1 = nn.Linear(1,8,bias=False)
        # #x*w
        # self.w1 = torch.nn.Parameter(torch.randn(1,8))
        # self.b1 = torch.nn.Parameter(torch.zeros(8))
        # self.w2 = torch.nn.Parameter(torch.randn(8,16))
        # self.b2 = torch.nn.Parameter(torch.zeros(16))
        # self.w3 = torch.nn.Parameter(torch.randn(16,32))
        # self.b3 = torch.nn.Parameter(torch.zeros(32))
        # self.w4 = torch.nn.Parameter(torch.randn(32,64))
        # self.b4 = torch.nn.Parameter(torch.zeros(64))
        # self.w5 = torch.nn.Parameter(torch.randn(64,128))
        # self.b5 = torch.nn.Parameter(torch.zeros(128))
        # self.w6 = torch.nn.Parameter(torch.randn(128,64))
        # self.b6 = torch.nn.Parameter(torch.zeros(64))
        # self.w7 = torch.nn.Parameter(torch.randn(64,32))
        # self.b7 = torch.nn.Parameter(torch.zeros(32))
        # self.w8 = torch.nn.Parameter(torch.randn(32,16))
        # self.b8 = torch.nn.Parameter(torch.zeros(16))
        # self.w9 = torch.nn.Parameter(torch.randn(16,1))
        # self.b9 = torch.nn.Parameter(torch.zeros(1))
        # # self.w10 = torch.nn.Parameter(torch.randn(8,1))
        # # self.b10 = torch.nn.Parameter(torch.zeros(1))

        # #w*x
        # self.w1 = torch.nn.Parameter(torch.randn(8,1))
        # self.b1 = torch.nn.Parameter(torch.zeros(20))
        # self.w2 = torch.nn.Parameter(torch.randn(16,8))
        # self.b2 = torch.nn.Parameter(torch.zeros(20))
        # self.w3 = torch.nn.Parameter(torch.randn(32,16))
        # self.b3 = torch.nn.Parameter(torch.zeros(20))
        # self.w4 = torch.nn.Parameter(torch.randn(64,32))
        # self.b4 = torch.nn.Parameter(torch.zeros(20))
        # self.w5 = torch.nn.Parameter(torch.randn(128,64))
        # self.b5 = torch.nn.Parameter(torch.zeros(20))
        # self.w6 = torch.nn.Parameter(torch.randn(64,128))
        # self.b6 = torch.nn.Parameter(torch.zeros(20))
        # self.w7 = torch.nn.Parameter(torch.randn(32,64))
        # self.b7 = torch.nn.Parameter(torch.zeros(20))
        # self.w8 = torch.nn.Parameter(torch.randn(16,32))
        # self.b8 = torch.nn.Parameter(torch.zeros(20))
        # self.w9 = torch.nn.Parameter(torch.randn(1,16))
        # self.b9 = torch.nn.Parameter(torch.zeros(1))

    def forward(self,x):

        fd1 = F.tanh(torch.matmul(x,self.w1) + self.b1)
        fd2 = F.tanh(torch.matmul(fd1,self.w2) + self.b2)
        fd3 = F.tanh(torch.matmul(fd2,self.w3) + self.b3)
        fd4 = F.tanh(torch.matmul(fd3,self.w4) + self.b4)
        fd5 = F.tanh(torch.matmul(fd4,self.w5) + self.b5)
        fd6 = torch.matmul(fd5,self.w6) + self.b6
        return fd6

        # #x*w
        # fd1 = torch.matmul(x,self.w1) + self.b1
        # fd2 = torch.matmul(fd1,self.w2) + self.b2
        # fd3 = torch.matmul(fd2,self.w3) + self.b3
        # fd4 = torch.matmul(fd3,self.w4) + self.b4
        # fd5 = torch.matmul(fd4,self.w5) + self.b5
        # fd6 = torch.matmul(fd5,self.w6) + self.b6
        # fd7 = torch.matmul(fd6,self.w7) + self.b7
        # fd8 = torch.matmul(fd7,self.w8) + self.b8
        # fd9 = torch.matmul(fd8,self.w9) + self.b9

        # fd10 = torch.matmul(fd9,self.w10) + self.b10
        # #w*x
        # fd1 = F.tanh(torch.matmul(self.w1,x) + self.b1)
        # fd2 = F.tanh(torch.matmul(self.w2,fd1) + self.b2)
        # fd3 = F.tanh(torch.matmul(self.w3,fd2) + self.b3)
        # fd4 = F.tanh(torch.matmul(self.w4,fd3) + self.b4)
        # fd5 = F.tanh(torch.matmul(self.w5,fd4) + self.b5)
        # fd6 = F.tanh(torch.matmul(self.w6,fd5) + self.b6)
        # fd7 = F.tanh(torch.matmul(self.w7,fd6) + self.b7)
        # fd8 = F.elu(torch.matmul(self.w8,fd7) + self.b8)
        # fd9 = torch.matmul(self.w9,fd8) + self.b9
        # ## print(fd9.mean())
        # return fd9

if __name__ == '__main__':
    net = Net()

    # opt = torch.optim.Adam(net.parameters(),lr=0.1)
    opt = AD.AdaBound(net.parameters(),lr=0.1,final_lr=0.001)
    loss1 = torch.nn.MSELoss()
    plt.ion()
    for i in range(100):
        z = net(x)
        # print(z)
        loss = loss1(z,y)

        opt.zero_grad()

        loss.backward()
        opt.step()

        print(loss.item())
        plt.cla()
        plt.scatter(x,y)
        y1 = net(x)

        x1 = x.reshape(20)
        y1 = y1.reshape(20)
        plt.plot(x1,y1.detach().numpy(),"r")

        plt.title(loss.item())
        plt.pause(0.01)
    # plt.scatter(x11.reshape(20),x11.reshape(20).pow(3)+torch.rand(20))
    y11 = net(x11)  # 预测
    plt.plot(x11.reshape(20), y11.reshape(20).detach().numpy(), 'g') #预测
    plt.ioff()
    plt.show()
