import torch
from torch import optim
from matplotlib import pyplot as plt
from torch.nn import functional as F

x = torch.arange(-1,1,0.1)

y = x.pow(3)

x = torch.unsqueeze(x,dim=0)


def log(x):
    return x**3
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.w1 = torch.nn.Parameter(torch.rand(20,8))
        self.b1 = torch.nn.Parameter(torch.zeros(8))
        self.w2 = torch.nn.Parameter(torch.rand(8,32))
        self.b2 = torch.nn.Parameter(torch.zeros(32))
        self.w3 = torch.nn.Parameter(torch.rand(32,20))
        self.b3 = torch.nn.Parameter(torch.zeros(20))

    def forward(self,x):
        fd1 = torch.matmul(x,self.w1) + self.b1
        fd2 = torch.matmul(fd1,self.w2) + self.b2
        fd3 = torch.matmul(fd2,self.w3) + self.b3
        # print("fd3:",fd3)
        return fd3

if __name__ == '__main__':
    net = Net()
    opt = optim.Adam(net.parameters(),lr=0.1)
    los = torch.nn.MSELoss()
    plt.ion()
    for i in range(2000):
        z = net(x)
        loss = los(z,y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss)
        if i%5==0:
            plt.cla()
            # x = x.reshape(y.shape)
            plt.scatter(x,y)
            print(x,z.detach().numpy())
            plt.plot(x.reshape(20),z.detach().numpy().reshape(20),'r')
            plt.pause(0.1)
    plt.ioff()
    plt.show()

