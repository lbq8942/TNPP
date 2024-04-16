import torch
import torch.nn as nn
import torch.nn.functional as tnf
from torch.autograd import grad
import time

class MLP(nn.Module):#这个就是一个两层MLP，好像也没有什么特别的啊。

    def __init__(self,input_dim,hidden_dim,output_dim,dropout=0.05):
        super(MLP,self).__init__()
        self.linear1=nn.Linear(input_dim,hidden_dim)
        self.hidden_drop=nn.Dropout(p=0.05)#原来0.05
        self.linear2=nn.Linear(hidden_dim,output_dim)#这个维度其实就是事件类型数量。

    def forward(self,x):
        #[bsize,seqlen,hdim]
        x=self.linear1(x)
        x=tnf.relu(x)
        x=self.hidden_drop(x)
        x=self.linear2(x)
        return x#[bsize,seqlen,marksnum],至于softmax之后再用。

class TimePara(nn.Module):#输入hidden以及当前，得到关于时间的参数。

    def __init__(self,input_dim,hidden_dim,output_dim,flownum,flowlen,num_marks):
        super(TimePara,self).__init__()

        self.flownum,self.flowlen,self.num_marks=flownum,flowlen,num_marks

        self.linear1=nn.Linear(input_dim,hidden_dim)
        self.linear2=nn.Linear(hidden_dim,hidden_dim)
        self.weights=[]
        self.biases=[]
        self.linear_weight=nn.Linear(hidden_dim,output_dim)#输出参数的weight。
        self.linear_bias=nn.Linear(hidden_dim,output_dim)#输出参数的bias
        self.hidden_drop=nn.Dropout(p=0.10)
        # self.flowdrop=nn.Dropout2d(p=0.05)



    def forward(self,x):
        #[bsize,marksnum,hdim]
        flownum, flowlen,num_marks=self.flownum,self.flowlen,self.num_marks
        x=self.linear1(x)
        x=tnf.relu(x)#第一层结束。
        x=self.linear2(x)
        x=tnf.relu(x)#第二层结束。
        x=self.hidden_drop(x)
        weight, bias=self.linear_weight(x),self.linear_bias(x)#[bsize,num_marks,flowlen*flownum]
        bsize=len(weight)
        # weight,bias=self.hidden_drop(weight),self.hidden_drop(bias)
        weight=weight.view(-1,flownum,flowlen)#其他的东西，我们全都变成bsize部分。这样肯定是可以的。这样的画，下面就不需要改动了。
        bias=bias.view(-1,flownum,flowlen)#
        weight=weight**2+1e-7#这样就保证了严格为正数。不过好像，这样平方之后，这个数字通常会特别的小，他妈的。
        # weight=torch.abs(weight)+1e-8#试过了这个，但是整体上好像差不多，没有太大区别。
        #需要保证weight为非负数，从而可以保证F为增函数。另外，由于我需要可逆，从而非负数还不行，必须为正数。
        #我的方法就是：1.平方2.绝对值函数3.sigmoid。但是sigmoid问题在于weight取值受限在了0,1,会不会太小了，另外，后面好像还会继续tanh，会不会更加容易造成梯度消失
        #bias可以为负数。
        return weight,bias


class activation():
    def __init__(self,beta=1,th=20,device=None):
        self.beta=beta
        self.th=th
        self.fact=nn.Softplus(beta=beta,threshold=th)
        thx=torch.tensor([th/beta])
        self.thy=self.fact(thx).to(device)

    def forward(self,x):
        return self.fact(x)
    def backward(self,y):
        mask=y<=self.thy
        x=y.clone()
        x[mask]=torch.log(torch.exp(self.beta*y[mask])-1)/self.beta#其他的那些mask就是不变，从而数值计算稳定。
        return x

