from dataclasses import dataclass
from black import Mode
import torch
import torch.nn as nn
import numpy as np
import sys
import types
import runpy
import inspect
import math
#先建个矩阵练练
a=torch.tensor(7)
print(a)
print(a.shape)
print(a.ndim)
row=torch.tensor([74,853,10])
print(row)
print(row.shape)
print(row.ndim)
print(row[0])
shsl=torch.tensor([
    [8,5,9,4],
    [89,44,2,1],
    [7,4,1,3]
])
print(shsl)
print(shsl.shape)
print(shsl.ndim)#有几根轴
print(shsl.numel())#[#3*4]
print(shsl[0])
print(shsl[2][3])
od=shsl.sum(dim=1)#把每排的四个数加起来,沿着第 1 维把 4 个格子收掉，第 0 维的 3 排留下来,
print(od)
print(od.shape)
oo=shsl.sum(dim=0)#填几，形状里第几维就从结果里消失,沿第 k 根轴，把那根轴上所有位置的数加成一个数,shape()里的轴
print(oo)
print(oo.shape)
xl=shsl.sum(dim=-1)
print(xl)
print(xl.shape)
yz=shsl.sum(dim=1,keepdim=True)#保留原轴型
print(yz)
print(shsl-yz)
print(shsl+yz)
users = torch.tensor([
    [25.,  8., 12., 3.],
    [34., 12.,  4., 7.],
    [19.,  3., 30., 1.]
])
ones = torch.ones(4,1)#4，1形状参数
print(users.shape)          
print(ones.shape)           
print(users @ ones)   # 矩阵乘法   
print((users @ ones).shape) 
print(torch.exp(users))#e的x次方
print(float('-inf'))#负无穷，是一个比任何有限数都小的数
print(5. + float('-inf'))  #加上一个有限数，它还是最小                        
print(torch.exp(torch.tensor(float('-inf'))))
s = torch.tensor([1., 1., 2.])#给 3 个分数，还 3 个数；这 3 个数都在 0 到 1 之间，加起来正好等于 1
p = torch.softmax(s, dim=-1)#把一排分数改成一排占比
print(p)                  
print(p.sum(dim=-1))
layer1 = nn.Linear(4, 2, bias=False)#一个线性层，里面存着一张权重表#输入 1 排 4 个数，出来 1 排 2 个数。新建的层里权重是随机值
x2 = torch.tensor([[1., 2., 3., 4.]])   
print(layer1(x2).shape) 
x3 = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])
print(x3.view(3, 2))#把同一串数换个摆法
print(x3.view(6))


k = torch.tensor([
    [1., 2.],
    [3., 4.],
    [5., 6.],
])
kt = k.transpose(-2, -1)#把两张轴换位置
print(k.shape)    
print(kt.shape)   
print(kt)

print('float32(1/3)        =', float(np.float32(1 / 3)))
def gelu(x):#夹在两层 nn.Linear 中间，让它们没法合并成一层
    return 0.5 * x * (1 + math.tanh(math.sqrt(2/math.pi) * (x + 0.044715*x**3)))

def linear(v, w):
    return v * w

x3 = 1.0
h = linear(x3, 0.5)
g = gelu(h)            
o = linear(g, 2.0)          
print(h, round(g, 6), round(o, 6))
r1 = torch.tensor([5., 5., 5.])
mean = r1.mean()
var = ((r1 - mean) ** 2).mean()
eps = 1e-5
print
(mean.item(), var.item())
print('带 eps :', (r1 - mean) / torch.sqrt(var + eps))
print('不带   :', (r1 - mean) / torch.sqrt(var))
class Config:                                  
    def __init__(self):                         
        self.dim = 80                           
        self.n_heads = 8                        
        self.dropout = 0.1                      

class Box:                                     
    def __init__(self):                         
        self.parts = []                         
    def __call__(self, x):                      
        return self.forward(x)                  
    def forward(self, x):                       
        return x * 2                            

class TinyAttention(Box):                      
    def __init__(self, args):                   
        super().__init__()                      
        self.n_heads = args.n_heads             
        self.head_dim = args.dim // self.n_heads  
        self.wq_width = self.head_dim * self.n_heads  
        self.parts.append('wq')                 
args = Config()                                
m = TinyAttention(args) 
print(args.dim, args.n_heads)      # 80 8
print(m.n_heads, m.head_dim)       # 8 10
print(m.wq_width)                  # 80
print(m.parts)                     # ['wq']
print(m(3))                        # 6                 
def attention(q,k,v,dropout=None):
    d_k = q.size(-1)#把某一根轴的长度变成一个普通数字
    scores = torch.matmul(q,k.transpose(-2,-1)/(d_k))**0.5
    pattern = torch.softmax(scores,dim=-1)
    if dropout !=None:
        pattern = dropout(pattern)
    return torch.matmul(pattern,v),pattern#q与k一起决定 scores，scores决定pattern,v被pattern 加权，矩阵相乘。v 全程没有参与打分，只在最后一步出现
x=torch.tensor([
    [1,85,7],
    [79,1,5],
    [7,452,1]],dtype=torch.float32)#seq3,d3
y=torch.randn(84,41)#seq84,d41
result,aw=attention(x,x,x)
yout,yw=attention(y,y,y)
print(result)
print(aw)
print(result.shape)
print(aw.shape)
print(yout.shape)
print(yw.shape)
@dataclass
class Modelargs:
    dim:int=79#名字  类型标注  默认值
    n_layers:int=6
    n_heads:int=8
    vocab_size:int=10000
    max_seq_len:int=512
    dropout:float=0.1
class MultiHeadAttention(nn.Module):
    def __init__(self,args:Modelargs,is_casual=False):
        super().__init__()
        assert args.dim%args.n_heads==0#assert判断条件是否为真，余数为 0 时 (dim // heads) × heads 正好回到 dim，余数不为 0 时，中间会差一截
        self.head_dim = args.dim // args.n_heads
        self.n_heads = args.n_heads
        self.wq=nn.Linear(args.dim,self.dim*self.n_heads,bias=False)#bias跳过if
        self.wk=nn.Linear(args.dim,self.dim*self.n_heads,bias=False)
        self.wv=nn.Linear(args.dim,self.dim*self.n_heads,bias=False)
        self.wo=nn.Linear(self.dim*self.n_heads,args.dim,bias=False)
        self.attention_dropout = nn.Dropout(args.dropout)#造出一个层对象，括号里那个数是概率
        self.residual_dropout = nn.Dropout(args.dropout)
        self.is_casual = is_casual
        if self.is_casual:
            mask = torch.full((args.max_seq_len,args.max_seq_len),float('-inf'))#占住一个位置，让它最后分不到权重。
            mask = torch.triu(mask,diagonal=1)
            self.register_buffer('mask',mask)# 把张量登记进 buffers 清单，同时把名字挂到对象上，后面 self.mask 与 hasattr(self, 'mask') 都读得到
def forward(self,q:torch.Tensor,k:torch.Tensor,v:torch.Tensor):
        batch_size,seq_len,_ = q.size()
        xq=self.wq(q).view(batch_size,seq_len,self.n_heads,self.head_dim).transpose(1,2)
        xk=self.wk(k).view(batch_size,seq_len,self.n_heads,self.head_dim).transpose(1,2)
        xv=self.wv(v).view(batch_size,seq_len,self.n_heads,self.head_dim).transpose(1,2)
        scores = torch.matmul(xq,xk.transpose(-2,-1))/self.head_dim**0.5
        if self.is_casual:
            assert hasattr(self,'mask')
            scores = scores + self.mask[:seq_len,:seq_len]
        scores = torch.softmax(scores,dim=-1).type_as(xq)
        scores = self.attention_dropout(scores)
        out = torch.matmul(scores,xv)
        out = out.transpose(1,2).contiguous().view(batch_size,seq_len)
        output = self.wo(out)
        output = self.residual_dropout(output)
        return output,scores
class LayerNorm(nn.Module):
    def __init__(self,dim,eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps
    def forward(self,x):
        mean = x.mean(dim=-1,keepdim=True)
        std = x.std(dim=-1,keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
class Encoder_Layer(nn.Module):
    def __init__(self,dim,eps=1e-5):
        super().__init__()
        self.attention = MultiHeadAttention(dim)
        self.norm1 = nn.LayerNorm(dim,eps=eps)#最后一根，保持形状#eps 是加在方差上的一个很小的正数，把分母从 0 抬开
        self.ffn = nn.Sequential(
            nn.Linear(dim,dim*4),
            nn.GELU(),
            nn.Linear(dim*4,dim)
        )
        self.norm2 = nn.LayerNorm(dim,eps=eps)
    def forward(self,x):
        attn_out,attn_weights = self.attention(x,x,x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x,attn_weights
class Encoder(nn.Module):
    def __init__(self,args:Modelargs):
        super().__init__()
        self.layers = nn.ModuleList([Encoder_Layer(args.dim) for _ in range(args.n_layers)])
    def forward(self,x):
        attn_weights_list = []
        for layer in self.layers:
            x,attn_weights = layer(x)
            attn_weights_list.append(attn_weights)
        return x,attn_weights_list
class Decoder_Layer(nn.Module):
    def __init__(self,dim,eps=1e-5):
        super().__init__()
        self.self_attention = MultiHeadAttention(dim,is_casual=True)
        self.norm1 = nn.LayerNorm(dim,eps=eps)
        self.cross_attention = MultiHeadAttention(dim)
        self.norm2 = nn.LayerNorm(dim,eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim,dim*4),
            nn.GELU(),#逐元素算，每个格子各算各的，形状不变，精确式
            nn.Linear(dim*4,dim)
        )#---造一个「按顺序过」的容器，写 self.ffn(x) 时三步依次执行，前一步的输出交给后一步

        self.norm3 = nn.LayerNorm(dim,eps=eps)
    def forward(self,x,encoder_output):
        self_attn_out,self_attn_weights = self.self_attention(x,x,x)
        x = self.norm1(x + self_attn_out)# 先做逐格加法、再做归一化
        cross_attn_out,cross_attn_weights = self.cross_attention(x,encoder_output,encoder_output)
        x = self.norm2(x + cross_attn_out)
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)
        return x,self_attn_weights,cross_attn_weights
class Decoder(nn.Module):
    def __init__(self,args:Modelargs):
        super().__init__()
        self.layers = nn.ModuleList([Decoder_Layer(args.dim) for _ in range(args.n_layers)])
    def forward(self,x,encoder_output):
        self_attn_weights_list = []
        cross_attn_weights_list = []
        for layer in self.layers:
            x,self_attn_weights,cross_attn_weights = layer(x,encoder_output)
            self_attn_weights_list.append(self_attn_weights)
            cross_attn_weights_list.append(cross_attn_weights)
        return x,self_attn_weights_list,cross_attn_weights_list
x = torch.tensor([[1.,85.,7.],[79.,1.,5.],[7.,452.,1.]])
r, w = attention(x, x, x)
print(r.shape, w.shape, w.sum(-1))   






