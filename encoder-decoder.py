from dataclasses import dataclass
from black import Mode
import torch
import torch.nn as nn
import sys
import types
import runpy
import inspect
def attention(q,k,v,dropout=None):
    d_k = q.size(-1)
    scores = torch.matmul(q,k.transpose(-2,-1)/(d_k))**0.5
    pattern = torch.softmax(scores,dim=-1)
    if dropout !=None:
        pattern = dropout(pattern)
    return torch.matmul(pattern,v),pattern
x=torch.tensor([[1,85,7],[79,1,5],[7,452,1]],dtype=torch.float32)
y=torch.randn(84,41)
result,aw=attention(x,x,x)
yout,yw=attention(y,y,y)
print(result.shape)
print(aw.shape)
print(yout.shape)
print(yw.shape)
@dataclass
class Modelargs:
    dim:int=79
    n_layers:int=6
    n_heads:int=8
    vocab_size:int=10000
    max_seq_len:int=512
    dropout:float=0.1
class MultiHeadAttention(nn.Module):
    def __init__(self,args:Modelargs,is_casual=False):
        super().__init__()
        assert args.dim%args.n_heads==0
        self.head_dim = args.dim // args.n_heads
        self.n_heads = args.n_heads
        self.wq=nn.Linear(args.n_embd,self.dim*self.n_heads,bias=False)
        self.wk=nn.Linear(args.n_embd,self.dim*self.n_heads,bias=False)
        self.wv=nn.Linear(args.n_embd,self.dim*self.n_heads,bias=False)
        self.wo=nn.Linear(self.dim*self.n_heads,args.dim,bias=False)
        self.attention_dropout = nn.Dropout(args.dropout)
        self.residual_dropout = nn.Dropout(args.dropout)
        self.is_casual = is_casual
        if self.is_casual:
            mask = torch.full((args.max_seq_len,args.max_seq_len),float('-inf'))
            mask = torch.triu(mask,diagonal=1)
            self.register_buffer('mask',mask)
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
        self.norm1 = nn.LayerNorm(dim,eps=eps)
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
            nn.GELU(),
            nn.Linear(dim*4,dim)
        )
        self.norm3 = nn.LayerNorm(dim,eps=eps)
    def forward(self,x,encoder_output):
        self_attn_out,self_attn_weights = self.self_attention(x,x,x)
        x = self.norm1(x + self_attn_out)
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










if 'black' not in sys.modules:
    _fake = types.ModuleType('black')
    _fake.Mode = object
    sys.modules['black'] = _fake


print('=== ① user_code.py 顶层被原样执行 ===')

_mod = types.ModuleType('user_code')
sys.modules['user_code'] = _mod
ns = runpy.run_path('user_code.py', run_name='user_code')
_mod.__dict__.update(ns)
print('=== ① 结束 ===\n')


Modelargs = ns['Modelargs']
MHA = ns['MultiHeadAttention']


Modelargs.n_embd = property(lambda s: s.dim)

MHA.dim = property(lambda s: s.head_dim)


MHA.forward = ns['forward']


_orig_init = MHA.__init__
def _init(self, args, is_casual=False):
    if isinstance(args, int):
        args = Modelargs(dim=args, n_heads=1)
    _orig_init(self, args, is_casual)
MHA.__init__ = _init

print('=== ② 你的文件里定义的对象 ===')
for name, obj in inspect.getmembers(ns):
    if name.startswith('_'):
        continue
    if inspect.isfunction(obj) or (inspect.isclass(obj) and obj.__module__ == 'user_code'):
        print('   ', name, '->', type(obj).__name__)
print('=== ② 结束 ===\n')

torch.manual_seed(0)


r, w = ns['attention'](ns['x'], ns['x'], ns['x'])
print('③ attention(x,x,x)      ->', tuple(r.shape), tuple(w.shape))


cfg = Modelargs(dim=1, n_heads=1, n_layers=2, vocab_size=100, max_seq_len=8)
print('③ Modelargs(...)        ->', cfg)


x = torch.randn(1, 1, 1)
print('③ 喂进去的 x           ->', tuple(x.shape))
print()

# 4.4 MHA 单独调
mha = MHA(cfg, is_casual=True).eval()
o, s = mha(x, x, x)
print('④ MultiHeadAttention    -> out', tuple(o.shape), 'scores', tuple(s.shape))


enc = ns['Encoder'](cfg).eval()
dec = ns['Decoder'](cfg).eval()

enc_out, enc_w = enc(x)
dec_out, self_w, cross_w = dec(x, enc_out)

print('⑤ Encoder               ->', tuple(enc_out.shape),
      '| self-attn', tuple(enc_w[0].shape), 'x', len(enc_w), '层')
print('⑥ Decoder               ->', tuple(dec_out.shape),
      '| self-attn', tuple(self_w[0].shape), 'x', len(self_w), '层',
      '| cross-attn', tuple(cross_w[0].shape), 'x', len(cross_w), '层')
print()
print('attention / Modelargs / MultiHeadAttention / '
      'forward / Encoder_Layer / Encoder / Decoder_Layer / Decoder 都跑过了。')
