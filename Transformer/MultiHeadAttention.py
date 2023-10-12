import torch
import numpy as np
from torch import nn, einsum
from einops import rearrange, repeat, reduce
from torchinfo import summary
from rich import console

console = console.Console()
device = 'cuda' if torch.device('cuda') else 'cpu'


class MultiHeadAttention(nn.Module):
    def __init__(self, head_num, emb_d):
        super(MultiHeadAttention, self).__init__()
        self.head_num = head_num
        self.emb_d = emb_d
        self.dk = (emb_d // head_num) ** (1 / 2)
        self.qkv_layer = nn.Linear(emb_d, emb_d * 3, bias=False)
        self.out_attention = nn.Linear(emb_d, emb_d, bias=False)

    def forward(self, x, mask=None, kv=None):
        qkv = self.qkv_layer(x)
        q, k, v = tuple(rearrange(qkv, pattern='b t (h k d) -> k b h t d', h=self.head_num, k=3))
        if kv is not None:
            # decoder 部分
            # kk = repeat(kv, 'b t (d)->b t (d h)', h=self.head_num)
            k = rearrange(kv, 'b t (h d)-> b h t d', h=self.head_num)
            v = k
        energy = einsum('... i d , ... j d -> ... i j', q, k)
        energy = energy / self.dk
        if mask is not None:
            # console.log(mask)
            energy += mask
            # console.log(energy[0][0])
        attention = torch.softmax(energy, dim=-1)
        # console.log(attention[0][0])
        xx = einsum('... i j , ... j d -> ... i d', attention, v)
        xx = rearrange(xx, 'b h t d -> b t (h d)')
        xx = self.out_attention(xx)
        # console.log(xx)
        return xx


if __name__ == '__main__':
    MHA = MultiHeadAttention().to(device)
    # testt = np.rand()
    # print(MHA())
    print(summary(MHA, [1, 3, 144]))
