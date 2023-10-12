from MyTrU.Transformer.MultiHeadAttention import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EncoderBlock(nn.Module):
    def __init__(self, mlp_d, head_num, emb_d, dropout):
        super(EncoderBlock, self).__init__()

        self.m_attention = MultiHeadAttention(head_num, emb_d)
        self.mlp_layer = MLP(emb_d, mlp_d)
        self.drop_layer = nn.Dropout(dropout)
        self.norm_layer1 = nn.LayerNorm(emb_d)
        self.norm_layer2 = nn.LayerNorm(emb_d)

    def forward(self, x):
        x = self.m_attention(x=x)
        _x = self.drop_layer(x)
        x = _x + x
        x = self.norm_layer1(x)

        x = self.mlp_layer(x)
        x = _x + x
        x = self.norm_layer2(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_d, middle_d):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_d, middle_d),
            nn.Dropout(0.1),
            nn.GELU(),
            nn.Linear(middle_d, input_d),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.layers(x)


if __name__ == '__main__':
    Block = EncoderBlock(100, 6, 768, 0.1)
    print(summary(Block, [2, 1, 768]))
