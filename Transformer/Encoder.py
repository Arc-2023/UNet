from MyTrU.Transformer.EncoderBlock import *
from rich import console

console = console.Console()


class Encoder(nn.Module):
    def __init__(self, block_size, mlp_d, head_num, emb_d, dropout):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(
            EncoderBlock(mlp_d, head_num, emb_d, dropout) for _ in range(block_size)
        )

    def forward(self, x):
        # b t emb_d
        for layer in self.layers:
            x = layer(x)
        return x
