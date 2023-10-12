import torch
from Trocr.Trans.Encoder import *
from rich import console

console = console.Console()


class VIT(nn.Module):
    def __init__(self, block_size, mlp_d, head_num, emb_d, dropout, in_channels, img_x, img_y, patch_d):
        super(VIT, self).__init__()
        self.img_w = img_x
        self.img_h = img_y
        self.patch_d = patch_d
        self.in_channels = in_channels
        self.token_d = in_channels * (patch_d ** 2)
        self.token_num = (img_y // patch_d) * (img_x // patch_d)

        self.pos_emb = nn.Parameter(
            torch.randint(low=0, high=5, size=(self.token_num, emb_d), device=device).float())
        nn.init.xavier_normal_(self.pos_emb)

        self.encoder = Encoder(block_size, mlp_d, head_num, emb_d, dropout)

        self.emb_layer = nn.Linear(self.token_d, emb_d)
        self.drop_layer = nn.Dropout(dropout)

    def forward(self, img):
        # console.log(img.shape)
        # B C H W
        img = rearrange(img, 'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)',
                        patch_x=self.patch_d,
                        patch_y=self.patch_d)
        img = self.emb_layer(img)
        # x: [b t emb_d]
        batch, tokens, _ = img.shape
        # 分类+位置初始化编码
        # cls = repeat(self.cls_emb, 'b ...-> (b batches) ...', batches=batch)
        # img = torch.cat([cls, img], dim=1)
        # console.log(img.shape)
        # console.log(self.pos_emb.shape)
        img += self.pos_emb
        # img = self.pos_emb(img)
        img = self.drop_layer(img)

        enc_out = self.encoder(img)
        return enc_out
