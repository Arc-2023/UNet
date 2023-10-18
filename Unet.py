import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from MyTrU.UNet.TransUNet import *
from MyTrU.Dataset.Mydataset import *
from MyTrU.Dataset.Mydataloader import *
from rich.console import Console
from MyTrU.UNet.Loss import *
from torch.optim import Adam
from MyTrU.utils import *
from torchvision import transforms
from torch import nn

console = Console()
# 12 192 156
csv_path = r'C:\Users\CUI\Desktop\JETB\AI\MyTrU\Dataset\data\dataframe.csv'
batch_size = 1  # *12
epoches = 50
T_0 = 1
T_mult = 2


def main():
    model = TransUNet(img_h=256, img_w=256, in_channels=1, patch_d=2, final_out_channels=10, block_size=2, mlp_d=72,
                      dropout=0.2, head_num=2, classes=1).to('cuda')
    dataset = Mydataset(csv_path)
    train_loader, test_loader = get_dataloader(dataset, batchsize=batch_size, ratio=0.9, shuffle=False)
    opti = Adam(model.parameters(), lr=0.003)
    schd_lr = CosineAnnealingWarmRestarts(opti, T_0=T_0, T_mult=T_mult)
    # loss_fn = focal_loss()
    loss_t = 0
    loss_test = 0
    pred = ''
    mask = ''
    for epoch in tqdm(range(epoches)):
        model.train()
        for imgs, masks in tqdm(train_loader, leave=False):
            # console.log(imgs.shape)
            pred = model(imgs.to('cuda'))
            mask = masks
            # console.log(pred.shape)
            # console.log(masks.shape)
            loss = dice_loss(pred, masks.to('cuda'))
            loss_t += loss
            opti.zero_grad()
            loss.backward()
            opti.step()
            schd_lr.step()
        model.eval()
        for imgs, masks in tqdm(test_loader, leave=False):
            # console.log(imgs.shape)
            with torch.no_grad():
                pred = model(imgs.to('cuda'))
                mask = masks
                # console.log(pred.shape)
                # console.log(masks.shape)
                loss = dice_loss(pred, masks.to('cuda'))
                loss_test += loss
        console.log(f'epoch:{epoch};train_loss:{loss_t / len(train_loader)};test_loss:{loss_test / len(test_loader)}')
        loss_t = loss_test = 0
        if epoch % 5 == 0:
            show_random_tensors_from_batch(mask, pred)


if __name__ == '__main__':
    main()
