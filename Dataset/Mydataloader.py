import torch.utils.data

from MyTrU.Dataset.Mydataset import *
from torch.utils.data import random_split
from einops import repeat


def get_dataloader(dataset: Mydataset, batchsize, ratio, shuffle):
    train_size = int(len(dataset) * ratio)
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])
    train_dataloader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batchsize, collate_fn=col_fn,
                                                   shuffle=shuffle)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batchsize, collate_fn=col_fn,
                                                  shuffle=shuffle)

    return train_dataloader, test_dataloader


def col_fn(batch):
    imgs = []
    masks = []
    for data in batch:
        img = data['img']
        mask = data['mask']
        imgs.append(img)
        masks.append(mask)
        # console.log(img.shape)
    imgs = torch.cat([img for img in imgs], dim=0)
    masks = torch.cat([mask for mask in masks], dim=0)
    imgs = repeat(imgs, 'c h w-> c e h w', e=1)
    masks = repeat(masks, 'c h w-> c e h w', e=1)
    return imgs, masks
