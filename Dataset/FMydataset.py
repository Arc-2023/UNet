import torch
from torch.utils.data import Dataset
import pandas as pd
import nibabel as nib
from einops import rearrange

class MyDataset(Dataset):
    def __init__(self, csv_path):
        super(MyDataset, self).__init__()
        self.data = pd.read_csv(csv_path, dtype=str)
        self.samples = []  # List to store all data samples

        for idx in range(len(self.data)):
            data_dict = self.data.iloc[idx]
            img = rearrange(nib.load(data_dict['img']), 'h w c->c h w')
            mask = rearrange(nib.load(data_dict['mask']), 'h w c->c h w')
            sample = {'img': img, 'mask': mask, 'img_path': data_dict['img'], 'mask_path': data_dict['mask']}
            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
