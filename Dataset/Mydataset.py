import torch
from torch import nn
from torch.utils.data import Dataset
import pandas as pd
import nibabel as nib
from einops import rearrange
from torchvision import transforms as F
from rich.console import Console

console = Console()
transs = F.Compose([
    F.ToTensor(),
    # F.Normalize(0.5, 0.5),
    F.Resize((256, 256)),
    F.ConvertImageDtype(torch.float),
])


def trans(x):
    return transs(x)


class Mydataset(Dataset):
    def __init__(self, csv_path):
        super(Dataset, self).__init__()
        self.data = pd.read_csv(csv_path, dtype=str)

    def __len__(self):
        # print(f'len:{len(self.data)}')
        return len(self.data)

    def __getitem__(self, idx):
        data_dict = self.data.iloc[idx]

        img = nib.load(data_dict['img']).get_fdata()
        mask = nib.load(data_dict['mask']).get_fdata()
        # 192 156 12
        return {'img': trans(img), 'mask': trans(mask), 'img_path': data_dict['img'],
                'mask_path': data_dict['mask']}
