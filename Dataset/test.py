import nibabel as nib
import torch

from rich import inspect
import matplotlib.pyplot as plt
from einops import rearrange
from rich.console import Console
from torchvision import transforms

# 192 156 12
console = Console()
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
    transforms.ConvertImageDtype(torch.float),
])


def show_nii(path):
    # path = r"C:\Users\CUI\Desktop\JETB\AI\MyTrU\Dataset\data\imgs\DET0000101_avg.nii.gz"
    # path = r"C:\Users\CUI\Desktop\JETB\AI\MyTrU\Dataset\data\lables\DET0000101_avg_seg.nii.gz"
    img = nib.load(path)
    ii = img.get_fdata()  # h w c
    # ii = rearrange(ii, 'h w c->c h w')
    ii = trans(ii)
    console.log(f'size:{ii.shape}')
    # console.log(ii[3, 100, :])
    plt.imshow(ii[1, :, :])
    plt.show()


show_nii(r'C:\Users\CUI\Desktop\JETB\AI\MyTrU\Dataset\data\lables\DET0003101_avg_seg.nii.gz')
