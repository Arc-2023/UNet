import torch
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
import random

transss = transforms.ToPILImage()


def show_random_tensors_from_batch(pred, mask):
    random_index = random.randint(0, pred.shape[0] - 1)  # 随机选择一个索引
    pred_img = transss(pred[random_index])  # 获取随机选择的预测张量
    mask_img = transss(mask[random_index])  # 获取随机选择的真实标签张量

    # 创建一个包含两个子图的画布
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # 显示预测张量
    axes[0].imshow(pred_img)
    axes[0].set_title("Prediction")

    # 显示真实标签张量
    axes[1].imshow(mask_img)
    axes[1].set_title("Ground Truth")

    plt.show()
