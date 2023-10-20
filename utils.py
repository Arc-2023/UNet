import torch
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
import random

transss = transforms.ToPILImage()


def show_random_tensors_from_batch(pred, mask, num_samples=5):
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))
    for i in range(num_samples):
        random_index = random.randint(0, pred.shape[0] - 2)  # 随机选择一个索引
        pred_img = transss(pred[random_index])  # 获取随机选择的预测张量
        mask_img = transss(mask[random_index])  # 获取随机选择的真实标签张量

        # 显示预测张量
        axes[i, 0].imshow(pred_img)
        axes[i, 0].set_title(f"Sample {i + 1} - GT")

        # 显示真实标签张量
        axes[i, 1].imshow(mask_img)
        axes[i, 1].set_title(f"Sample {i + 1} - Pred")

    plt.show()
