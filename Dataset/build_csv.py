import os
import pandas as pd
from rich.console import Console
console = Console()
from  rich import inspect
# 指定包含 NIfTI 文件的目录
img_dir = r'C:\Users\CUI\Desktop\JETB\AI\MyTrU\Dataset\data\imgs'
mask_dir = r'C:\Users\CUI\Desktop\JETB\AI\MyTrU\Dataset\data\lables'

# 初始化一个空的 DataFrame
df = pd.DataFrame(columns=["img", "mask"])

# 遍历目录中的文件
for filename in os.listdir(img_dir):
    if filename.endswith("_avg.nii.gz"):
        img_path = os.path.join(img_dir, filename)
        mask_filename = filename.replace("_avg.nii.gz", "_avg_seg.nii.gz")
        mask_path = os.path.join(mask_dir, mask_filename)
        df = pd.concat([df, pd.DataFrame({"img": [img_path], "mask": [mask_path]})])

# 指定保存 DataFrame 的输出路径
output_path = r'C:\Users\CUI\Desktop\JETB\AI\MyTrU\Dataset\data\dataframe.csv'

# 将 DataFrame 保存为 CSV 文件
df.to_csv(output_path, index=False)

print(f"DataFrame 已保存至 {output_path}")
inspect(df.iloc[1, :])
