import numpy as np
import cv2
import random
import os

"""
    随机挑选CNum张图片，进行按通道计算均值mean和标准差std
    先将像素从0～255归一化至 0-1 再计算
"""

total_txt_path = os.path.join("../..", "total.txt")

CNum = 2527  # 挑选多少图片进行计算

img_w, img_h = 224, 224  # 模型的输入尺寸
imgs = np.zeros([img_w, img_h, 3, 1])
means, stdevs = [], []

with open(total_txt_path, 'r') as f:
    lines = f.readlines()
    random.shuffle(lines)  # shuffle , 随机挑选图片

    for i in range(CNum):
        img_path = lines[i].rstrip().split()[0]

        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_h, img_w))

        img = img[:, :, :, np.newaxis]
        imgs = np.concatenate((imgs, img), axis=3)
        # print(i)

imgs = imgs.astype(np.float32) / 255.

for i in range(3):
    pixels = imgs[:, :, i, :].ravel()  # 拉成一行
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))

means.reverse()  # BGR --> RGB
stdevs.reverse()

print(f"normMean = {means:.3f}")
print(f"normStd = {stdevs:.3f}")
print(f'transforms.Normalize(normMean = {means:.3f}, normStd = {stdevs:.3f})')

# normMean = [0.673, 0.639, 0.604]
# normStd = [0.208, 0.209, 0.231]
# transforms.Normalize(normMean = [0.673, 0.639, 0.604], normStd = [0.208, 0.209, 0.231])
