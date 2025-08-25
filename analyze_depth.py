import cv2
import numpy as np
import os

# 分析深度图
depth_file = "dataset/promptda/137/depth_maps/000135.png"
img = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)

print("深度图信息:")
print(f"数据类型: {img.dtype}")
print(f"形状: {img.shape}")
print(f"最小值: {img.min()}")
print(f"最大值: {img.max()}")
print(f"平均值: {img.mean():.2f}")
print(f"标准差: {img.std():.2f}")

# 显示前10个像素值
print("前10个像素值:")
print(img.flatten()[:10])

# 分析非零像素
nonzero_pixels = img[img > 0]
if len(nonzero_pixels) > 0:
    print(f"\n非零像素统计:")
    print(f"非零像素数量: {len(nonzero_pixels)}")
    print(f"非零像素最小值: {nonzero_pixels.min()}")
    print(f"非零像素最大值: {nonzero_pixels.max()}")
    print(f"非零像素平均值: {nonzero_pixels.mean():.2f}")
else:
    print("没有非零像素")
