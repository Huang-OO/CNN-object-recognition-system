import numpy as np
import os
import matplotlib.pyplot as plt

train_path = "img/train/"  # 图片存储位置
test_path = "img/test/"  # 图片存储位置

# 获取数据
train_x_path = []
train_y_data = []

test_x_path = []
test_y_data = []


def img_data(path, x, y):
    # 读取图片路径以及生成标签数据
    for item in os.listdir(path):
        file_path = path+item
        x.append(file_path)
        if "T" in item:
            y.append([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif "r" in item:
            y.append([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        elif "w" in item:
            y.append([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        elif "m" in item:
            y.append([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        elif "d" in item:
            y.append([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        elif "cat" in item:
            y.append([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        elif "s" in item:
            y.append([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        elif "watermelon" in item:
            y.append([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        elif "C" in item:
            y.append([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        else:
            y.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    # 转化为矩阵
    x_path = np.array(x)
    y_data = np.array(y)

    # 乱序原始数据
    np.random.seed(100)
    order = np.random.permutation(len(y_data))
    x_path = x_path[order]
    y_data = y_data[order]

    # 归一化图片
    def readimg(file_path):
        img = plt.imread(file_path)
        img = img/255
        return img

    # 加载图片数据
    x_data = []
    for path in x_path:
        img = readimg(path)
        x_data.append(img)

    return x_data, y_data

train_x_data, train_y_data = img_data(train_path, train_x_path, train_y_data)

test_x_data, test_y_data = img_data(test_path, test_x_path, test_y_data)
np.savez("data/train_image_data.npz", train_image_list=train_x_data, train_image_label=train_y_data)
np.savez("data/test_image_data.npz", test_image_list=test_x_data, test_image_label=test_y_data)
print("图片信息存储完成")