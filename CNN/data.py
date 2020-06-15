import numpy as np
import matplotlib.pyplot as plt

#获取数据
train_data = np.load("data/train_image_data.npz")
test_data = np.load("data/test_image_data.npz")
#获取训练集
train_image_list = train_data['train_image_list']
train_image_label = train_data['train_image_label']
img = train_image_list[1]
print(train_image_list[1])
print(train_image_label[1])
plt.imshow(img.reshape([100, 100,3]))
plt.show()
