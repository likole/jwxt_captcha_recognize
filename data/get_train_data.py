from PIL import Image
import os
import numpy as np
import string
import random
from matplotlib import pyplot as plt

# 获取验证码列表
label_list = os.listdir("/home/likole/Downloads/jwxt/train")

# 字符列表
characters = string.digits + string.ascii_lowercase

# 图片的宽,高,字符数,字符种类数
width, height, n_len, n_class = 180, 60, 4, len(characters)

# 数据量
train_len = len(label_list)


def get_image_and_labels(batch_size=32):
    x = np.zeros([batch_size, height, width, 1])
    y = [np.zeros([batch_size, n_class]) for _ in range(n_len)]
    while True:
        for i in range(batch_size):
            label = label_list[random.randrange(0, train_len)]
            image = np.array(Image.open("/home/likole/Downloads/jwxt/train/" + label).convert('L'))
            x[i, :] = image.reshape([height, width, 1])
            for j, ch in enumerate(label):
                y[j][i, :] = 0
                y[j][i, characters.find(ch)] = 1
        yield x, y
