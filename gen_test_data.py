from PIL import Image
import os
import numpy as np
import string

# 获取验证码列表
label_list = os.listdir("/home/likole/Downloads/jwxt/test")

# 字符列表
characters = string.digits + string.ascii_lowercase

# 图片的宽,高,字符数,字符种类数
width, height, n_len, n_class = 180, 60, 4, len(characters)

# 数据量
test_len = len(label_list)

x_train = np.zeros([test_len, height, width], dtype=np.uint8)
y_train = [np.zeros([test_len, n_class], dtype=np.uint8) for _ in range(n_len)]

for i in range(test_len):
    label = label_list[i].replace(".jpg","")
    image = np.array(Image.open("/home/likole/Downloads/jwxt/test/" + label+".jpg").convert('L'))
    x_train[i, :] = image.reshape(height,width,1)
    for j, ch in enumerate(label):
        y_train[j][i, characters.find(ch)] = 1

np.save("x_test.npy", x_train)
np.save("y_test.npy", y_train)
