import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import string
import numpy as np
from keras import Model
from keras.layers import *

# 字符列表
characters = string.digits + string.ascii_lowercase

# 图片的宽,高,字符数,字符种类数
width, height, n_len, n_class = 180, 60, 4, len(characters)

#训练集
x_train=np.load("x_train_2d_float32.npy")
y_train=np.load("y_train_float32.npy")

# 数据量
train_len=x_train.shape[0]

#reshape
x_train=x_train.reshape([train_len,height,width,1])

#输入层
input_tensor=Input(shape=(height,width,1),dtype='float32')

#第一层
x = Conv2D(32, (3, 3), activation='relu')(input_tensor)
x = Conv2D(32, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

#第二层
x = Conv2D(64, (3, 3), activation='relu')(input_tensor)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

#第三层
x = Conv2D(128, (3, 3), activation='relu')(input_tensor)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

#第四层
x = Conv2D(256, (3, 3), activation='relu')(input_tensor)
x = Conv2D(256, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

#输出层
x = Flatten()(x)
x = Dropout(0.25)(x)
x = [Dense(n_class, activation='softmax', name='c%d'%(i+1))(x) for i in range(4)]

#模型
model = Model(input=input_tensor, output=x)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train,[y_train[0],y_train[1],y_train[2],y_train[3]],epochs=2)