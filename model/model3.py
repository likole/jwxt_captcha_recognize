import string
from data.get_train_data import get_image_and_labels
from keras import Model
from keras.layers import *
from keras.utils import plot_model

# 字符列表
characters = string.digits + string.ascii_lowercase

# 图片的宽,高,字符数,字符种类数
width, height, n_len, n_class = 180, 60, 4, len(characters)

#输入层
input_tensor=Input(shape=(height,width,1),dtype='float32')

#第一层
x = Conv2D(32, (3, 3), activation='relu')(input_tensor)
x = Conv2D(32, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

#第二层
x = Conv2D(64, (3, 3), activation='relu')(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

#第三层
x = Conv2D(128, (3, 3), activation='relu')(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

#全连接层
x = Flatten()(x)
x = Dropout(0.25)(x)
x=Dense(256,activation='relu')(x)

#输出层
x = [Dense(n_class, activation='softmax', name='c%d'%(i+1))(x) for i in range(4)]

#模型
model = Model(inputs=input_tensor, outputs=x)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

plot_model(model, to_file='model3.png')

model.fit_generator(get_image_and_labels(), samples_per_epoch=10000, nb_epoch=1,
                    nb_worker=2, pickle_safe=True)

model.save("model3.h5")

y_test=np.load("data/y_test.npy")

acc=model.evaluate(np.load("data/x_test.npy"),[y_test[0],y_test[1],y_test[2],y_test[3]])
print(acc)