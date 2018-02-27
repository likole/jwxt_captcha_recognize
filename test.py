import string

from keras.models import load_model

from data.get_train_data import get_image_and_labels
from keras import Model
from keras.layers import *
from keras.utils import plot_model

x_test=np.load("data/x_test.npy")
y_test=np.load("data/y_test.npy")

model =load_model("model.h5")
acc=model.evaluate(x_test,[y_test[0],y_test[1],y_test[2],y_test[3]])
print(acc)

model =load_model("model10.h5")
acc=model.evaluate(x_test,[y_test[0],y_test[1],y_test[2],y_test[3]])
print(acc)