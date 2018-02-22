import numpy as np
import keras
from keras.models import load_model

model=load_model("model2.h5")
y_test=np.load("data/y_test.npy")

acc=model.evaluate(np.load("data/x_test.npy"),[y_test[0],y_test[1],y_test[2],y_test[3]])
print(acc)