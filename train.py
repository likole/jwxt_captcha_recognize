import keras
import numpy as np
from keras.models import load_model
from tensorflow_serving.session_bundle import exporter
from data.get_train_data import get_image_and_labels

def test():
    model = load_model("model3.h5")

    y_test = np.load("data/y_test.npy")
    acc = model.evaluate(np.load("data/x_test.npy"), [y_test[0], y_test[1], y_test[2], y_test[3]])
    print(acc)