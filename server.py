import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from PIL import Image
import numpy as np
import flask
import io

from flask import send_file
from keras.models import load_model
import string

app = flask.Flask(__name__)
model = None
# 字符列表
characters = string.digits + string.ascii_lowercase

def decode(y):
    y = np.argmax(np.array(y), axis=2)[:,0]
    return ''.join([characters[x] for x in y])

@app.route("/")
def intro():
    return send_file("intro.html")

@app.route("/predict", methods=["POST"])
def predict():
    # data = {"success": False}

    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            image = np.array(image.convert("L"))
            image=image.reshape([1,60, 180, 1])
            # classify the input image and then initialize the list
            # of predictions to return to the client
            result=model.predict(image)
            # data["result"]=decode(result)
            # data["success"] = True

    # return flask.jsonify(data)
    return decode(result)

if __name__ == "__main__":
    model = load_model("model.h5")
    app.run(host='0.0.0.0')