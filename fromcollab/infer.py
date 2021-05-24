import cv2
import numpy as np
from cv2 import cv2
import silence_tensorflow.auto  # pylint: disable=unused-import
import tensorflow as tf
from tensorflow import keras

gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


model = keras.models.load_model(f"./all.h5")


frames = np.array(
    [
        cv2.resize(cv2.imread("../test/f0.jpg"), dsize=(270, 480)),
        cv2.resize(cv2.imread("../test/f1.jpg"), dsize=(270, 480)),
        cv2.resize(cv2.imread("../test/l0.jpg"), dsize=(270, 480)),
        cv2.resize(cv2.imread("../test/r0.jpg"), dsize=(270, 480)),
    ]
)
print(frames.shape)
frames = frames / 255

YHat = model.predict(frames)
print(YHat.shape)

for yHat in YHat:
    if yHat[0] > yHat[1] and yHat[0] > yHat[2]:
        print(yHat, "L")
    elif yHat[1] > yHat[0] and yHat[1] > yHat[2]:
        print(yHat, "R")
    else:
        print(yHat, "F")