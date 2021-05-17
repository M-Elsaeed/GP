import cv2
import numpy as np
from cv2 import cv2
import silence_tensorflow.auto  # pylint: disable=unused-import
import tensorflow as tf
from tensorflow import keras

gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# model = keras.models.load_model(f"./trainedmodel.h5")
model = keras.models.load_model(f"./color_normalized_nodropout.h5")


url = 'http://192.168.1.112:8080/video'
cap = cv2.VideoCapture(url)


cv2.namedWindow("output", cv2.WINDOW_NORMAL)
while(True):
    ret, frame = cap.read()
    if frame is not None:
        cv2.imshow('output',frame)
        frame = frame / 255
        x_to_predict = np.array([cv2.resize(frame, dsize=(270, 480))])
        print(model.predict(x_to_predict))
    q = cv2.waitKey(1)
    if q == ord("q"):
        break
cv2.destroyAllWindows()