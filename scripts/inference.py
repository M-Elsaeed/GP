import cv2
import numpy as np
from cv2 import cv2
import silence_tensorflow.auto  # pylint: disable=unused-import
import tensorflow as tf
from tensorflow import keras

gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

model = keras.models.load_model(f"./trainedModels/example3aug.h5")


# url = 'http://192.168.1.89:8080/video'
url = 'http://192.168.1.101:8080/video'
cap = cv2.VideoCapture(url)
# cap = cv2.VideoCapture("../test/right_test.mp4")


cv2.namedWindow("output", cv2.WINDOW_NORMAL)
while(True):
    ret, frame = cap.read()
    # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if frame is not None:
        cv2.imshow('output',frame)
        frame = frame / 255
        x_to_predict = np.array([cv2.resize(frame, dsize=(224, 224))])
        # print()
        yHat = model.predict(x_to_predict)[0]
        print(yHat, end=" ")
        # print(len(yHat))
        if yHat[0] > yHat[1] and yHat[0] > yHat[2]:
            print("L")
        elif yHat[1] > yHat[0] and yHat[1] > yHat[2]:
            print("R")
        else:
            print("F")
    
    q = cv2.waitKey(1)
    if q == ord("q"):
        break
cv2.destroyAllWindows()