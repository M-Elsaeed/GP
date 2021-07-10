import cv2
import numpy as np
import cv2
import silence_tensorflow.auto  # pylint: disable=unused-import
import tensorflow as tf
from tensorflow import keras
from utilities import cropFaceDNN, cropFaceHaarCascade

gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# model = keras.models.load_model(f"./trainedModels/blkshirts2.h5")
model = keras.models.load_model(f"./trainedModels/croppedModelFC.h5")



# url = 0 for camera
url = 'http://192.168.1.102:8080/video'

cap = cv2.VideoCapture(url)
# cap = cv2.VideoCapture("../test/right_test.mp4")
def cautiousImshow(name, image):
    try:
        cv2.imshow(name, image)
    except:
        print("image invalid")

cv2.namedWindow("output", cv2.WINDOW_NORMAL)
while(True):
    ret, frame = cap.read()
    # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if frame is not None:
        # cv2.imshow('feed',frame)
        frame = cropFaceDNN(frame)
        if frame is not None:
            x_to_predict = np.array([cv2.resize(frame, dsize=(224, 224))]).astype(np.float32)
            x_to_predict = x_to_predict / 255.0
            # print()
            yHat = model.predict(x_to_predict)[0]
            prediction = None
            # print(yHat, end=" ")
            # print(len(yHat))
            if yHat[0] > yHat[1] and yHat[0] > yHat[2]:
                prediction="L"
            elif yHat[1] > yHat[0] and yHat[1] > yHat[2]:
                prediction="R"
            else:
                prediction="F"
            cautiousImshow('output',cv2.putText(frame, prediction, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2))
    
    q = cv2.waitKey(1)
    if q == ord("q"):
        break
cv2.destroyAllWindows()