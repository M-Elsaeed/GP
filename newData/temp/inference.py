import cv2
import numpy as np
import silence_tensorflow.auto  # pylint: disable=unused-import
import tensorflow as tf
from tensorflow import keras


def comp_(a1):
    return a1[1]

def analyzeFaceDetections(detections, image):
    arrDs = []
        # loop over the detections
    i = 0
    while i < detections.shape[2]:
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > 0.95:
            box = detections[0, 0, i, 3:7] * np.array(
                [w, h, w, h]
            )
            (startX, startY, endX, endY) = box.astype("int")
            eyes = eye_cascade.detectMultiScale(cv2.cvtColor(image[startY:endY, startX:endX], cv2.COLOR_BGR2GRAY), 1.1, 4)
            if len(eyes) >= 1:
                arrDs.append([i, confidence])
        i+=1
    # print(arrDs)
    arrDs.sort(key=comp_)
    return arrDs


gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

model = keras.models.load_model(f"./trainedModels/blkshirts2.h5")

net = cv2.dnn.readNetFromCaffe(
    "./deploy.prototxt", "./res10_300x300_ssd_iter_140000.caffemodel"
)
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

url = 'http://192.168.0.106:8080/video'
cap = cv2.VideoCapture(url)


while True:
    success, image = cap.read()
    if success:
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        net.setInput(blob)
        detections = net.forward()

        arrDs = analyzeFaceDetections(detections, image)
        # print(arrDs)

        # compute the (x, y)-coordinates of the bounding box for the
        # object
        if len(arrDs):
            box = detections[0, 0, arrDs[len(arrDs) - 1][0], 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # print("before", image.shape)
            crop_img = image[startY:endY, startX:endX]
            # print("crop_img", crop_img.shape, startY, endY, startX, endX)
            crop_img = cv2.resize(crop_img, (224, 224))

            cv2.imshow('output',crop_img)
    
            x_to_predict = np.array([cv2.resize(image, dsize=(224, 224))]).astype(np.float32)
            x_to_predict = x_to_predict / 255.0
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