import cv2
import numpy as np


def comp_(a1):
    return a1[1]


rootDir = "."

leftCap = cv2.VideoCapture("D:/Updated/GP/raw_video_data/left4.mp4")
rightCap = cv2.VideoCapture("D:/Updated/GP/raw_video_data/right4.mp4")
falseCap = cv2.VideoCapture("D:/Updated/GP/raw_video_data/false4.mp4")

net = cv2.dnn.readNetFromCaffe(
    "./deploy.prototxt", "./res10_300x300_ssd_iter_140000.caffemodel"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml')

newX = []
newY = []

success, image = leftCap.read()
lbl = [1, 0, 0]
while success:
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()
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
        newX.append(crop_img)
        newY.append(lbl)
    success, image = leftCap.read()

success, image = rightCap.read()
lbl = [0, 1, 0]
while success:
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()
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
        newX.append(crop_img)
        newY.append(lbl)
    success, image = rightCap.read()


success, image = falseCap.read()
lbl = [0, 0, 1]
while success:
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()
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
        newX.append(crop_img)
        newY.append(lbl)
    success, image = falseCap.read()


npArr = np.array(newX)
npArr = npArr.astype(np.float32)
npArr /= 255

np.save("./x_test_cropped_float_normalized", npArr)
np.save("./y_test_cropped", np.array(newY))