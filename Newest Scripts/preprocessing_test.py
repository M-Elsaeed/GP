import cv2
import numpy as np
from environment import rootDir
from environment import shuffle
from tensorflow.keras.utils import to_categorical
import os

def makeDataSetFromVideos(
    saveToDrive=False,
    shuffleArray=False,
    normalizeArray=False,
    rotateImages=False,
    resizeImages=False,
):
    print("makingArrays")
    desc = f"{'shuffled_' if shuffleArray else ''}{'normalized_' if normalizeArray else ''}{'rotated_' if rotateImages else ''}{'resized_' if resizeImages else ''}"
    print(desc)

    leftCaps = []
    rightCaps = []
    falseCaps = []
    print(os.listdir(f"D:/Updated/GP/raw_video_data/"))
    for i in os.listdir(f"D:/Updated/GP/raw_video_data/"):
        if "left" in i and ("MOV" in i or "mp4" in i):
            leftCaps.append(cv2.VideoCapture(f"D:/Updated/GP/raw_video_data/{i}"))
        elif "right" in i and ("MOV" in i or "mp4" in i):
            rightCaps.append(cv2.VideoCapture(f"D:/Updated/GP/raw_video_data/{i}"))
        elif "false" in i and ("MOV" in i or "mp4" in i):
            falseCaps.append(cv2.VideoCapture(f"D:/Updated/GP/raw_video_data/{i}"))

    left = []
    right = []
    false = []
    for leftCap in leftCaps:
        success, image = leftCap.read()
        count = 0
        while success:
            if count == 40:
                if resizeImages:
                    image = cv2.resize(image, (224, 224))
                if rotateImages:
                    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                left.append(image)
                count = 0
            count += 1
            success, image = leftCap.read()
    for rightCap in rightCaps:
        success, image = rightCap.read()
        count = 0
        while success:
            if count == 40:
                if resizeImages:
                    image = cv2.resize(image, (224, 224))
                if rotateImages:
                    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                right.append(image)
                count = 0
            count += 1
            success, image = rightCap.read()
    for falseCap in falseCaps:
        success, image = falseCap.read()
        count = 0
        while success:
            if count == 40:
                if resizeImages:
                    image = cv2.resize(image, (224, 224))
                if rotateImages:
                    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                false.append(image)
                count = 0
            count += 1
            false.append(image)
            success, image = falseCap.read()

    if normalizeArray:
        left = np.array(left)
        right = np.array(right)
        false = np.array(false)
        left = left.astype(np.float32)
        right = right.astype(np.float32)
        false = false.astype(np.float32)
        left /= 255.0
        right /= 255.0
        false /= 255.0

    x_test = np.concatenate((left, right, false))

    leftLabels = [0] * left.shape[0]
    rightLabels = [1] * right.shape[0]
    falseLabels = [2] * false.shape[0]

    y_test = np.concatenate((leftLabels, rightLabels, falseLabels))
    y_test = np.reshape(y_test, (y_test.shape[0], 1))
    y_test = to_categorical(y_test, num_classes=3)

    if shuffleArray:
        x_test, y_test = shuffle(x_test, y_test)

    if saveToDrive:
        np.save(f"{rootDir}/npys/x_test{desc}", x_test)
        np.save(f"{rootDir}/npys/y_test{desc}", y_test)

    return x_test, y_test


def cropDatasetNoEyes(x_test, y_test):

    net = cv2.dnn.readNetFromCaffe(
        "./faceDetectionModel/deploy.prototxt",
        "./faceDetectionModel/res10_300x300_ssd_iter_140000.caffemodel",
    )

    newX = []
    newY = []
    scales = []
    k = 0

    r = x_test[..., 0].mean() * 255
    g = x_test[..., 1].mean() * 255
    b = x_test[..., 2].mean() * 255
    print(r, g, b)

    while k < len(x_test):
        image = (x_test[k] * 255).astype(np.uint8)
        lbl = y_test[k]
        (h, w) = image.shape[:2]

        blob = cv2.dnn.blobFromImage(
            # cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
            cv2.resize(image, (300, 300)),
            1.0,
            (300, 300),
            (r, g, b),
        )
        net.setInput(blob)
        detections = net.forward()
        # loop over the detections
        i = 0
        while i < detections.shape[2]:
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            # print(confidence)
            if confidence > 0.95:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                if startY <= 90:
                    crop_img = image[startY:endY, startX:endX]
                    try:
                        crop_img = cv2.resize(crop_img, (224, 224))
                        newX.append(crop_img)
                        newY.append(lbl)
                        scales.append((confidence, startY, endY, startX, endX, abs(endY - startY) / abs(endX - startX)))
                    except:
                        print(startY, endY, startX, endX)
            i += 1
        k += 1

    del x_test
    del y_test

    npArr = np.array(newX)
    npArr = npArr.astype(np.float32)
    npArr /= 255

    np.save("./npys/x_test_cropped_noeyes", npArr)
    np.save("./npys/y_test_cropped_noeyes", np.array(newY))
    np.save("./npys/scales_cropped_noeyes", np.array(scales))

makeDataSetFromVideos(True, False, True, True, True)
cropDatasetNoEyes(
    np.load("./npys/x_test_normalized_rotated_resized.npy"),
    np.load("./npys/y_test_normalized_rotated_resized.npy"),
)
