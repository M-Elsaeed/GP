import cv2
import numpy as np
from environment import rootDir
from environment import shuffle
from tensorflow.keras.utils import to_categorical
import os


def cropImages(ims):
    for i in ims:
        pass

def makeTrainingDataAndLabels(
    saveToDrive=False,
    shuffleArray=False,
    normalizeArray=False,
    rotateImages=False,
    resizeImages=False,
    cropFaces=False,
):
    print("makingArrays")
    desc = f"{'shuffled_' if shuffleArray else ''}{'normalized_' if normalizeArray else ''}{'rotated_' if rotateImages else ''}{'resized_' if resizeImages else ''}{'cropped' if cropFaces else ''}"
    print(desc)

    left = np.load(f"{rootDir}/npys/left.npy")
    right = np.load(f"{rootDir}/npys/right.npy")
    false = np.load(f"{rootDir}/npys/false.npy")

    if rotateImages:
        for i in range(len(left)):
            left[i] = cv2.rotate(left[i], cv2.ROTATE_90_CLOCKWISE)
        for i in range(len(right)):
            right[i] = cv2.rotate(right[i], cv2.ROTATE_90_CLOCKWISE)
        for i in range(len(false)):
            false[i] = cv2.rotate(false[i], cv2.ROTATE_90_CLOCKWISE)

    if resizeImages:
        for i in range(len(left)):
            left[i] = cv2.resize(left[i], (224, 224))
        for i in range(len(right)):
            right[i] = cv2.resize(right[i], (224, 224))
        for i in range(len(false)):
            false[i] = cv2.resize(false[i], (224, 224))

    # if cropFaces:
    #     newLeft = []
    #     newRight = []
    #     newFalse = []

    #     for i in left:
    #         fc = face_detector(i)
    #         if fc is not None:
    #             newLeft.append(fc)
    #     left = newLeft

    #     for i in right:
    #         fc = face_detector(i)
    #         if fc is not None:
    #             newRight.append(fc)
    #     right = newRight

    #     for i in false:
    #         fc = face_detector(i)
    #         if fc is not None:
    #             newFalse.append(fc)
    #     false = newFalse

    #     left = np.array(left)
    #     right = np.array(right)
    #     false = np.array(false)

    print(left.shape, right.shape, false.shape)

    if normalizeArray:
        left = left.astype(np.float32)
        right = right.astype(np.float32)
        false = false.astype(np.float32)
        left /= 255.0
        right /= 255.0
        false /= 255.0

    x_train = np.concatenate((left, right, false))

    leftLabels = [0] * left.shape[0]
    rightLabels = [1] * right.shape[0]
    falseLabels = [2] * false.shape[0]

    y_train = np.concatenate((leftLabels, rightLabels, falseLabels))
    y_train = np.reshape(y_train, (y_train.shape[0], 1))
    y_train = to_categorical(y_train, num_classes=3)

    if shuffleArray:
        x_train, y_train = shuffle(x_train, y_train)

    if saveToDrive:
        np.save(f"{rootDir}/npys/x_train{desc}", x_train)
        np.save(f"{rootDir}/npys/y_train{desc}", y_train)

    return x_train, y_train


def makeDataSetFromVideos(
    saveToDrive=False,
    shuffleArray=False,
    normalizeArray=False,
    rotateImages=False,
    resizeImages=False
):
    print("makingArrays")
    desc = f"{'shuffled_' if shuffleArray else ''}{'normalized_' if normalizeArray else ''}{'rotated_' if rotateImages else ''}{'resized_' if resizeImages else ''}"
    print(desc)

    leftCaps  = []
    rightCaps  = []
    falseCaps  = []
    print(os.listdir(f'./rawData/'))
    for i in os.listdir(f'./rawData/'):
        if "left" in i and ("MOV" in i or "mp4" in i):
            leftCaps.append(cv2.VideoCapture(f"./rawData/{i}"))
        elif "right" in i and ("MOV" in i or "mp4" in i):
            rightCaps.append(cv2.VideoCapture(f"./rawData/{i}"))
        elif "false" in i and ("MOV" in i or "mp4" in i):
            falseCaps.append(cv2.VideoCapture(f"./rawData/{i}"))
    
    left = []
    right = []
    false = []
    for leftCap in leftCaps:
        success, image = leftCap.read()
        while success:
            if resizeImages:
                image = cv2.resize(image, (224, 224))
            if rotateImages:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            left.append(image)
            success, image = leftCap.read()
    for rightCap in rightCaps:
        success, image = rightCap.read()
        while success:
            if resizeImages:
                image = cv2.resize(image, (224, 224))
            if rotateImages:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            right.append(image)
            success, image = rightCap.read()
    for falseCap in falseCaps:
        success, image = falseCap.read()
        while success:
            if resizeImages:
                image = cv2.resize(image, (224, 224))
            if rotateImages:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
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

    x_train = np.concatenate((left, right, false))

    leftLabels = [0] * left.shape[0]
    rightLabels = [1] * right.shape[0]
    falseLabels = [2] * false.shape[0]

    y_train = np.concatenate((leftLabels, rightLabels, falseLabels))
    y_train = np.reshape(y_train, (y_train.shape[0], 1))
    y_train = to_categorical(y_train, num_classes=3)

    if shuffleArray:
        x_train, y_train = shuffle(x_train, y_train)

    if saveToDrive:
        np.save(f"{rootDir}/npys/x_train{desc}", x_train)
        np.save(f"{rootDir}/npys/y_train{desc}", y_train)

    return x_train, y_train

makeDataSetFromVideos(True, False, True, True, True)