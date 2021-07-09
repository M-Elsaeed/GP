import os
from cv2 import cv2
import numpy as np


def makeTrainingDataAndLabels(
    saveToDrive=False, shuffleArray=False, normalizeArray=False
):

    imagesLeft = []
    imagesRight = []
    imagesFalse = []


    leftCaps = []
    rightCaps = []
    falseCaps = []
    
    print(os.listdir(f'./'))
    for i in os.listdir(f'./'):
      if "left" in i and ("MOV" in i or "mp4" in i):
        leftCaps.append(cv2.VideoCapture(f"./{i}"))
      elif "right" in i and ("MOV" in i or "mp4" in i):
        rightCaps.append(cv2.VideoCapture(f"./{i}"))
      elif "false" in i and ("MOV" in i or "mp4" in i):
        falseCaps.append(cv2.VideoCapture(f"./{i}"))

    count = 0
    for i in leftCaps:
        success, image = i.read()
        while success:
            # if count == 5:
            imagesLeft.append(cv2.resize(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), (224, 224)))
            count = 0
            success, image = i.read()
            count+=1
    
    imagesLeft = np.array(imagesLeft)
    np.save("./left", imagesLeft)
    del imagesLeft
    print("done left")
    
    count = 0
    for i in rightCaps:
        success, image = i.read()
        while success:
            # if count == 5:
            imagesRight.append(cv2.resize(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), (224, 224)))
            count = 0
            success, image = i.read()
            count+=1
    imagesRight = np.array(imagesRight)
    np.save("./right", imagesRight)
    del imagesRight
    print("done right")

    count = 0
    for i in falseCaps:
        success, image = i.read()
        while success:
            # if count == 5:
            imagesFalse.append(cv2.resize(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), (224, 224)))
            count = 0
            success, image = i.read()
            count+=1

    imagesFalse = np.array(imagesFalse)
    np.save("./false", imagesFalse)
    print("done false")


    # if normalizeArray:
    #     left /= 255.0
    #     right /= 255.0
    #     false /= 255.0

    # x_train = np.concatenate((left, right, false))

    # leftLabels = [0] * left.shape[0]
    # rightLabels = [1] * right.shape[0]
    # falseLabels = [2] * false.shape[0]

    # y_train = np.concatenate((leftLabels, rightLabels, falseLabels))
    # y_train = np.reshape(y_train, (y_train.shape[0], 1))
    # y_train = to_categorical(y_train, num_classes=3)

    # if shuffleArray:
    #     x_train, y_train = shuffle(x_train, y_train)

    # if saveToDrive:
    #     np.save(f"{rootDir}/npys/x_train", x_train)
    #     np.save(f"{rootDir}/npys/y_train", y_train)

    # return x_train, y_train

makeTrainingDataAndLabels()