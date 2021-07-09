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
    resizeImages=False,
):
    print("makingArrays")
    desc = f"{'shuffled_' if shuffleArray else ''}{'normalized_' if normalizeArray else ''}{'rotated_' if rotateImages else ''}{'resized_' if resizeImages else ''}"
    print(desc)

    leftCaps = []
    rightCaps = []
    falseCaps = []
    print(os.listdir(f"./rawData/"))
    for i in os.listdir(f"./rawData/"):
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


def cropDataset(x_train, y_train):
    def comp_(a1):
        return a1[1]

    net = cv2.dnn.readNetFromCaffe(
        "./faceDetectionModel/deploy.prototxt",
        "./faceDetectionModel/res10_300x300_ssd_iter_140000.caffemodel",
    )
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    newX = []
    newY = []
    k = 0
    # print(x_train.mean(axis=(0,1)))
    # means =calcMean()

    r = x_train[..., 0].mean() * 255
    g = x_train[..., 1].mean() * 255
    b = x_train[..., 2].mean() * 255

    # means = np.mean(x_train*255, axis=(1,0,2))
    print(r, g, b)
    while k < len(x_train):
        image = (x_train[k] * 255).astype(np.uint8)
        lbl = y_train[k]
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
        arrDs = []
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
                # cv2.imshow("out", image[startY:endY, startX:endX])
                # cv2.waitKey(0)
                try:
                    eyes = eye_cascade.detectMultiScale(
                        cv2.cvtColor(
                            image[startY:endY, startX:endX], cv2.COLOR_BGR2GRAY
                        ),
                        1.1,
                        4,
                    )
                    # print(len(eyes))
                    if len(eyes) >= 1:
                        arrDs.append([i, confidence])
                except:
                    print(startY, endY, startX, endX)
            i += 1
        # print(arrDs)
        arrDs.sort(key=comp_)
        # print(arrDs)

        # compute the (x, y)-coordinates of the bounding box for the
        # object
        if len(arrDs):
            box = detections[0, 0, arrDs[len(arrDs) - 1][0], 3:7] * np.array(
                [w, h, w, h]
            )
            (startX, startY, endX, endY) = box.astype("int")

            # print("before", image.shape)
            crop_img = image[startY:endY, startX:endX]
            # print("crop_img", crop_img.shape, startY, endY, startX, endX)
            crop_img = cv2.resize(crop_img, (224, 224))
            newX.append(crop_img)
            newY.append(lbl)

            # # draw the bounding box of the face along with the associated
            # # probability
            # confidence = arrDs[len(arrDs) - 1][1]
            # text = "{:.2f}%".format(confidence * 100)
            # y = startY - 10 if startY - 10 > 10 else startY + 10
            # cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
            # cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

            # print('Count ', len(arrDs))
            # # show the output image
            # cv2.imshow("Output", image)
            # cv2.waitKey(0)
        # cv2.imshow("out", x_train[k])
        # cv2.waitKey(0)
        k += 1

    del x_train
    del y_train

    npArr = np.array(newX)
    npArr = npArr.astype(np.float32)
    npArr /= 255

    np.save("./npys/x_train_cropped", npArr)
    np.save("./npys/y_train_cropped", np.array(newY))


def cropDatasetHaarCascade(x_train, y_train):
    def face_detector(img):
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
        )
        # load color (BGR) image
        # img = cv2.imread(img_path)
        # convert BGR image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # find faces in image
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # print number of faces detected in the image
        # print("Number of faces detected:", len(faces))

        # get bounding box for each detected face
        for (x, y, w, h) in faces:
            # add bounding box to color image
            # cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            crop_img = img[y : y + h, x : x + w]
            crop_img = cv2.resize(crop_img, (224, 224))
            # print(crop_img.shape)
            # crop_img_gray = gray[x:x+w, y:y+h]
            # align_Face(crop_img, crop_img_gray)
            # cv2.waitKey(0)

        # convert BGR image to RGB for plotting
        # cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(cv_rgb, (480, 480))
        if len(faces):
            # cv2.imshow('cropped', crop_img)
            # cv2.waitKey(50)
            # print(crop_img.shape)
            return crop_img
        return None

    newX = []
    newY = []
    k = 0
    while k < len(x_train):
        image = (x_train[k] * 255).astype(np.uint8)
        lbl = y_train[k]
        crop_img = face_detector(image)
        if crop_img is not None:
            newX.append(crop_img)
            newY.append(lbl)
        k += 1

    del x_train
    del y_train

    npArr = np.array(newX)
    npArr = npArr.astype(np.float32)
    npArr /= 255
    print(npArr.shape)
    np.save("./npys/x_train_cropped_haar", npArr)
    np.save("./npys/y_train_cropped_haar", np.array(newY))


def cropDatasetNoEyes(x_train, y_train):

    net = cv2.dnn.readNetFromCaffe(
        "./faceDetectionModel/deploy.prototxt",
        "./faceDetectionModel/res10_300x300_ssd_iter_140000.caffemodel",
    )

    newX = []
    newY = []
    scales = []
    k = 0

    r = x_train[..., 0].mean() * 255
    g = x_train[..., 1].mean() * 255
    b = x_train[..., 2].mean() * 255
    print(r, g, b)

    while k < len(x_train):
        image = (x_train[k] * 255).astype(np.uint8)
        lbl = y_train[k]
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

    del x_train
    del y_train

    npArr = np.array(newX)
    npArr = npArr.astype(np.float32)
    npArr /= 255

    np.save("./npys/x_train_cropped_noeyes", npArr)
    np.save("./npys/y_train_cropped_noeyes", np.array(newY))
    np.save("./npys/scales_cropped_noeyes", np.array(scales))


cropDatasetNoEyes(
    np.load("./npys/x_train_normalized_rotated_resized.npy"),
    np.load("./npys/y_train_normalized_rotated_resized.npy"),
)
