import cv2
import numpy as np


def comp_(a1):
    return a1[1]


rootDir = "."

x_train = np.load(f"{rootDir}/x_train.npy")
y_train = np.load(f"{rootDir}/y_train.npy")
print(x_train.shape, y_train.shape)

net = cv2.dnn.readNetFromCaffe(
    "./deploy.prototxt", "./res10_300x300_ssd_iter_140000.caffemodel"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml')

newX = []
newY = []
k = 0
while k < len(x_train):
    # image = (x_train[k] * 255).astype(np.uint8)
    image = x_train[k]
    lbl = y_train[k]
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

np.save("./x_train_cropped_checkpoint", npArr)
np.save("./y_train_cropped", np.array(newY))
print(npArr.shape, np.array(newY))

npArr = npArr.astype(np.float32)
np.save("./x_train_cropped_float", npArr)

npArr /= 255
np.save("./x_train_cropped_float_normalized", npArr)
