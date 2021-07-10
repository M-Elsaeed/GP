import cv2
import numpy as np

net = cv2.dnn.readNetFromCaffe(
    "./faceDetectionModel/deploy.prototxt",
    "./faceDetectionModel/res10_300x300_ssd_iter_140000.caffemodel",
)
def cropFaceDNN(image):
    # print(image.dtype, image.shape)
    crop_img = None

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()
    i = 0
    # print("faces", detections.shape[2])
    while i < detections.shape[2]:
        confidence = detections[0, 0, i, 2]
        if confidence > 0.95:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # print(startY, endY, startX, endX)
            if startY <= image.shape[0]/2 and startY >=0 and endY >=0 and startX >=0 and endX >=0 :
                crop_img = image[startY:endY, startX:endX]
                try:
                    crop_img = cv2.resize(crop_img, (224, 224))
                except:
                    print(startY, endY, startX, endX)
        i += 1
    return crop_img

def cropFaceHaarCascade(image):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
    )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        crop_img = image[y : y + h, x : x + w]
        crop_img = cv2.resize(crop_img, (224, 224))
    if len(faces):
        return crop_img
    return None

def printModelDetails(modelPath):
    import keras
    model = keras.models.load_model(modelPath)
    model.summary()
