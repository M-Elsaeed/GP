import cv2
import numpy as np
import cv2
import silence_tensorflow.auto  # pylint: disable=unused-import
import tensorflow as tf
from tensorflow import keras

gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# model = keras.models.load_model(f"./trainedModels/blkshirts2.h5")
model = keras.models.load_model(f"./trainedModels/croppedModel.h5")



# url = 'http://192.168.1.89:8080/video'
url = 'http://172.20.10.2:8080/video'
url = 'http://192.168.0.104:8080/video'

cap = cv2.VideoCapture(url)
# cap = cv2.VideoCapture("../test/right_test.mp4")

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

cv2.namedWindow("output", cv2.WINDOW_NORMAL)
while(True):
    ret, frame = cap.read()
    # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if frame is not None:
        frame = face_detector(frame)
        if frame is not None:
            cv2.imshow('output',frame)
            x_to_predict = np.array([cv2.resize(frame, dsize=(224, 224))]).astype(np.float32)
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