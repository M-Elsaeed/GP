import os
import gc
import platform
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet import ResNet50
from makeDataset import makeTrainingDataAndLabels

gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

rootDir = "."

# Shuffles the passed arrays while maintaining correspondance
# uses numpy permutation generator
def shuffle(a, b):
    p = np.random.permutation(len(a))
    return a[p], b[p]


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



try:
    x_train = np.load(f"{rootDir}/npys/x_train.npy")
    y_train = np.load(f"{rootDir}/npys/y_train.npy")
except:
    x_train, y_train = makeTrainingDataAndLabels(
        saveToDrive=True,
        shuffleArray=True,
        normalizeArray=False,
        rotateImages=False,
        resizeImages=False,
        cropFaces=False,
    )

# i = 0
# while i < len(x_train):
#     lbl = "ERROR"
#     if y_train[i][0]:
#         lbl = "L"
#     elif y_train[i][1]:
#         lbl = "R"
#     elif y_train[i][2]:
#         lbl = "F"

#     cv2.putText(x_train[i], lbl, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
#     cv2.imshow("out", x_train[i])
#     cv2.waitKey(1000)
#     face_detector((x_train[i] * 255).astype(np.uint8))
#     # cv2.destroyWindow("out")
#     i += 200

# for i in range(len(x_train)):
#     lbl = "ERROR"
#     if y_train[i][0]:
#          lbl = "L"
#     elif y_train[i][1]:
#          lbl = "R"
#     elif y_train[i][2]:
#          lbl = "F"

#     cv2.putText(x_train[i],lbl,(10,100), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2)
#     cv2.imshow("out", x_train[i])
#     cv2.waitKey(1000)
#     # cv2.destroyWindow("out")
#     i+=500


x_test = np.load(f"{rootDir}/npys/x_test.npy")
y_test = np.load(f"{rootDir}/npys/y_test.npy")

print(x_train[0])

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


model = keras.models.Sequential()

inputTensor = keras.layers.Input(shape=(224, 224, 3))


coreModel = keras.applications.resnet50.ResNet50(
    include_top=False, input_tensor=inputTensor
)
# for i in range(0, len(coreModel.layers) - 100):
#     coreModel.layers[i].trainable = False
model.add(coreModel)

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(3, activation="softmax"))


# 0.0001
adamOptimizer = keras.optimizers.Adam(learning_rate=0.0001)

callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=3, mode="max"
)


model.compile(
    loss="categorical_crossentropy", optimizer=adamOptimizer, metrics=["accuracy"]
)

model.summary()

model = keras.models.load_model(f"{rootDir}/trainedModels/blkshirts.h5")

history = model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=20,
    validation_data=(x_test, y_test),
    callbacks=[callback],
)


model.save(f"{rootDir}/trainedModels/blkshirts3.h5", save_format="h5")

print(history)