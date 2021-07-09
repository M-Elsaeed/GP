import os
import gc
import platform
import matplotlib.pyplot as plt
from cv2 import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from keras.applications.resnet import ResNet50


gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

rootDir = "."

# Shuffles the passed arrays while maintaining correspondance
# uses numpy permutation generator
def shuffle(a, b):
    p = np.random.permutation(len(a))
    return a[p], b[p]


def makeTrainingDataAndLabels(
    saveToDrive=False, shuffleArray=False, normalizeArray=False
):

    left = np.load(f"{rootDir}/npys/left.npy")
    right = np.load(f"{rootDir}/npys/right.npy")
    false = np.load(f"{rootDir}/npys/false.npy")

    if normalizeArray:
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
        np.save(f"{rootDir}/npys/x_train", x_train)
        np.save(f"{rootDir}/npys/y_train", y_train)

    return x_train, y_train


try:
    x_train = np.load(f"{rootDir}/npys/x_train.npy")
    y_train = np.load(f"{rootDir}/npys/y_train.npy")
except:
    x_train, y_train = makeTrainingDataAndLabels(
        saveToDrive=True, shuffleArray=True, normalizeArray=True
    )

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

history = model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=20,
    validation_data=(x_test, y_test),
    callbacks=[callback],
)


model.save(f"{rootDir}/trainedModels/example3.h5", save_format="h5")

print(history)