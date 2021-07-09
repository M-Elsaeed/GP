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

rootDir = "."

x_train = np.load(f"{rootDir}/npys/x_train_augmented.npy")
y_train = np.load(f"{rootDir}/npys/y_train.npy")

x_test = np.load(f"{rootDir}/npys/x_test.npy")
y_test = np.load(f"{rootDir}/npys/y_test.npy")


# data_augmentation = tf.keras.Sequential(
#     [
#         tf.keras.layers.experimental.preprocessing.RandomTranslation(
#             0.02, 0.2, "nearest"
#         ),
#         layers.experimental.preprocessing.RandomRotation(0.01, "nearest"),
#     ]
# )

# x_train = np.concatenate(
#     (
#         np.array(data_augmentation(x_train[:1000])),
#         np.array(data_augmentation(x_train[1000:2000])),
#         np.array(data_augmentation(x_train[2000:3000])),
#         np.array(data_augmentation(x_train[3000:4000])),
#         np.array(data_augmentation(x_train[4000:5000])),
#         np.array(data_augmentation(x_train[5000:6000])),
#         np.array(data_augmentation(x_train[6000:7000])),
#         np.array(data_augmentation(x_train[7000:])),
#     )
# )


# print(x_train.shape)
# np.save("./x_train_augmented", x_train)
# print(x_train.shape)
# print(x_train[0])

# cv2.namedWindow("output", cv2.WINDOW_NORMAL)
# for img in x_train:
#     if img is not None:
#         cv2.imshow("output", img)
#         cv2.waitKey(0)


print(x_train[0])

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

model = keras.models.load_model(f"{rootDir}/trainedModels/example3.h5")

callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=3, mode="max"
)
history = model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=6,
    validation_data=(x_test, y_test),
    callbacks=[callback],
)


model.save(f"{rootDir}/trainedModels/example3aug.h5", save_format="h5")

print(history)