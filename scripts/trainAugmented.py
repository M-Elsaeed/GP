import os
import platform
import numpy as np
from cv2 import cv2
import silence_tensorflow.auto  # pylint: disable=unused-import
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical

# tf_config = tf.ConfigProto(allow_soft_placement=False)
gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


from keras.applications.resnet import ResNet50
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten



imagesLeft = np.load("./leftAugmented.npy")
imagesRight = np.load("./rightAugmented.npy")
imagesFalse = np.load("./falseAugmentedTill2000.npy")
np.concatenate((imagesFalse, np.load("./falseAugmentedFrom2000.npy")))


x_train = np.concatenate((imagesLeft, imagesRight, imagesFalse))
x_train = x_train / 255

leftLabels = np.array([0] * imagesLeft.shape[0])
rightLabels = np.array([1] * imagesRight.shape[0])
falseLabels = np.array([2] * imagesFalse.shape[0])

y_train = np.concatenate((leftLabels, rightLabels, falseLabels))
y_train = np.reshape(y_train, (y_train.shape[0], 1))
y_train = to_categorical(y_train, num_classes=3)
print(x_train.shape, y_train.shape)

model = keras.models.load_model("./beforeAugmentation.h5")

model.fit(x_train, y_train, batch_size=8, epochs=10, validation_split=0.1)

model.save(f"./AfterAugmentation.h5", save_format="h5")
