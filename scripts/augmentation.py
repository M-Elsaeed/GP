import silence_tensorflow  # pylint: disable=unused-import
import matplotlib.pyplot as plt
from cv2 import cv2
import numpy as np
import tensorflow as tf

# import tensorflow_datasets as tfds

from tensorflow.keras import layers

data_augmentation = tf.keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomRotation(0.2, "nearest"),
    ]
)

# imagesLeft = np.load("./left.npy")
# imagesRight = np.load("./right.npy")
imagesFalse = np.load("./false.npy")
# # print (imagesLeft.shape)
# # test  = np.load("./leftAugmented.npy")

# imaug = data_augmentation(imagesLeft)
# np.save("./leftAugmented", imaug)
# imagesLeft = None
# imaug = None

# np.save("./rightAugmented", data_augmentation(imagesRight))
# imagesRight = None

# np.save("./falseAugmentedTill2000", data_augmentation(imagesFalse[0:2000]))
# np.save("./falseAugmentedFrom2000", data_augmentation(imagesFalse[2000:]))

# imaug = np.load("./leftAugmented.npy")
# cv2.namedWindow("output", cv2.WINDOW_NORMAL)
# print(imaug.shape)
# for img in imaug:
#     if img is not None:
#         cv2.imshow("output", img)
#         cv2.waitKey(0)
