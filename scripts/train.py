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


# # Can use this in future
# # keras.preprocessing.image_dataset_from_directory()


try:
    imagesLeft = np.load("./left.npy")
    imagesRight = np.load("./right.npy")
    imagesFalse = np.load("./false.npy")
except:
    imagesLeft = []
    imagesRight = []
    imagesFalse = []

    count = 0
    for i in os.listdir("./class_left"):
        if count == 5:
            imagesLeft.append(
                cv2.resize(cv2.imread(f"./class_left/{i}"), dsize=(270, 480))
            )
            count = 0
        count += 1

    count = 0
    for i in os.listdir("./class_right"):
        if count == 5:
            imagesRight.append(
                cv2.resize(cv2.imread(f"./class_right/{i}"), dsize=(270, 480))
            )
            count = 0
        count += 1

    count = 0
    for i in os.listdir("./class_false"):
        if count == 5:
            imagesFalse.append(
                cv2.resize(cv2.imread(f"./class_false/{i}"), dsize=(270, 480))
            )
            count = 0
        count += 1

    imagesLeft = np.array(imagesLeft)
    imagesRight = np.array(imagesRight)
    imagesFalse = np.array(imagesFalse)

    print(imagesLeft.shape, imagesRight.shape, imagesFalse.shape)

    np.save("./left", imagesLeft)
    np.save("./right", imagesRight)
    np.save("./false", imagesFalse)

# imagesLeft = imagesLeft[:100]
# imagesRight = imagesRight[:100]
# imagesFalse = imagesFalse[:100]

x_train = np.concatenate((imagesLeft, imagesRight, imagesFalse))
x_train = x_train / 255

leftLabels = np.array([0] * imagesLeft.shape[0])
rightLabels = np.array([1] * imagesRight.shape[0])
falseLabels = np.array([2] * imagesFalse.shape[0])

y_train = np.concatenate((leftLabels, rightLabels, falseLabels))
y_train = np.reshape(y_train, (y_train.shape[0], 1))
y_train = to_categorical(y_train, num_classes=3)
print(x_train.shape, y_train.shape)

new_input = keras.Input(shape=(480, 270, 3))
coreModel = ResNet50(include_top=False, input_tensor=new_input)
model = keras.models.Sequential()
model.add(coreModel)
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(3, activation="softmax"))

model.summary()
# 0.0001
adamOptimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    loss="categorical_crossentropy", optimizer=adamOptimizer, metrics=["accuracy"]
)

model.fit(x_train, y_train, batch_size=8, epochs=10, validation_split=0.1)

model.save(f"./beforeAugmentation.h5", save_format="h5")


# Training on augmented data

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

# 
model.fit(x_train, y_train, batch_size=8, epochs=10, validation_split=0.1)

model.save(f"./AfterAugmentation.h5", save_format="h5")
