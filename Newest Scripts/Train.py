import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from environment import rootDir

gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


x_train = np.load(f"{rootDir}npys/x_train_cropped_noeyes.npy")
y_train = np.load(f"{rootDir}npys/y_train_cropped_noeyes.npy")

x_test = np.load(f"{rootDir}npys/x_test_cropped_noeyes.npy")
y_test = np.load(f"{rootDir}npys/y_test_cropped_noeyes.npy")

print(x_train[0])

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


model = keras.models.Sequential()

inputTensor = keras.layers.Input(shape=(224, 224, 3))

coreModel = keras.applications.resnet50.ResNet50(
    include_top=False, input_tensor=inputTensor
)
model.add(coreModel)
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(3, activation="softmax"))


# 0.0001
adamOptimizer = keras.optimizers.Adam(learning_rate=0.0001)

callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=4, mode="max"
)


model.compile(
    loss="categorical_crossentropy", optimizer=adamOptimizer, metrics=["accuracy"]
)

model.summary()

# model = keras.models.load_model(f"{rootDir}/trainedModels/blkshirts.h5")

history = model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=20,
    validation_data=(x_test, y_test),
    callbacks=[callback],
)


model.save(f"{rootDir}/trainedModels/croppedModel.h5", save_format="h5")

print(history)