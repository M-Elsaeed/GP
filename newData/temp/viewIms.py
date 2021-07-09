import cv2
import numpy as np

rootDir = "."

x_train = np.load(f"{rootDir}/x_train_cropped_float_normalized.npy")
y_train = np.load(f"{rootDir}/y_train_cropped.npy")

# x_train = np.load(f"{rootDir}/x_test.npy")
# y_train = np.load(f"{rootDir}/y_test.npy")

print(x_train[0])
print(y_train[0])

print(x_train.shape, y_train.shape)

k = 0
while k < len(x_train):
    lbl = "ERROR"
    if y_train[k][0]:
        lbl = "L"
    elif y_train[k][1]:
        lbl = "R"
    elif y_train[k][2]:
        lbl = "F"
    cv2.putText(x_train[k], lbl, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("out", x_train[k])
    cv2.waitKey(0)
    k += 1