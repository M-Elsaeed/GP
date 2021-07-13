import cv2
import numpy as np


def detectMarkers(image):
    # 7x7, 999:solution, 777: swab. Size is 100mm^2
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_1000)
    arucoParams = cv2.aruco.DetectorParameters_create()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(
        image, arucoDict, parameters=arucoParams
    )
    print(ids)
    return ids

# detectMarkers(cv2.imread("C:/Users/melsa/Desktop/999.jpg"))
# cv2.imshow("out", cv2.imread("C:/Users/melsa/Desktop/999.jpg"))
# cv2.waitKey(0)
# detectMarkers(cv2.imread("C:/Users/melsa/Desktop/777.jpg"))
# cv2.imshow("out", cv2.resize(cv2.imread("C:/Users/melsa/Desktop/777.jpg"), (100,100)))
# cv2.waitKey(0)
# url = 0 #for camera
# cap = cv2.VideoCapture(url)
# while True:
#     success, image = cap.read()
#     if success:
#         detectMarkers(image)
#         cv2.imshow("out", image)
#         cv2.waitKey(1)
#     success, image = cap.read()
