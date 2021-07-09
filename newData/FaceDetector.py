import numpy as np
import cv2


def face_detector(img_path):

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # load color (BGR) image
    # img = cv2.imread(img_path)
    img = img_path
    # convert BGR image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # find faces in image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # print number of faces detected in the image
    print('Number of faces detected:', len(faces))

    # get bounding box for each detected face
    for (x, y, w, h) in faces:
        # add bounding box to color image
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        crop_img = img[x:x+w, y:y+h]
        print(crop_img.shape)
        crop_img_gray = gray[x:x+w, y:y+h]
        # align_Face(crop_img, crop_img_gray)
        # cv2.waitKey(0)

    # convert BGR image to RGB for plotting
    # cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(cv_rgb, (480, 480))
    if len(faces):
        cv2.imshow('cropped', crop_img)
        cv2.waitKey(1)


def align_Face(img, img_gray):

    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(img_gray, 1.1, 4)
    index = 0
    # Creating for loop in order to divide one eye from another
    for (ex, ey,  ew,  eh) in eyes:
        if index == 0:
            eye_1 = (ex, ey, ew, eh)
        elif index == 1:
            eye_2 = (ex, ey, ew, eh)
    # Drawing rectangles around the eyes
        cv2.rectangle(img, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 3)
        index = index + 1
    # get left and right eye
    if eye_1[0] < eye_2[0]:
        left_eye = eye_1
        right_eye = eye_2

    else:
        left_eye = eye_2
        right_eye = eye_1
    # Calculating coordinates of a central points of the rectangles
    left_eye_center = (
        int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
    left_eye_x = left_eye_center[0]
    left_eye_y = left_eye_center[1]

    right_eye_center = (
        int(right_eye[0] + (right_eye[2]/2)), int(right_eye[1] + (right_eye[3]/2)))
    right_eye_x = right_eye_center[0]
    right_eye_y = right_eye_center[1]

    cv2.circle(img, left_eye_center, 5, (255, 0, 0), -1)
    cv2.circle(img, right_eye_center, 5, (255, 0, 0), -1)
    cv2.line(img, right_eye_center, left_eye_center, (0, 200, 200), 3)

    if left_eye_y > right_eye_y:
        A = (right_eye_x, left_eye_y)
   # Integer -1 indicates that the image will rotate in the clockwise direction
        direction = -1
    else:
        A = (left_eye_x, right_eye_y)
    # Integer 1 indicates that image will rotate in the counter clockwise
    # direction
        direction = 1

    cv2.circle(img, A, 5, (255, 0, 0), -1)

    cv2.line(img, right_eye_center, left_eye_center, (0, 200, 200), 3)
    cv2.line(img, left_eye_center, A, (0, 200, 200), 3)
    cv2.line(img, right_eye_center, A, (0, 200, 200), 3)
    delta_x = right_eye_x - left_eye_x
    delta_y = right_eye_y - left_eye_y
    angle = np.arctan(delta_y/delta_x)
    angle = (angle * 180) / np.pi

    # Width and height of the image
    h, w = img.shape[:2]
    # Calculating a center point of the image
    # Integer division "//"" ensures that we receive whole numbers
    center = (w // 2, h // 2)
    # Defining a matrix M and calling
    # cv2.getRotationMatrix2D method
    M = cv2.getRotationMatrix2D(center, (angle), 1.0)
    # Applying the rotation to our image using the
    # cv2.warpAffine method
    rotated = cv2.warpAffine(img, M, (w, h))
    cv2.imshow("rotated", rotated)
    cv2.waitKey(0)

    dist_1 = np.sqrt((delta_x * delta_x) + (delta_y * delta_y))

    # calculate distance between the eyes in the second image
    # dist_2 = np.sqrt((delta_x_1 * delta_x_1) + (delta_y_1 * delta_y_1))
    # calculate the ratio
    ratio = dist_1 / dist_1
    # Defining the width and height
    h = 476
    w = 488
    # Defining aspect ratio of a resized image
    dim = (int(w * ratio), int(h * ratio))
    # We have obtained a new image that we call resized3
    resized = cv2.resize(rotated, dim)
    cv2.imshow(resized)
    cv2.waitKey(1)

    cv2.imshow(img)
    cv2.waitKey(0)

vc = cv2.VideoCapture(0)
while True:
    sucess, i = vc.read()
    face_detector(i)
