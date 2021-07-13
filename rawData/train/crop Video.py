import cv2
import numpy as np

net = cv2.dnn.readNetFromCaffe(
    "D:/Updated/GP/faceDetectionModel/deploy.prototxt",
    "D:/Updated/GP/faceDetectionModel/res10_300x300_ssd_iter_140000.caffemodel",
)
def cropFaceDNN(image):
    # print(image.dtype, image.shape)
    crop_img = None

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()
    i = 0
    # print("faces", detections.shape[2])
    while i < detections.shape[2]:
        confidence = detections[0, 0, i, 2]
        if confidence > 0.95:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # print(startY, endY, startX, endX)
            if startY <= image.shape[0]/3 and startY >=0 and endY >=0 and startX >=0 and endX >=0 :
                crop_img = image[startY:endY, startX:endX]
                try:
                    crop_img = cv2.resize(crop_img, (224, 224))
                except:
                    print(startY, endY, startX, endX)
        i += 1
    return crop_img

newX = []
vidCap = cv2.VideoCapture("./false0.mp4")

count = 0
success, image = vidCap.read()
height, width, layers = image.shape
size = (width,height)
while success and count <= 300:
    image = cropFaceDNN(image)
    if image is not None:
        newX.append(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))
        count += 1
    success, image = vidCap.read()
print(len(newX))
cv2.imshow("out", newX[0])
cv2.waitKey(0)
print(size)
out = cv2.VideoWriter('cropped.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 30, (224, 224))
 
for i in range(len(newX)):
    out.write(newX[i])
out.release()