from cv2 import cv2
import os

folderPath = input(
    "Please Create a folder called 'frames_color' then\nEnter the ABSOLUTE Path to the Folder of videos\n"
)

vidNames = os.listdir(folderPath)
print(vidNames)


vidCaps = []
for i in vidNames:
    vidCaps.append(cv2.VideoCapture(folderPath + f"\\{i}"))


count = 0
for i in vidCaps:
    success, image = i.read()
    while success:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(f"./frames_color/frame{count}.jpg", image)
        success, image = i.read()
        count += 1