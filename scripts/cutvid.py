from cv2 import cv2
import os

LOWRES_OUTPUT = False
GREYSCALE_OUTPUT = False

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
        
        if GREYSCALE_OUTPUT:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if LOWRES_OUTPUT:
            cv2.imwrite(f"./frames_color/frame{count}.jpg", image, [cv2.IMWRITE_JPEG_QUALITY,50])
        else:
            cv2.imwrite(f"./frames_color/frame{count}.jpg", image)
        
        success, image = i.read()
        count += 1