from cv2 import cv2

imagesPath = input(
    "Please Create a folder called 'frames_greyscale' then\nEnter the ABSOLUTE Path to the folder containing the colored images\n"
)
print()
frameCount = int(input("Please Enter the frame count.\n"))

for count in range(frameCount):
    image = cv2.imread(f"{imagesPath}/frame{count}.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"./frames_greyscale/frame{count}.jpg", gray)