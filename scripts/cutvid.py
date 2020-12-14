from cv2 import cv2

videoPath = input(
    "Please Create a folder called 'frames_color' then\nEnter the ABSOLUTE Path to the video including its extension\n"
)
vidcap = cv2.VideoCapture(videoPath)

success, image = vidcap.read()

count = 0
while success:
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(f"./frames_color/frame{count}.jpg", image)
    success, image = vidcap.read()
    count += 1