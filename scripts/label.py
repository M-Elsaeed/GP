from cv2 import cv2

# To stop skipping of unlabelled frames
lastX = None
lastY = None

# function to display the coordinates of
# of the points clicked on the image
def click_event(event, x, y, flags, params):

    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        # displaying the coordinates
        # on the Shell
        global lastX
        global lastY
        global count
        lastX = x
        lastY = y
        # print(x, " ", y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, str(x) + "," + str(y), (x, y), font, 1, (255, 0, 0), 2)
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.imshow("frame", image)


if __name__ == "__main__":
    # Take User Input for Starting and Ending frame numbers
    frameNoFrom = int(input("Enter Frame frame number to Label\n"))
    print()
    frameNoTo = int(input("Enter Last frame number to Label\n"))
    print()

    # Take Absolute Path for Input Frames
    inputPath = input(
        "Enter the ABSOLUTE path with forward slashes for input images where they are named frame1.jpg, frame2.jpg ....\n"
    )
    print()

    # For Dev purposes
    inputPath = "../Data/ehab_color"

    # Add column headings if starting from first frame, else, append only
    if frameNoFrom == 0:
        f = open("result.txt", "a")
        f.write(f"frame_no,x,y\n")
        f.close()

    # Loop Over Required Frames, Only proceed to next frame if a coordinate was selected
    count = frameNoFrom

    while count <= frameNoTo:
        image = cv2.imread(f"{inputPath}/frame{count}.jpg")
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.imshow("frame", image)

        cv2.setMouseCallback("frame", click_event)
        cv2.waitKey(0)

        if (lastX != None) and (lastY != None):
            cv2.destroyAllWindows()

            f = open("result.txt", "a")
            f.write(f"{count},{lastX},{lastY}\n")
            f.close()

            lastX = None
            lastY = None

            count += 1


# frame_no, x, y
# 1, 2, 3
# 2, 10, 20