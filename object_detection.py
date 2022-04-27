import string
import cv2
import numpy as np

def findObject(object:string):
    if (object == "cube"):
        lower_range = np.array([169, 100, 100])
        upper_range = np.array([189, 255, 255])
        color = (0,255,0)
    else:
        lower_range = np.array([[50, 100, 100]])
        upper_range = np.array([90, 255, 255])
        color = (255,0,0)
    mask = cv2.inRange(hsv, lower_range, upper_range)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    cent , _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(resized,cent,-1,color,3)

    for (i,c) in enumerate(cent):
        M= cv2.moments(c)
        cx= int(M['m10']/M['m00'])
        cy= int(M['m01']/M['m00'])
        x,y,w,h= cv2.boundingRect(c)
        cv2.putText(resized, text= object, org=(cx-10,y+30),
                fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0),
                thickness=1, lineType=cv2.LINE_AA)

    count = (len(cent))
    print(object+":",count)    


img = cv2.imread("shapes.jpg")

#Work with the image
scale_percent = 40 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

findObject("cube")
findObject("cylinder")


cv2.imshow("Image", resized)

cv2.waitKey(0)
cv2.destroyAllWindows()