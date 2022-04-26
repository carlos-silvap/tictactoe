print('Setting UP')
import os
from utils import *



########################################################################
pathImage = "image.jpg"

heightImg = 450
widthImg = 450
########################################################################


#### 1. PREPARE THE IMAGE
img = cv2.imread(pathImage)
img = cv2.resize(img, (widthImg, heightImg))  # RESIZE IMAGE TO MAKE IT A SQUARE IMAGE
imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)  # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED
imgThreshold = preProcessHSV(img)

#### 2. FIND ALL COUNTOURS
imgContours = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
imgBigContour = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3) # DRAW ALL DETECTED CONTOURS

### 3. FIND THE BIGGEST COUNTOUR AND USE IT AS SUDOKU
biggest, maxArea = biggestContour(contours) # FIND THE BIGGEST CONTOUR
print(biggest)
if biggest.size != 0:
    biggest = reorder(biggest)
    print(biggest)
    cv2.drawContours(imgBigContour, biggest, -1, (0, 0, 255), 25) # DRAW THE BIGGEST CONTOUR
    pts1 = np.float32(biggest) # PREPARE POINTS FOR WARP
    pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
    matrix = cv2.getPerspectiveTransform(pts1, pts2) # GER
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    imgDetectedDigits = imgBlank.copy()
    #imgWarpColored = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)

    #### 4. SPLIT THE IMAGE AND FIND EACH DIGIT AVAILABLE
    imgSolvedDigits = imgBlank.copy()
    boxes = splitBoxes(imgWarpColored)
    print(len(boxes))

    
    #numbers = getPredection(boxes, model)
    #for i in range(9):
    #    hsv = cv2.cvtColor(boxes[i], cv2.COLOR_BGR2HSV)
    #    findObject("cube", hsv, boxes[i])
    #    findObject("cylinder", hsv, boxes[i])

    #j = 161
    #Save images for dataset
    #for i in range(9):
    #    cv2.imwrite('images/'+str(j)+'.jpg', boxes[i])
    #    j=j+1 
       #cv2.imshow("Sample"+str(i),boxes[i])

imageArray = ([img,imgThreshold,imgContours,imgBigContour],
                [imgWarpColored, imgBlank,imgBlank,imgBlank])
stackedImage = stackImages(imageArray, 1)

cv2.imshow('Stacked Images', stackedImage)
#
#else:
#    print("No Sudoku Found")

cv2.waitKey(0)