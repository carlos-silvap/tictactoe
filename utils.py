import cv2
import numpy as np


#### 1 - Preprocessing Image
def preProcess(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # CONVERT IMAGE TO GRAY SCALE
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # ADD GAUSSIAN BLUR
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2)  # APPLY ADAPTIVE THRESHOLD
    return imgThreshold

def preProcessHSV(img):
    #greenLower = (40, 0, 0)
    #greenUpper = (255, 255, 255)
    kernel = np.ones((5,5), np.uint8)
    #greenLower = (25, 0, 175)
    #greenUpper = (70, 255, 255)
    greenLower = (50, 0, 0)
    greenUpper = (255, 255, 255)
    blurred = cv2.GaussianBlur(img, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    imgThreshold = cv2.inRange(hsv, greenLower, greenUpper)
    imgThreshold = cv2.morphologyEx(imgThreshold, cv2.MORPH_CLOSE, kernel)
    return imgThreshold


#### 2 - Reorder points for Warp Perspective
def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] =myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] =myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew


#### 3 - FINDING THE BIGGEST COUNTOUR ASSUING THAT IS THE SUDUKO PUZZLE
def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest,max_area


#### 4 - TO SPLIT THE IMAGE INTO 9 DIFFRENT IMAGES
def splitBoxes(img):
    rows = np.vsplit(img,3)
    boxes=[]
    for r in rows:
        cols= np.hsplit(r,3)
        for box in cols:
            boxes.append(box)
    return boxes


#### 6 - TO STACK ALL THE IMAGES IN ONE WINDOW
def stackImages(imgArray,scale):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    return ver


#### 4 - GET PREDECTIONS ON ALL IMAGES
def getPredection(boxes,model):
    result = []
    for image in boxes:
        ## PREPARE IMAGE
        img = np.asarray(image)
        img = img[4:img.shape[0] - 4, 4:img.shape[1] -4]
        img = cv2.resize(img, (28, 28))
        img = img / 255
        img = img.reshape(1, 28, 28, 1)
        ## GET PREDICTION
        predictions = model.predict(img)
        classIndex = model.predict_classes(img)
        probabilityValue = np.amax(predictions)
        ## SAVE TO RESULT
        if probabilityValue > 0.8:
            result.append(classIndex[0])
        else:
            result.append(0)
    return result

def findObject(object:str, hsv, frame):
    if (object == "cube"):
        lower_range = np.array([0, 120, 190])
        upper_range = np.array([60, 255, 255])
        color = (0,255,0)
    else:
        lower_range = np.array([[55, 40, 160]])
        upper_range = np.array([85, 255, 255])
        color = (255,0,0)
    mask = cv2.inRange(hsv, lower_range, upper_range)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    cent , _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(frame,cent,-1,color,3)

    for (i,c) in enumerate(cent):
        M= cv2.moments(c)
        cx= int(M['m10']/M['m00'])
        cy= int(M['m01']/M['m00'])
        x,y,w,h= cv2.boundingRect(c)
        cv2.putText(frame, text= object, org=(cx-10,y+30),
                fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,0,0),
                thickness=1, lineType=cv2.LINE_AA)

    count = (len(cent))
    print(object+":",count)  