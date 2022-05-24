import cv2
import numpy as np
import pickle

import rclpy
from rclpy.node import Node

from sensor_msgs.msg._compressed_image import CompressedImage


class RosCameraSubscriber(Node, ):
    """ROS node subscribing to the image topics."""

    def __init__(self, node_name: str, side: str) -> None:
        """Set up the node.
        Subscribe to the requested image topic (either /left_image or /right_image).
        """
        super().__init__(node_name=node_name)

        self.camera_sub = self.create_subscription(CompressedImage, side + '_image', 
            self.on_image_update, 1, )

        self.cam_img = None

    def on_image_update(self, msg):
        """Get data from image. Callback for "/'side'_image "subscriber."""
        data = np.frombuffer(msg.data.tobytes(), dtype=np.uint8)
        self.cam_img = cv2.imdecode(data, cv2.IMREAD_COLOR)

    def update_image(self):
        """Get the last image by spinning the node."""
        rclpy.spin_once(self)

def intializePredectionModel():
    """Prepares the trained model

    Returns:
        model: trained model to identify cubes, cylinders or empty
    """    
    pickle_in = open("models/model_v7.p","rb")
    #pickle_in = open("models/model_trained_new_lights.p","rb")
    model = pickle.load(pickle_in)
    return model
    
def preProcessHSV(img:np.array, low, high):
    """Preprocess image to detect the board

    Args:
        img (np.array): numpy array with image

    Returns:
        np.array: array with hsv masked image
    """   
    kernel = np.ones((5,5), np.uint8)
    blurred = cv2.GaussianBlur(img, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    imgThreshold = cv2.inRange(hsv, low, high)
    imgThreshold = cv2.dilate(imgThreshold,kernel,iterations = 1)
    return imgThreshold

def reorder(myPoints:np.array):
    """Reorder points for the Warp Perspective

    Args:
        myPoints (np.array): array with the points

    Returns:
        np.array: array with the new points in order
    """    
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] =myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] =myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

def biggestContour(contours:list):
    """Find the biggest contour, this will be the board

    Args:
        contours (list): list with the contours found

    Returns:
        np.array: biggest contour
        int     : max area
    """    
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

def splitBoxes(img:np.array):
    """Split the image into 9 images

    Args:
        img (np.array): np array with the image to split

    Returns:
        list: list with the 9 boxes of the board
    """    
    rows = np.vsplit(img,3)
    boxes=[]
    for r in rows:
        cols= np.hsplit(r,3)
        for box in cols:
            boxes.append(box)
    return boxes

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

def preProcessing(img:np.array):
    """Preprocess the image to input into the predictor

    Args:
        img (np.array): array with the image to predict

    Returns:
        np.array: image preprossed to be predicted
    """    
    img = np.asarray(img)
    img = cv2.resize(img, (32, 32))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    img = img.reshape(1, 32, 32, 1)
    return img

def getPredection(boxes:list,model):
    """get the prediction of the selected box

    Args:
        boxes (list): list with the 9 boxes to predict
        model (_type_): model that makes the predictions

    Returns:
        list: contains the order of the predicttions in the 9 boxes
    """    
    result = []
    for img in boxes:
        img = preProcessing(img)
        predictions = model.predict(img)
        #print(predictions)
        classIndex  = np.argmax(predictions,axis=1)
        probabilityValue = np.amax(predictions)
        if probabilityValue > 0.85:
            result.append(classIndex[0])
        else:
            result.append(0)
    return result

