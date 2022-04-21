import time
import numpy as np
import imutils
import argparse

from collections import deque
from reachy_sdk import ReachySDK

#ROS libraries
import rclpy
from rclpy.node import Node

#Image libraries
from sensor_msgs.msg._compressed_image import CompressedImage
import cv2 as cv

#Define Reachy and get IP
reachy = ReachySDK('localhost')  



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
        self.cam_img = cv.imdecode(data, cv.IMREAD_COLOR)

    def update_image(self):
        """Get the last image by spinning the node."""
        rclpy.spin_once(self)



def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-b", "--buffer", type=int, default=16,
	help="max buffer size")
    arguments = vars(parser.parse_args())
    
    #Initial position of the head
    reachy.turn_on('head')
    
    #Parameters for tracking the defined color
    greenLower = (89, 0, 162)
    greenUpper = (255, 107, 224)
    
    pts = deque(maxlen=arguments["buffer"])
    

    #Initialize ROS
    rclpy.init()
    time.sleep(1)
    image_getter = RosCameraSubscriber(node_name='image_viewer', side = "right")
    
    while True:
        image_getter.update_image()
        
        #cv.imshow(args.side + ' camera', image_getter.cam_img)
        frame = image_getter.cam_img
        # resize the frame, blur it, and convert it to the HSV color space
        frame = imutils.resize(frame, width = 600)
        blurred = cv.GaussianBlur(frame, (11, 11), 0)
        hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
        
        # construct a mask for the color "green", then perform a series of dilations and erosions to remove any small
        # blobs left in the mask
        mask = cv.inRange(hsv, greenLower, greenUpper)
        for i in range(5):
            mask = cv.erode(mask, None, iterations=2)
            mask = cv.dilate(mask, None, iterations=2)        

        # find contours in the mask and initialize the current (x, y) center of the ball
        cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            rect = cv.minAreaRect(c)
            xc = int(rect[0][0])
            yc = int(rect[0][1])
            h = int(rect[1][0])
            w = int(rect[1][1])
            rx = rect[1][0]/2
            ry= rect[1][1]/2
            x = int(xc - rx) 
            y = int(yc - ry)
            if((rx > 150) and (ry > 150)):
                box = cv.boxPoints(rect)
                box = np.int0(box)
                cv.drawContours(frame,[box],0,(0,0,255),2)
                ROI = frame[y:y+h, x:x+w]
        cv.imshow("Frame", frame)
    


        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        
        

if __name__ == "__main__":
    main()