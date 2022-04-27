
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
#reachy = ReachySDK('localhost')  



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
    #reachy.turn_on('head')
    
    #Parameters for tracking the defined color
    greenLower = (89, 0, 162)
    greenUpper = (255, 107, 224)
    greenLower = (90, 10, 160)
    greenUpper = (255, 100, 255)
    
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
        cnts = imutils.grab_contours(cnts)
        center = None
        
        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv.contourArea)
            ((x, y), radius) = cv.minEnclosingCircle(c)
            M = cv.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            #print(center)
            # Define the center point of the frame
            center_point = (frame.shape[0]/2, frame.shape[1]/2)
            
            # Calculate the offsets
            x_offset = center_point[0] - center[0]
            y_offset = center_point[1] - center[1]
            
            #print(center_point)
            #print(x_offset, y_offset)
            #print(radius)
            
            
            # only proceed if the radius meets a minimum size
            if radius > 15:
                # draw the circle and centroid on the frame, then update the list of tracked points
                cv.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv.circle(frame, center, 5, (0, 0, 255), -1)
                
            
        #cv.imwrite('image.png', frame)
        
        
        
        # update the points queue
        pts.appendleft(center)

        # loop over the set of tracked points
        for i in range(1, len(pts)):
            # if either of the tracked points are None, ignore them
            if pts[i - 1] is None or pts[i] is None:
                continue
        # show the frame to our screen
        cv.imshow("Frame", frame)
        cv.imshow("Mask", mask)


        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        
        

if __name__ == "__main__":
    main()