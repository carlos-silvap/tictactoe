"""Test Reachy's cameras, either by subscribing to ROS topics or by using OpenCV.

If you're usig ROS, make sure that the camera_publisher.launch.py has been launched so that
the topics /left_image and /right_image are actually published.

On the contrary, if you're using OpenCV, camera_publisher.launch.py must NOT be launched,
or you won't get acces to the cameras.
"""

import time
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg._compressed_image import CompressedImage
import cv2 as cv


class RosCameraSubscriber(
                Node,
                ):
    """ROS node subscribing to the image topics."""

    def __init__(self, node_name: str, side: str) -> None:
        """Set up the node.

        Subscribe to the requested image topic (either /left_image or /right_image).
        """
        super().__init__(node_name=node_name)

        self.camera_sub = self.create_subscription(
            CompressedImage,
            side+'_image',
            self.on_image_update,
            1,
        )

        self.cam_img = None

    def on_image_update(self, msg):
        """Get data from image. Callback for "/'side'_image "subscriber."""
        data = np.frombuffer(msg.data.tobytes(), dtype=np.uint8)
        self.cam_img = cv.imdecode(data, cv.IMREAD_COLOR)

    def update_image(self):
        """Get the last image by spinning the node."""
        rclpy.spin_once(self)


def main():
    def callback(x):
        pass
    
    #create trackbar window
    cv.namedWindow('image')
    
    # initial limits
    ilowH = 0
    ihighH = 255

    ilowS = 0
    ihighS = 255

    ilowV = 0
    ihighV = 255

    # create trackbars for color change
    cv.createTrackbar('lowH','image',ilowH,255,callback)
    cv.createTrackbar('highH','image',ihighH,255,callback)

    cv.createTrackbar('lowS','image',ilowS,255,callback)
    cv.createTrackbar('highS','image',ihighS,255,callback)

    cv.createTrackbar('lowV','image',ilowV,255,callback)
    cv.createTrackbar('highV','image',ihighV,255,callback)
    
    """Instanciate the correct CameraViewer object for the requested side."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('side')

    args = parser.parse_args()

    requested_side = args.side

    if requested_side not in ['left', 'right']:
        raise ValueError("side argument must either be 'left' or 'right'")

    rclpy.init()
    time.sleep(1)
    image_getter = RosCameraSubscriber(node_name='image_viewer', side=requested_side)


    while True:
        frame = image_getter.update_image()
        
        #Display robot's view
        cv.imshow(args.side + ' camera', image_getter.cam_img)
        
        #Get trackbar positions
        ilowH = cv.getTrackbarPos('lowH', 'image')
        ihighH = cv.getTrackbarPos('highH', 'image')
        ilowS = cv.getTrackbarPos('lowS', 'image')
        ihighS = cv.getTrackbarPos('highS', 'image')
        ilowV = cv.getTrackbarPos('lowV', 'image')
        ihighV = cv.getTrackbarPos('highV', 'image')
        #Read frame
        hsv = cv.cvtColor(image_getter.cam_img, cv.COLOR_BGR2HSV)
        cv.imshow('hsv', hsv)
        lower_hsv = np.array([ilowH, ilowS, ilowV])
        higher_hsv = np.array([ihighH, ihighS, ihighV])
        mask = cv.inRange(hsv, lower_hsv, higher_hsv)
        cv.imshow('mask', mask)
        print (ilowH, ilowS, ilowV)
        print (ihighH, ihighS, ihighV)
        
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()