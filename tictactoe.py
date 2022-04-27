print('Setting UP')
import os
from utils import *
import imutils

########################################################################
pathImage = "image.jpg"
pickle_in = open("model_trained_new.p","rb")
model = pickle.load(pickle_in)
#model = intializePredectionModel()  # LOAD THE CNN MODEL
heightImg = 450
widthImg = 450
########################################################################


rclpy.init()
time.sleep(2)
image_getter = RosCameraSubscriber(node_name='image_viewer', side = "right")

while True:
    image_getter.update_image()
    frame = image_getter.cam_img
 


     #### 1. PREPARE THE IMAGE
    img = frame
    img = cv2.resize(img, (widthImg, heightImg))                                                        # RESIZE IMAGE TO MAKE IT A SQUARE IMAGE
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)                                             # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED
    imgThreshold = preProcessHSV(img)


    #### 2. FIND ALL COUNTOURS
    imgContours = img.copy()                                                                            # COPY IMAGE FOR DISPLAY PURPOSES
    imgBigContour = img.copy()                                                                          # COPY IMAGE FOR DISPLAY PURPOSES
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    # FIND ALL CONTOURS
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3)                                         # DRAW ALL DETECTED CONTOURS

    ### 3. FIND THE BIGGEST COUNTOUR AND USE IT AS SUDOKU
    biggest, maxArea = biggestContour(contours)                                                         # FIND THE BIGGEST CONTOUR

    if biggest.size != 0:
        biggest = reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 0, 255), 25)                                   # DRAW THE BIGGEST CONTOUR
        pts1 = np.float32(biggest)                                                                      # PREPARE POINTS FOR WARP
        pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]])                 # PREPARE POINTS FOR WARP
        matrix = cv2.getPerspectiveTransform(pts1, pts2)                                                # GER
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
        imgDetectedDigits = imgBlank.copy()

        #### 4. SPLIT THE IMAGE AND FIND EACH DIGIT AVAILABLE
        imgSolvedDigits = imgBlank.copy()
        boxes = splitBoxes(imgWarpColored)
        j=349
        for box in boxes:
            cv2.imwrite("images/cylinders/"+str(j)+'.jpg', box)
            j=j+1 
        numbers = getPredection(boxes, model)
        numbers = np.asarray(numbers)
        posArray = np.where(numbers > 0, 0, 1)

        #### 5. FIND SOLUTION OF THE BOARD
        board = np.array_split(numbers,3)
        print(board[0])
        print(board[1])
        print(board[2])
        print("      ")
        print("      ")



    imageArray = ([img,imgThreshold,imgContours,imgBigContour],
                    [imgWarpColored, imgBlank,imgBlank,imgBlank])
    stackedImage = stackImages(imageArray, 1)
    cv2.imshow('Stacked Images', stackedImage) 


        
        
        
    #cv2.imshow("Frame", frame)
"""     
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break """



#cv2.imshow('Stacked Images', stackedImage)
#
#else:
#    print("No Sudoku Found")

#cv2.waitKey(0)