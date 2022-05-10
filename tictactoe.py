from utils import *
import time
from rl_agent import *

model       = intializePredectionModel()  
heightImg   = 300
widthImg    = 300


class PlaygroundTicTacToe():
    def __init__(self):
        rclpy.init()
        time.sleep(2)
        self.image_getter = RosCameraSubscriber(node_name='image_viewer', side = "right")

 # Playground and game functions
    def coin_flip(self):
        coin = np.random.rand() > 0.5
        
        print('reachy' if coin else 'human')
        return coin

    def live_view(self):
        while True:
            self.image_getter.update_image()
            img = self.image_getter.cam_img
        
            #### 1. PREPARE THE IMAGE
            img = cv2.resize(img, (widthImg, heightImg))                                                        # RESIZE IMAGE TO MAKE IT A SQUARE IMAGE
            imgThreshold = preProcessHSV(img, lowS, lowV)


            #### 2. FIND ALL COUNTOURS
            imgContours = img.copy()                                                                            # COPY IMAGE FOR DISPLAY PURPOSES
            imgBigContour = img.copy()                                                                          # COPY IMAGE FOR DISPLAY PURPOSES
            contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    # FIND ALL CONTOURS
            cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3)                                         # DRAW ALL DETECTED CONTOURS

            ### 3. FIND THE BIGGEST COUNTOUR AND USE IT AS BOARD
            biggest, maxArea = biggestContour(contours)                                                         # FIND THE BIGGEST CONTOUR

            if biggest.size != 0:
                biggest = reorder(biggest)
                cv2.drawContours(imgBigContour, biggest, -1, (0, 0, 255), 25)                                   # DRAW THE BIGGEST CONTOUR
                pts1 = np.float32(biggest)                                                                      # PREPARE POINTS FOR WARP
                pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]])                 # PREPARE POINTS FOR WARP
                matrix = cv2.getPerspectiveTransform(pts1, pts2)                                                # GER
                imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
                

                #### 4. SPLIT THE IMAGE AND FIND EACH DIGIT AVAILABLE
                boxes = splitBoxes(imgWarpColored)
                numbers = getPredection(boxes, model)
                numbers = np.asarray(numbers)
                #posArray = np.where(numbers > 0, 0, 1)
                
                imageArray = ([img,imgThreshold,imgContours],
                                [imgBigContour, imgWarpColored, img])
                stackedImage = stackImages(imageArray, 1)
                cv2.imshow('Stacked Images', stackedImage)
                
                #### 5. FIND SOLUTION OF THE BOARD
                board = np.array_split(numbers,3)
                print(board[0])
                print(board[1])
                print(board[2])
                print("      ")
                print("      ")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def get_board(self, lowS, lowV):
        self.image_getter.update_image()
        img = self.image_getter.cam_img
        img = cv2.resize(img, (widthImg, heightImg))                                                        # RESIZE IMAGE TO MAKE IT A SQUARE IMAGE
        imgThreshold = preProcessHSV(img, lowS, lowV)
        contours, _ = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)            # FIND ALL CONTOURS
        biggest, _ = biggestContour(contours)                                                               # FIND THE BIGGEST CONTOUR
        if biggest.size != 0:
            biggest = reorder(biggest)
            pts1 = np.float32(biggest)                                                                      # PREPARE POINTS FOR WARP
            pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]])                 # PREPARE POINTS FOR WARP
            matrix = cv2.getPerspectiveTransform(pts1, pts2)                                                # GER
            imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
            boxes = splitBoxes(imgWarpColored)
            numbers = getPredection(boxes, model)
            numbers = np.asarray(numbers)
            board = np.array_split(numbers,3)
            return numbers

    def choose_next_action(self, board):
        actions = value_actions(board)
        if np.all(board == 0):
            while True:
                i = np.random.randint(0, 8)
                a, _ = actions[i]
                print('empty')
                if a != 8:
                    break
        elif np.sum(board) == 1:
            a, _ = actions[0]
            if a == 8:
                i = 1
            else:
                i = 0
        else:
            i = 0

        best_action, value = actions[i]


        return best_action, value, actions

    def get_winner(self, board):
        win_configurations = (
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),

            (0, 3, 6),
            (1, 4, 7),
            (2, 5, 8),

            (0, 4, 8),
            (2, 4, 6),
        )

        for c in win_configurations:
            trio = set(board[i] for i in c)
            for id in range(3):
                if trio == set([id]):
                    winner = id
                    if winner in (1, 2):
                        return winner

        return 'nobody'

    def get_images(self, lowS, lowV):
        self.image_getter.update_image()
        img = self.image_getter.cam_img
        img = cv2.resize(img, (widthImg, heightImg))                                                        # RESIZE IMAGE TO MAKE IT A SQUARE IMAGE
        imgThreshold = preProcessHSV(img, lowS, lowV)
        contours, _ = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)            # FIND ALL CONTOURS
        biggest, _ = biggestContour(contours)                                                               # FIND THE BIGGEST CONTOUR
        if biggest.size != 0:
            biggest = reorder(biggest)
            pts1 = np.float32(biggest)                                                                      # PREPARE POINTS FOR WARP
            pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]])                 # PREPARE POINTS FOR WARP
            matrix = cv2.getPerspectiveTransform(pts1, pts2)                                                # GER
            imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
            boxes = splitBoxes(imgWarpColored)
        return boxes

    def calibrate_HSV(self):
        def callback(x):
            pass
        #create trackbar window
        cv2.namedWindow('image')
        
        # initial limits
        ilowH = 0
        ihighH = 255

        ilowS = 0
        ihighS = 255

        ilowV = 0
        ihighV = 255

        # create trackbars for color change
        cv2.createTrackbar('lowH','image',ilowH,255,callback)
        cv2.createTrackbar('highH','image',ihighH,255,callback)

        cv2.createTrackbar('lowS','image',ilowS,255,callback)
        cv2.createTrackbar('highS','image',ihighS,255,callback)

        cv2.createTrackbar('lowV','image',ilowV,255,callback)
        cv2.createTrackbar('highV','image',ihighV,255,callback)

        time.sleep(1)

        while True:
            self.image_getter.update_image()
            frame = self.image_getter.cam_img
            #Display robot's view
            cv2.imshow('View', frame)
            
            #Get trackbar positions
            ilowH = cv2.getTrackbarPos('lowH', 'image')
            ihighH = cv2.getTrackbarPos('highH', 'image')
            ilowS = cv2.getTrackbarPos('lowS', 'image')
            ihighS = cv2.getTrackbarPos('highS', 'image')
            ilowV = cv2.getTrackbarPos('lowV', 'image')
            ihighV = cv2.getTrackbarPos('highV', 'image')
            #Read frame
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            cv2.imshow('hsv', hsv)
            lower_hsv = np.array([ilowH, ilowS, ilowV])
            higher_hsv = np.array([ihighH, ihighS, ihighV])
            mask = cv2.inRange(hsv, lower_hsv, higher_hsv)
            cv2.imshow('mask', mask)
            #print (ilowH, ilowS, ilowV)
            #print (ihighH, ihighS, ihighV)
            
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            file1 = open("hsv.txt","w")
            L = [str(ilowS)+' '+str(ilowV)] 
            file1.writelines(L)
            file1.close() 
        return ilowS,ilowV

    def get_HSV(self):
        given_file = open('hsv.txt', 'r')

        lines = given_file.readlines()
        vals = []
        for line in lines:
            for word in line.split():
                    vals.append(int(word))
        ls = vals[0]
        lv = vals[1]
        given_file.close()
        return ls, lv

    def incoherent_board_detected(self, board):
        nb_cubes = len(np.where(board == 2)[0])
        nb_cylinders = len(np.where(board == 1)[0])

        if abs(nb_cubes - nb_cylinders) <= 1:
            return False
        else : 
            return True

class Robot():
    def __init__(self) -> None:
        self.reachy = ReachySDK('localhost')
        for f in self.reachy.fans.values():
            f.on()

    def play_pawn(self, grab_index, box_index):
        self.reachy.r_arm.r_gripper.speed_limit = 80
        self.reachy.r_arm.r_gripper.compliant = False
        self.reachy.turn_on('r_arm')

        # Goto base position
        self.goto_base_position() 
        self.reachy.r_arm.r_gripper.goal_position = -10 #open the gripper 
        path = f'/home/reachy/repos/TicTacToe/tictactoe/movements/moves-2021_right/grab_{grab_index}.npz'
        self.trajectoryPlayer(path)
        time.sleep(1)
        self.reachy.r_arm.r_gripper.compliant = False 
        self.reachy.r_arm.r_gripper.goal_position = 13 #close the gripper to take the cylinder 
        time.sleep(1)


        if grab_index >= 4:
            goto(
            goal_positions = {
                self.reachy.r_arm.r_shoulder_pitch : self.reachy.r_arm.r_shoulder_pitch.goal_position+10, 
                self.reachy.r_arm.r_elbow_pitch : self.reachy.r_arm.r_elbow_pitch.goal_position+10, 
                },
                duration =1.0,
                interpolation_mode=InterpolationMode.MINIMUM_JERK
            )

        # Lift it
        path = '/home/reachy/repos/TicTacToe/tictactoe/movements/moves-2021_right/lift.npz'
        self.goto_position(path)
        time.sleep(0.1)

        # Put it in box_index

        path = f'/home/reachy/repos/TicTacToe/tictactoe/movements/moves-2021_right/put_{box_index}.npz'
        self.trajectoryPlayer(path)
        time.sleep(1)
        self.reachy.r_arm.r_gripper.compliant = False
        self.reachy.r_arm.r_gripper.goal_position = -10
        time.sleep(1)

        path = f'/home/reachy/repos/TicTacToe/tictactoe/movements/moves-2021_right/back_{box_index}_upright.npz'
        self.goto_position(path)


        if box_index in (8, 9):

            path = '/home/reachy/repos/TicTacToe/tictactoe/movements/moves-2021_right/back_to_back.npz'
            self.goto_position(path)

        path = '/home/reachy/repos/TicTacToe/tictactoe/movements/moves-2021_right/back_rest.npz'
        self.goto_position(path)

        self.goto_rest_position()

    def goto_rest_position(self, duration=1.0):
        time.sleep(0.1)
        self.reachy.head.look_at(x=1, y=0, z=0, duration=1) 
        self.goto_base_position(0.6 * duration)
        time.sleep(0.1)

        self.reachy.turn_on('r_arm')

    def goto_base_position(self, duration=1.0):
 
        self.reachy.turn_on('r_arm')

        time.sleep(0.1)
        goto(
            goal_positions=
                    {self.reachy.r_arm.r_shoulder_pitch: 60,
                    self.reachy.r_arm.r_shoulder_roll: -15,
                    self.reachy.r_arm.r_arm_yaw: 0,
                    self.reachy.r_arm.r_elbow_pitch: -95,
                    self.reachy.r_arm.r_forearm_yaw: -15,
                    self.reachy.r_arm.r_wrist_pitch: -50,
                    self.reachy.r_arm.r_wrist_roll: 0},
                duration=1.0,
                interpolation_mode=InterpolationMode.MINIMUM_JERK
            )
        time.sleep(0.1)
        self.reachy.r_arm.r_shoulder_pitch.torque_limit = 75
        self.reachy.r_arm.r_elbow_pitch.torque_limit = 75

    def goto_position(self, path): 
        self.reachy.turn_on('r_arm')
        move = np.load(path)
        move.allow_pickle=1
        listMoves = move['move'].tolist()
        listTraj = {}
        for key,val in listMoves.items():
            listTraj[eval('self.'+key)] = float(val)
        goto(
            goal_positions=listTraj, 
            duration=2.0,
            interpolation_mode=InterpolationMode.MINIMUM_JERK
        )

    def trajectoryPlayer(self,path):
        self.reachy.turn_on('r_arm')
        move = np.load(path)
        move.allow_pickle=1
        listMoves = move['move'].tolist()
        listTraj = [val for key,val in listMoves.items()]
        listTraj = np.array(listTraj).T.tolist()

        sampling_frequency = 100  #en hertz

        recorded_joints = []
        for joint,val in listMoves.items():
            if 'neck' in joint : 
                fullName = 'self.' + joint
            elif 'r_' in joint: 
                fullName = 'self.' + joint
            elif 'l_' in joint: 
                fullName = 'self.' + joint
            recorded_joints.append(eval(fullName))
            
        first_point = dict(zip(recorded_joints, listTraj[0]))
        goto(first_point, duration=3.0)

        for joints_positions in listTraj:
            for joint, pos in zip(recorded_joints, joints_positions):
                joint.goal_position = pos
            time.sleep(1 / sampling_frequency)

    def rest(self, duration=1.0):
        self.reachy.turn_on('r_arm')
        time.sleep(0.1)
        goto(
            goal_positions=
                    {self.reachy.r_arm.r_shoulder_pitch: 55,
                    self.reachy.r_arm.r_shoulder_roll: -15,
                    self.reachy.r_arm.r_arm_yaw: 0,
                    self.reachy.r_arm.r_elbow_pitch: -85,
                    self.reachy.r_arm.r_forearm_yaw: -10,
                    self.reachy.r_arm.r_wrist_pitch: -50,
                    self.reachy.r_arm.r_wrist_roll: 0},
                duration=1.0,
                interpolation_mode=InterpolationMode.MINIMUM_JERK
            )
        time.sleep(0.1)
        self.reachy.r_arm.r_shoulder_pitch.torque_limit = 75
        self.reachy.r_arm.r_elbow_pitch.torque_limit = 75

    def happy(self):    
        self.reachy.joints.l_antenna.speed_limit = 0.0
        self.reachy.joints.r_antenna.speed_limit = 0.0
        
        
        for _ in range(9):
            self.reachy.joints.l_antenna.goal_position = 10.0
            self.reachy.joints.r_antenna.goal_position = -10.0

            time.sleep(0.1)

            self.reachy.joints.l_antenna.goal_position = -10.0
            self.reachy.joints.r_antenna.goal_position = 10.0

            time.sleep(0.1)
        
        self.reachy.joints.l_antenna.goal_position = 0.0
        self.reachy.joints.r_antenna.goal_position = 0.0
        time.sleep(1)

    def sad(reachy):
        pos = [
            (-0.5, 150),
            (-0.4, 110),
            (-0.5, 150),
            (-0.4, 110),
            (-0.5, 150),
            (0, 90),
            (0, 20),
        ]

        for (z, antenna_pos) in pos:
            reachy.head.look_at(0.5, 0.0, z, duration=1.0)
            #TC reachy.goto({
            #    'head.left_antenna': antenna_pos,#changer
            #    'head.right_antenna': -antenna_pos,
            #}, duration=1.5, wait=True, interpolation_mode='minjerk')
            reachy.joints.l_antenna.goal_position = antenna_pos
            reachy.joints.r_antenna.goal_position = -antenna_pos



board = PlaygroundTicTacToe()
robot = Robot()
lowS, lowV = board.get_HSV()

#lowS, lowV = board.calibrate_HSV()
#board.live_view()

i = 1
while (True):
    
    state = board.get_board(lowS, lowV)
    winner = board.get_winner(state)
    
    if winner == 'nobody':

        if (board.incoherent_board_detected(state)):
            double_check_board = board.get_board(lowS, lowV)
            if np.any(double_check_board != state):
                print("RESTART") 
                #tictactoe_playground.shuffle_board()
                break

        game_inv = state[::-1]
        game_inv = np.array_split(game_inv,3)
        print(game_inv[0])
        print(game_inv[1])
        print(game_inv[2])
        print('       ')
        
        action =  - board.choose_next_action(state)[0] + 9
        robot.play_pawn(i,action)
        time.sleep(2)
        
        i = i + 1 
    elif winner==1:
        print('Robot won')
        robot.reachy.turn_on('reachy')
        robot.happy()
        break
    else:
        print('Human won')
        robot.reachy.turn_on('reachy')
        robot.sad()
        break

robot.rest()
robot.reachy.turn_off('reachy')
