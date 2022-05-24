from utils import *
import time
from rl_agent import *

from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto
from reachy_sdk.trajectory.interpolation import InterpolationMode

model       = intializePredectionModel()  
heightImg   = 300
widthImg    = 300


class PlaygroundTicTacToe():
    def __init__(self):
        rclpy.init()
        time.sleep(2)
        self.image_getter = RosCameraSubscriber(node_name='image_viewer', side = "right")
        


    def reset(self):
        self.pawn_played = 0
        empty_board = np.zeros((3, 3), dtype=np.uint8).flatten()
        return empty_board

 # Playground and game functions
    def coin_flip(self):
        coin = np.random.rand() > 0.5
        #print(coin)
        #print('reachy' if coin else 'human')
        return coin

    def live_view(self):
        while True:
            self.image_getter.update_image()
            img = self.image_getter.cam_img
        
            #### 1. PREPARE THE IMAGE
            img = cv2.resize(img, (widthImg, heightImg))                                                        # RESIZE IMAGE TO MAKE IT A SQUARE IMAGE
            imgThreshold = preProcessHSV(img, low, high)


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

    def get_board(self, low, high):
        boards = []
        i = 0
        for i in range(10):
            self.image_getter.update_image()
            img = self.image_getter.cam_img
            img = cv2.resize(img, (widthImg, heightImg))                                                        # RESIZE IMAGE TO MAKE IT A SQUARE IMAGE
            imgThreshold = preProcessHSV(img, low, high)
            contours, _ = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)            # FIND ALL CONTOURS
            biggest, maxArea = biggestContour(contours)                                                               # FIND THE BIGGEST CONTOUR
            if biggest.size != 0 and maxArea>25000:
                biggest = reorder(biggest)
                pts1 = np.float32(biggest)                                                                      # PREPARE POINTS FOR WARP
                pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]])                 # PREPARE POINTS FOR WARP
                matrix = cv2.getPerspectiveTransform(pts1, pts2)                                                # GER
                imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
                boxes = splitBoxes(imgWarpColored)
                numbers = getPredection(boxes, model)
                numbers = np.asarray(numbers)
            #boards.append(numbers)
            return numbers
        else:   
            return None

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
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            file = open("hsv.txt","w")
            L = [str(ilowH)+' '+str(ilowS)+' '+str(ilowV)+' ']
            H = [str(ihighH)+' '+str(ihighS)+' '+str(ihighV)] 
            file.writelines(L+H)
            file.close()
        return lower_hsv, higher_hsv

    def get_HSV(self):
        given_file = open('hsv.txt', 'r')

        lines = given_file.readlines()
        vals = []
        for line in lines:
            for word in line.split():
                    vals.append(int(word))
        l = vals[0], vals[1], vals[2]
        h = vals[3],vals[4],vals[5]
        given_file.close()
        return l, h

    def incoherent_board_detected(self, board):
        nb_cubes = len(np.where(board == 2)[0])
        nb_cylinders = len(np.where(board == 1)[0])

        if abs(nb_cubes - nb_cylinders) <= 1:
            return False
        else : 
            return True

    def cheating_detected(self, board, last_board, reachy_turn):
        # last is just after the robot played
        delta = board - last_board

        # Nothing changed
        if np.all(delta == 0):
            return False

        # A single cube was added
        if len(np.where(delta == 1)) == 1:
            return False

        # A single cylinder was added
        if len(np.where(delta == 2)) == 1:
            # If the human added a cylinder
            if not reachy_turn:
                return True
            return False

        return True

    def has_human_played(self, current_board, last_board):
        cube = 2

        return (
            np.any(current_board != last_board) and
            np.sum(current_board == cube) > np.sum(last_board == cube)
        )
    def is_final(self, board):
        winner = self.get_winner(board)
        if winner in (1, 2):
            return True
        else:
            return 0 not in board


class Robot():
    def __init__(self) -> None:
        self.reachy = ReachySDK('localhost')
        for f in self.reachy.fans.values():
            f.on()
    def reset(self):
        self.pawn_played = 0

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
        self.reachy.turn_on('reachy')    
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

    def sad(self):
        self.reachy.turn_on('reachy')  
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
            #reachy.head.look_at(0.5, 0.0, z, duration=1.0)
            #TC reachy.goto({
            #    'head.left_antenna': antenna_pos,#changer
            #    'head.right_antenna': -antenna_pos,
            #}, duration=1.5, wait=True, interpolation_mode='minjerk')
            self.reachy.joints.l_antenna.goal_position = antenna_pos
            self.reachy.joints.r_antenna.goal_position = -antenna_pos

    def run_my_turn(self):
        self.goto_base_position()
        self.reachy.turn_on('r_arm')
        path = '/home/reachy/repos/TicTacToe/tictactoe/movements/moves-2021_right/my-turn.npz'
        self.trajectoryPlayer(path)
        self.goto_rest_position()

    def run_your_turn(self):
        self.goto_base_position()
        self.reachy.turn_on('r_arm')
        path = '/home/reachy/repos/TicTacToe/tictactoe/movements/moves-2021_right/your-turn.npz'
        self.trajectoryPlayer(path)
        self.goto_rest_position()

    def run_random_idle_behavior(self):
        time.sleep(2)

    def play(self, action, actual_board):
        board = actual_board.copy()
        self.play_pawn(
            grab_index=self.pawn_played + 1,
            box_index=action + 1,
        )

        self.pawn_played += 1

        board[action] = 1

        return board


playground = PlaygroundTicTacToe()
robot      = Robot()
low, high = playground.get_HSV()
boardEmpty = np.zeros((3, 3), dtype=np.uint8).flatten()

#low, high = playground.calibrate_HSV()
playground.live_view()
    


option = 0

if option == 0:
    robot.reset()
   
    last_board = playground.reset()
    reachy_turn = playground.coin_flip()

    if reachy_turn:
        robot.run_my_turn()
    else:
        robot.run_your_turn()
    
    #game loop
    while True:
        board = playground.get_board(low, high)

        if board is not None:
            print("in")
            game = np.array_split(board,3)
            print(game[0])
            print(game[1])
            print(game[2])
            print('     ')
            print('     ')
            if not reachy_turn:
                if playground.has_human_played(board, last_board):
                    reachy_turn = True
                else:
                    robot.run_random_idle_behavior()

            if (playground.incoherent_board_detected(board) or
                        playground.cheating_detected(board, last_board, reachy_turn)):
                print("incoherent board or cheating")
                double_check_board = playground.get_board(low, high)
                print(double_check_board)
                if np.any(double_check_board != last_board): 
                    #tictactoe_playground.shuffle_board()
                    print("shuffle board")
                    break
                else :
                        # False detection, we will check again next loop
                        continue
                
            if (not playground.is_final(board)) and reachy_turn:
                action = -playground.choose_next_action(board)[0]+8

                board = robot.play(action, board)
                
                last_board = board
                reachy_turn = False
            
            if playground.is_final(board):
                winner = playground.get_winner(board)

                if winner == 1:
                    robot.happy()
                    winner = "Robot"
                elif winner == 2:
                    robot.sad()
                    winner = "Human"
                else:
                    robot.sad()
                    winner = "Tie"
                print(winner)
elif option==1:
    j = 488
    for k in range(150):
    #Save images for dataset
        for i in range(9):
            boxes = playground.get_images(low, high)
            cv2.imwrite('images/new/'+str(j)+'.jpg', boxes[i])
            j=j+1
        k=k+1
        time.sleep(6)
else:
    while True:
        #print(low)
        #print(high)
        board = playground.get_board(low, high)
        print(board)