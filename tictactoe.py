print('Setting UP')
from utils import *
import time
from rl_agent import *

model = intializePredectionModel()  # LOAD THE CNN MODEL
heightImg = 300
widthImg = 300

class PlaygroundTicTacToe():
    def __init__(self):
        self.reachy = ReachySDK('localhost')
        rclpy.init()
        time.sleep(2)
        self.image_getter = RosCameraSubscriber(node_name='image_viewer', side = "right")

    def setup(self):
        self.reachy.turn_on('head')
        self.reachy.head.look_at(x=1, y=0, z=0, duration=1.5) 
        self.reachy.head.l_antenna.speed_limit = 50.0
        self.reachy.head.r_antenna.speed_limit = 50.0
        self.reachy.head.l_antenna.goal_position = 0
        self.reachy.head.r_antenna.goal_position = 0
        #self.goto_rest_position()

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
            imgThreshold = preProcessHSV(img)


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

    def get_board(self):
        self.image_getter.update_image()
        img = self.image_getter.cam_img
        img = cv2.resize(img, (widthImg, heightImg))                                                        # RESIZE IMAGE TO MAKE IT A SQUARE IMAGE
        imgThreshold = preProcessHSV(img)
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

    def trajectoryPlayer(self , path):
        self.reachy.turn_on('r_arm')
        move = np.load(path)
        move.allow_pickle=1

        listMoves = move['move'].tolist()
        listTraj = [ val for key,val in listMoves.items()]
        listTraj = np.array(listTraj).T.tolist()

        sampling_frequency = 100  #en hertz
        
        recorded_joints = []
        for joint,val in listMoves.items():
            if 'neck' in joint : 
                fullName = 'self.'+ joint
            elif 'r_' in joint: 
                fullName = 'self.'+joint
            elif 'l_' in joint: 
                fullName = 'self.'+joint
            recorded_joints.append(eval(fullName))

        for joint in recorded_joints:
            joint.compliant = False

        first_point = dict(zip(recorded_joints, listTraj[0]))
        goto(first_point, duration=3.0)

        for joints_positions in listTraj:
            for joint, pos in zip(recorded_joints, joints_positions):
                joint.goal_position = pos
            time.sleep(1 / sampling_frequency)

    def goto_rest_position(self, duration=1.0):
        time.sleep(0.1)
        self.reachy.head.look_at(x=1, y=0, z=0, duration=1) 
        self.goto_base_position(0.6 * duration)
        time.sleep(0.1)

        self.reachy.turn_on('r_arm')


        goto(
            goal_positions=
                    {self.reachy.r_arm.r_shoulder_pitch: 50,
                    self.reachy.r_arm.r_shoulder_roll: -15,
                    self.reachy.r_arm.r_arm_yaw: 0,
                    self.reachy.r_arm.r_elbow_pitch: -100,
                    self.reachy.r_arm.r_forearm_yaw: -15,
                    self.reachy.r_arm.r_wrist_pitch: -60,
                    self.reachy.r_arm.r_wrist_roll: 0},
                duration=1.0,
                interpolation_mode=InterpolationMode.MINIMUM_JERK
            )
        time.sleep(1)


        time.sleep(0.25)

        self.reachy.r_arm.r_shoulder_roll.comliant = True
        self.reachy.r_arm.r_arm_yaw.comliant = True
        self.reachy.r_arm.r_elbow_pitch.comliant = True
        self.reachy.r_arm.r_forearm_yaw.comliant = True
        self.reachy.r_arm.r_wrist_pitch.comliant = True
        self.reachy.r_arm.r_wrist_roll.comliant = True
        self.reachy.r_arm.r_gripper.comliant = True

        #self.reachy.turn_off('r_arm')

        time.sleep(0.25)

    def play_pawn(self, grab_index, box_index):
        self.reachy.r_arm.r_gripper.speed_limit = 80
        self.reachy.r_arm.r_gripper.compliant = False
        self.reachy.turn_on('r_arm')

        # Goto base position
        self.goto_base_position() 
        self.reachy.r_arm.r_gripper.goal_position = -10 #open the gripper 
        path = f'/home/reachy/repos/TicTacToe/tictactoe/movements/moves-2021_right/grab_{grab_index}.npz'
        self.goto_position(path)
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



board = PlaygroundTicTacToe()
#board.live_view()
#state = board.get_board()
#print(state[::-1])
#state_m  = np.where(state == 1, 3, state)
#state_m  = np.where(state_m == 2, 1, state_m)
#state_m  = np.where(state_m == 3, 2, state_m)

#board.play_pawn(1,1)
#print(state_m)
#print(board.choose_next_action(state_m))
#action = - board.choose_next_action(state_m)[0] +9
#print(action)


for i in range(5):
    state = board.get_board()
    winner = board.get_winner(state)
    if winner == 'nobody':
        state_m  = np.where(state == 1, 3, state)
        state_m  = np.where(state_m == 2, 1, state_m)
        state_m  = np.where(state_m == 3, 2, state_m)
        #game = state_m[::-1]
        game = np.array_split(state_m,3)
        print(game[0])
        print(game[1])
        print(game[2])
        print('     ')
        print('     ')

        action = - board.choose_next_action(state_m)[0] +9
        board.play_pawn(1,action)
        time.sleep(2)
    else:
        board.goto_rest_position()
        board.rest()
        board.reachy.turn_off('r_arm')
        break

