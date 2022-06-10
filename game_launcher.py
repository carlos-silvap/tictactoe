
from classes import PlaygroundTicTacToe, Robot
from utils import np

def game(playground, robot):
    ready = False
    
    while not ready:
        board = playground.get_board(low, high)
        if np.all(board == 0):
            ready = True
        #print("not ready")

    robot.reset()
    last_board = playground.reset()
    reachy_turn = playground.coin_flip()

    if reachy_turn:
        robot.run_my_turn()
    else:
        robot.run_your_turn()
    
    
    while True:
        board = playground.get_board(low, high)
        print('LBoard', last_board)
        print('NBoard',board)
        print(' ')
        delta = board - last_board
        delta = np.sum(delta)
        print(delta)
        if delta <= 2 and delta >=0:
            for i in range(9):
                if last_board[i] == 1:
                    if board[i] != 1:
                        print('FALSE 1')
                        board = playground.get_board(low, high)
                        print(board)
                elif last_board[i] == 2:
                    if board[i] != 2:
                        print('FALSE 2')
                        board = playground.get_board(low, high)
                        print(board)

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
                    print("false detection")
                        # False detection, we will check again next loop
                    continue
                
            if (not playground.is_final(board)) and reachy_turn:
                action = -playground.choose_next_action(board)[0]+8
                robot.play(action, board)
                board = playground.get_board(low, high)
                #delta = np.sum(last_board)
                #delta1 =np.sum(board)
                #if np.sum(last_board) == np.sum(board):

                print('RBoard', board)
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
                return winner

playground = PlaygroundTicTacToe()  
robot =  Robot() 

if __name__ == "__main__":
    #playground.calibrate_HSV()
    low, high = playground.get_HSV()
    playground.live_view(low, high)
    
    
    playground.reset()
    robot.reset()
    game_played = 0
        
    while True:
        winner = game(playground, robot)
        game_played += 1
        print(game_played)
