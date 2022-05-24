from tictactoe import PlaygroundTicTacToe, Robot
from utils import np

def game_loop(playground, robot):
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



if __name__ == '__main__':

    if args.log_file is not None:
        n = len(glob(f'{args.log_file}*.log')) + 1

        now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%f')
        args.log_file += f'-{n}-{now}.log'

    logger = zzlog.setup(
        logger_root='',
        filename=args.log_file,
    )

    logger.info(
        'Creating a Tic Tac Toe playground.'
    )

    with TictactoePlayground() as tictactoe_playground:
        tictactoe_playground.setup()

        game_played = 0

        while True:
            winner = run_game_loop(tictactoe_playground)
            game_played += 1
            logger.info(
                'Game ended',
                extra={
                    'game_number': game_played,
                    'winner': winner,
                }
            )

            if tictactoe_playground.need_cooldown():
                logger.warning('Reachy needs cooldown')
                tictactoe_playground.enter_sleep_mode()
                tictactoe_playground.wait_for_cooldown()
                tictactoe_playground.leave_sleep_mode()
                logger.info('Reachy cooldown finished')
