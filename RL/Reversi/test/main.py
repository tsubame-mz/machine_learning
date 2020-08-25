import sys
import random

sys.path.append("../")
from reversi.board import Board  # NOQA

size = 8
board = Board(size)

done = False
pass_num = 0
while not done:
    print("=" * 80)
    print(board)
    board_info = board.get_board_info()
    legal_moves = board.get_legal_moves()
    print("board_info: ", board_info)
    print("legal_moves: ", legal_moves)

    if len(legal_moves) == 0:
        print("action: Pass")
        board.pass_move()
        pass_num += 1
    else:
        action = random.choice(legal_moves)
        print("action: ", action)
        reversibles = board.get_flippable_discs(action)
        print("reversibles: ", reversibles)
        board.move(action)
        pass_num = 0

    if (pass_num == 2) or (board.get_blank_num() == 0):
        done = True

print("=" * 80)
print(board)
board_info = board.get_board_info()
if board_info["black_bb"] > board_info["white_bb"]:
    print("Black win")
elif board_info["black_bb"] < board_info["white_bb"]:
    print("White win")
else:
    print("Draw")
print("=" * 80)
print("History")
print(board.get_history())
