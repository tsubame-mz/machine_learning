from typing import NamedTuple, List
from enum import IntEnum


# 各方角のマスク
class BitMask(NamedTuple):
    H: int  # 水平方向
    V: int  # 垂直方向
    D: int  # 斜め方向
    U: int  # 上
    UR: int  # 右上
    R: int  # 右
    BR: int  # 右下
    B: int  # 下
    BL: int  # 左下
    L: int  # 左
    UL: int  # 左上


# 位置
class Position(NamedTuple):
    X: int
    Y: int


class Board:
    EMPTY = 0
    BLACK = 1
    WHITE = 2
    PIECE_STRS = [" _", " ●", " ○"]

    def __init__(self, size=8):
        self.size = size  # ボードサイズ(1辺)

        # 初期配置
        center = size // 2
        total_size = size * size
        black_bb = 1 << ((total_size - 1) - (size * (center - 1) + center))
        black_bb |= 1 << ((total_size - 1) - (size * center + (center - 1)))
        white_bb = 1 << ((total_size - 1) - (size * (center - 1) + (center - 1)))
        white_bb |= 1 << ((total_size - 1) - (size * center + center))
        self.black_bb = black_bb
        self.white_bb = white_bb

        # 石の数
        self.black_score = 2
        self.white_score = 2
        self.blank_num = total_size - 4  # 空きマス数

        # 各方角のマスク
        self.mask_bb = BitMask(
            H=int("".join((["0"] + (["1"] * (size - 2)) + ["0"]) * size), 2),
            V=int("".join((["0"] * size) + ((["1"] * size) * (size - 2)) + (["0"] * size)), 2),
            D=int("".join((["0"] * size) + ((["0"] + (["1"] * (size - 2)) + ["0"]) * (size - 2)) + (["0"] * size)), 2),
            U=int("".join(((["1"] * size) * (size - 1)) + (["0"] * size)), 2),
            UR=int("".join(((["0"] + (["1"] * (size - 1))) * (size - 1)) + (["0"] * size)), 2),
            R=int("".join((["0"] + (["1"] * (size - 1))) * size), 2),
            BR=int("".join((["0"] * size) + ((["0"] + (["1"] * (size - 1))) * (size - 1))), 2),
            B=int("".join((["0"] * size) + ((["1"] * size) * (size - 1))), 2),
            BL=int("".join((["0"] * size) + (((["1"] * (size - 1)) + ["0"]) * (size - 1))), 2),
            L=int("".join(((["1"] * (size - 1)) + ["0"]) * size), 2),
            UL=int("".join((((["1"] * (size - 1)) + ["0"]) * (size - 1)) + (["0"] * size)), 2),
        )

        # 履歴
        self.history = []

        self.current_color = self.BLACK
        self.turn = 0

    def get_board_info(self):
        return {
            "turn": self.turn,
            "blank_num": self.blank_num,
            "current_color": self.current_color,
            "black_bb": self.black_bb,
            "white_bb": self.white_bb,
            "black_score": self.black_score,
            "white_score": self.white_score,
        }

    def get_legal_moves(self) -> List[Position]:
        if self.current_color == self.BLACK:
            player_bb, opponent_bb = self.black_bb, self.white_bb
        else:
            player_bb, opponent_bb = self.white_bb, self.black_bb

        horizontal_bb = opponent_bb & self.mask_bb.H
        vertical_bb = opponent_bb & self.mask_bb.V
        diagonal_bb = opponent_bb & self.mask_bb.D
        blank_bb = ~(player_bb | opponent_bb)  # 空きマス

        legal_moves_bb = 0
        legal_moves_bb |= self._get_legal_moves_lshift(horizontal_bb, player_bb, blank_bb, 1)  # 左
        legal_moves_bb |= self._get_legal_moves_rshift(horizontal_bb, player_bb, blank_bb, 1)  # 右
        legal_moves_bb |= self._get_legal_moves_lshift(vertical_bb, player_bb, blank_bb, self.size)  # 上
        legal_moves_bb |= self._get_legal_moves_rshift(vertical_bb, player_bb, blank_bb, self.size)  # 上
        legal_moves_bb |= self._get_legal_moves_lshift(diagonal_bb, player_bb, blank_bb, self.size + 1)  # 左上
        legal_moves_bb |= self._get_legal_moves_lshift(diagonal_bb, player_bb, blank_bb, self.size - 1)  # 右上
        legal_moves_bb |= self._get_legal_moves_rshift(diagonal_bb, player_bb, blank_bb, self.size - 1)  # 左下
        legal_moves_bb |= self._get_legal_moves_rshift(diagonal_bb, player_bb, blank_bb, self.size + 1)  # 右下

        legal_moves = []
        mask = 1 << (self.size * self.size - 1)
        for y in range(self.size):
            for x in range(self.size):
                if legal_moves_bb & mask:
                    legal_moves.append(Position(X=x, Y=y))
                mask >>= 1

        return legal_moves

    def move(self, pos: Position) -> int:
        size = self.size
        total_size = size * size
        pos_index = (total_size - 1) - ((pos.Y * size) + pos.X)
        if (pos_index < 0) or (pos_index > (total_size - 1)):
            return 0

        flippable_discs = self.get_flippable_discs(pos)
        if len(flippable_discs) == 0:
            return 0

        flippable_discs_bb = 0
        for flip_pos in flippable_discs:
            flippable_discs_bb |= 1 << ((total_size - 1) - ((flip_pos.Y * size) + flip_pos.X))

        put_bb = 1 << pos_index
        if self.current_color == self.BLACK:
            self.black_bb ^= put_bb | flippable_discs_bb
            self.white_bb ^= flippable_discs_bb
            self.black_score += 1 + len(flippable_discs)
            self.white_score -= len(flippable_discs)
            self.current_color = self.WHITE
        else:
            self.white_bb ^= put_bb | flippable_discs_bb
            self.black_bb ^= flippable_discs_bb
            self.white_score += 1 + len(flippable_discs)
            self.black_score -= len(flippable_discs)
            self.current_color = self.BLACK
        self.turn += 1
        self.blank_num -= 1

        self.history.append({"pos": pos, "flippable_discs": flippable_discs})

        return flippable_discs_bb

    def pass_move(self):
        self.history.append(None)
        if self.current_color == self.BLACK:
            self.current_color = self.WHITE
        else:
            self.current_color = self.BLACK
        self.turn += 1

    def undo(self):
        his_data = self.history.pop()
        if his_data:
            size = self.size
            total_size = size * size

            pos = his_data["pos"]
            flippable_discs = his_data["flippable_discs"]

            flippable_discs_bb = 0
            for flip_pos in flippable_discs:
                flippable_discs_bb |= 1 << ((total_size - 1) - ((flip_pos.Y * size) + flip_pos.X))

            pos_index = (total_size - 1) - ((pos.Y * size) + pos.X)
            put_bb = 1 << pos_index
            if self.current_color == self.BLACK:
                self.white_bb ^= put_bb | flippable_discs_bb
                self.black_bb ^= flippable_discs_bb
                self.white_score -= 1 + len(flippable_discs)
                self.black_score += len(flippable_discs)
            else:
                self.black_bb ^= put_bb | flippable_discs_bb
                self.white_bb ^= flippable_discs_bb
                self.black_score -= 1 + len(flippable_discs)
                self.white_score += len(flippable_discs)
            self.blank_num += 1

        if self.current_color == self.BLACK:
            self.current_color = self.WHITE
        else:
            self.current_color = self.BLACK
        self.turn -= 1

    def get_flippable_discs(self, pos: Position):
        size = self.size

        if self.current_color == self.BLACK:
            player_bb, opponent_bb = self.black_bb, self.white_bb
        else:
            player_bb, opponent_bb = self.white_bb, self.black_bb

        # 置く場所
        pos_index = 1 << (((size * size) - 1) - ((pos.Y * size) + pos.X))

        reversibles_bb = 0
        for direction in ["U", "UR", "R", "BR", "B", "BL", "L", "UL"]:
            check_bb = self._get_next_put(pos_index, direction)

            # 相手の石がなくなるまで移動
            temp_bb = 0
            while check_bb & opponent_bb:
                temp_bb |= check_bb
                check_bb = self._get_next_put(check_bb, direction)

            # 対岸に自分の石があれば相手の石の位置を保持
            if check_bb & player_bb:
                reversibles_bb |= temp_bb

        reversibles = []
        mask = 1 << (self.size * self.size - 1)
        for y in range(self.size):
            for x in range(self.size):
                if reversibles_bb & mask:
                    reversibles.append(Position(X=x, Y=y))
                mask >>= 1

        return reversibles

    def get_history(self):
        return self.history

    def get_blank_num(self):
        return self.blank_num

    def __str__(self) -> str:
        size = self.size
        total_size = size * size
        mask = 1 << (total_size - 1)

        out_str = ""
        out_str += "    " + " ".join([chr(97 + i) for i in range(size)]) + "\n"
        for y in range(size):
            out_str += "{:2d} ".format(y + 1)
            for x in range(size):
                if self.black_bb & mask:
                    out_str += self.PIECE_STRS[self.BLACK]
                elif self.white_bb & mask:
                    out_str += self.PIECE_STRS[self.WHITE]
                else:
                    out_str += self.PIECE_STRS[self.EMPTY]
                mask >>= 1
            out_str += "\n"
        black_score_str = self.PIECE_STRS[self.BLACK] + "[{}]".format(self.black_score)
        white_score_str = self.PIECE_STRS[self.WHITE] + "[{}]".format(self.white_score)
        out_str += "Score: " + black_score_str + " / " + white_score_str + "\n"
        out_str += "Turn: {:2d}".format(self.turn) + ",  "
        out_str += "Current: {}".format(self.PIECE_STRS[self.current_color])

        return out_str

    def _get_legal_moves_lshift(self, mask_bb, player_bb, blank_bb, shift_size) -> int:
        temp_bb = mask_bb & (player_bb << shift_size)
        for _ in range(self.size - 3):
            temp_bb |= mask_bb & (temp_bb << shift_size)
        return blank_bb & (temp_bb << shift_size)

    def _get_legal_moves_rshift(self, mask_bb, player_bb, blank_bb, shift_size) -> int:
        temp_bb = mask_bb & (player_bb >> shift_size)
        for _ in range(self.size - 3):
            temp_bb |= mask_bb & (temp_bb >> shift_size)
        return blank_bb & (temp_bb >> shift_size)

    def _get_next_put(self, index, direction):
        if direction == "U":
            return (index << self.size) & self.mask_bb.U
        elif direction == "UR":
            return (index << (self.size - 1)) & self.mask_bb.UR
        elif direction == "R":
            return (index >> 1) & self.mask_bb.R
        elif direction == "BR":
            return (index >> (self.size + 1)) & self.mask_bb.BR
        elif direction == "B":
            return (index >> self.size) & self.mask_bb.B
        elif direction == "BL":
            return (index >> (self.size - 1)) & self.mask_bb.BL
        elif direction == "L":
            return (index << 1) & self.mask_bb.L
        elif direction == "UL":
            return (index << (self.size + 1)) & self.mask_bb.UL
        else:
            return 0
