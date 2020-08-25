import unittest
import sys

sys.path.append("../")
from reversi.board import Board  # NOQA


class TestBoardMask(unittest.TestCase):
    def test_4x4(self):
        board = Board(size=4)
        mask_bb = board.mask_bb
        self.assertEqual(0x6666, mask_bb.H)
        self.assertEqual(0x0FF0, mask_bb.V)
        self.assertEqual(0x0660, mask_bb.D)
        self.assertEqual(0xFFF0, mask_bb.U)
        self.assertEqual(0x7770, mask_bb.UR)
        self.assertEqual(0x7777, mask_bb.R)
        self.assertEqual(0x0777, mask_bb.BR)
        self.assertEqual(0x0FFF, mask_bb.B)
        self.assertEqual(0x0EEE, mask_bb.BL)
        self.assertEqual(0xEEEE, mask_bb.L)
        self.assertEqual(0xEEE0, mask_bb.UL)

    def test_8x8(self):
        board = Board(size=8)
        mask_bb = board.mask_bb
        self.assertEqual(0x7E7E7E7E7E7E7E7E, mask_bb.H)
        self.assertEqual(0x00FFFFFFFFFFFF00, mask_bb.V)
        self.assertEqual(0x007E7E7E7E7E7E00, mask_bb.D)
        self.assertEqual(0xFFFFFFFFFFFFFF00, mask_bb.U)
        self.assertEqual(0x7F7F7F7F7F7F7F00, mask_bb.UR)
        self.assertEqual(0x7F7F7F7F7F7F7F7F, mask_bb.R)
        self.assertEqual(0x007F7F7F7F7F7F7F, mask_bb.BR)
        self.assertEqual(0x00FFFFFFFFFFFFFF, mask_bb.B)
        self.assertEqual(0x00FEFEFEFEFEFEFE, mask_bb.BL)
        self.assertEqual(0xFEFEFEFEFEFEFEFE, mask_bb.L)
        self.assertEqual(0xFEFEFEFEFEFEFE00, mask_bb.UL)


if __name__ == "__main__":
    unittest.main()
