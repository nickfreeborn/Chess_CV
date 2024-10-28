import numpy as np
from enum import Enum
from collections import namedtuple
from itertools import count

N, E, S, W = -10, 1, 10, -1
directions = {
    "P": (N, N+N, N+W, N+E),
    "N": (N+N+E, E+N+E, E+S+E, S+S+E, S+S+W, W+S+W, W+N+W, N+N+W),
    "B": (N+E, S+E, S+W, N+W),
    "R": (N, E, S, W),
    "Q": (N, E, S, W, N+E, S+E, S+W, N+W),
    "K": (N, E, S, W, N+E, S+E, S+W, N+W)
}
A1, H1, A8, H8 = 91, 98, 21, 28
Move = namedtuple("Move", "i j prom")

# class Colour(Enum):
#     NONE = -1
#     BLACK = 0
#     WHITE = 1

# class PieceType(Enum):
#     NONE = ""
#     PAWN = "Pawn"
#     KNIGHT = "Knight"
#     BISHOP = "Bishop"
#     ROOK = "Rook"
#     QUEEN = "Queen"
#     KING = "King"

class Position(namedtuple("Position", "board score wc bc ep kp")):
    """A state of a chess game
    board -- a 120 char representation of the board
    score -- the board evaluation
    wc -- the castling rights, [west/queen side, east/king side]
    bc -- the opponent castling rights, [west/king side, east/queen side]
    ep - the en passant square
    kp - the king passant square
    """

    def __str__(self):
        return self.board[20:99]

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            # Positions are equivalent if and only if the pieces are set up the same
            # Castling rights, en-passant and king-passant squares are ignored for now
            matchwithoutrotate = (self.board == other.board)
            self.rotate()
            matchwithrotate = (self.board == other.board)
            self.rotate()
            if (matchwithoutrotate or matchwithrotate):
                return True
            return False
        else:
            return False

    def isLegal(self):
        pass

    def gen_moves(self):
            # For each of our pieces, iterate through each possible 'ray' of moves,
            # as defined in the 'directions' map. The rays are broken e.g. by
            # captures or immediately in case of pieces such as knights.
            for i, p in enumerate(self.board):
                if not p.isupper():
                    continue
                for d in directions[p]:
                    for j in count(i + d, d):
                        q = self.board[j]
                        # Stay inside the board, and off friendly pieces
                        if q.isspace() or q.isupper():
                            break
                        # Pawn move, double move and capture
                        if p == "P":
                            if d in (N, N + N) and q != ".": break
                            if d == N + N and (i < A1 + N or self.board[i + N] != "."): break
                            if (
                                d in (N + W, N + E)
                                and q == "."
                                and j not in (self.ep, self.kp, self.kp - 1, self.kp + 1)
                                #and j != self.ep and abs(j - self.kp) >= 2
                            ):
                                break
                            # If we move to the last row, we can be anything
                            if A8 <= j <= H8:
                                for prom in "NBRQ":
                                    yield Move(i, j, prom)
                                break
                        # Move it
                        yield Move(i, j, "")
                        # Stop crawlers from sliding, and sliding after captures
                        if p in "PNK" or q.islower():
                            break
                        # Castling, by sliding the rook next to the king
                        if i == A1 and self.board[j + E] == "K" and self.wc[0]:
                            yield Move(j + E, j + W, "")
                        if i == H1 and self.board[j + W] == "K" and self.wc[1]:
                            yield Move(j + W, j + E, "")
    
    def move(self, move):
        i, j, prom = move
        p, q = self.board[i], self.board[j]
        put = lambda board, i, p: board[:i] + p + board[i + 1 :]
        # Copy variables and reset ep and kp
        board = self.board
        wc, bc, ep, kp = self.wc, self.bc, 0, 0
        # score = self.score + self.value(move)
        # Actual move
        board = put(board, j, board[i])
        board = put(board, i, ".")
        # Castling rights, we move the rook or capture the opponent's
        if i == A1: wc = (False, wc[1])
        if i == H1: wc = (wc[0], False)
        if j == A8: bc = (bc[0], False)
        if j == H8: bc = (False, bc[1])
        # Castling
        if p == "K":
            wc = (False, False)
            if abs(j - i) == 2:
                kp = (i + j) // 2
                board = put(board, A1 if j < i else H1, ".")
                board = put(board, kp, "R")
        # Pawn promotion, double move and en passant capture
        if p == "P":
            if A8 <= j <= H8:
                board = put(board, j, prom)
            if j - i == 2 * N:
                ep = i + N
            if j == self.ep:
                board = put(board, j + S, ".")
        # We rotate the returned position, so it's ready for the next player
        # return Position(board, score, wc, bc, ep, kp).rotate()
        return Position(board, 0, wc, bc, ep, kp)
    
    def rotate(self, nullmove=False):
        """Rotates the board, preserving enpassant, unless nullmove"""
        return Position(
            self.board[::-1].swapcase(), -self.score, self.bc, self.wc,
            119 - self.ep if self.ep and not nullmove else 0,
            119 - self.kp if self.kp and not nullmove else 0,
        )

def getBoard(board_repr):

    board = []
    for i in range(120):
        row = (i // 10)
        col = (i % 10)

        if col == 9:
            board.append("\n")
            continue

        if i < 20 or i > 99:
            board.append(" ")
            continue

        if col == 0:
            board.append(" ")
            continue

        piece = board_repr[row - 2][col - 1]

        c = piece_symbol(piece)
    
        # if colour == Colour.NONE.value:
        #     board.append(".")
        #     continue
        
        # # set c to the first character of the corresponding entry in 'pieces', for example K for King, B for Bishop.
        # if pieces[row-2][col-1] == PieceType.KNIGHT.value:
        #     # set c to "N" it it's a Knight. This way we don't confuse it with K for King
        #     c = "N"
        # else:
        #     c = pieces[row-2][col-1][0]

        # if colour == Colour.BLACK.value:
        #     c = c.lower()
        
        board.append(c)
    
    return ''.join(board)

def piece_symbol(piece_id):
    symbols = {
        -1: ".",  # Empty square
        0: "P",   # White Pawn
        1: "R",   # White Rook
        2: "B",   # White Bishop
        3: "N",   # White Knight
        4: "K",   # White King
        5: "Q",   # White Queen
        6: "p",   # Black Pawn
        7: "r",   # Black Rook
        8: "b",   # Black Bishop
        9: "q",   # Black Knight
        10: "k",  # Black King
        11: "n"   # Black Queen
    }
    return symbols.get(piece_id, "?")

# pieces = [
#     ["Rook", "Knight", "Bishop", "Queen", "King", "Bishop", "Knight", "Rook"],
#     ["Pawn", "Pawn", "Pawn", "Pawn", "Pawn", "Pawn", "Pawn", "Pawn"],
#     [" ", " ", " ", " ", " ", " ", " ", " "],
#     [" ", " ", " ", " ", " ", " ", " ", " "],
#     [" ", " ", " ", " ", " ", " ", " ", " "],
#     [" ", " ", " ", " ", " ", " ", " ", " "],
#     ["Pawn", "Pawn", "Pawn", "Pawn", "Pawn", "Pawn", "Pawn", "Pawn"],
#     ["Rook", "Knight", "Bishop", "Queen", "King", "Bishop", "Knight", "Rook"]
# ]

# colours = [
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [-1, -1, -1, -1, -1, -1, -1, -1],
#     [-1, -1, -1, -1, -1, -1, -1, -1],
#     [-1, -1, -1, -1, -1, -1, -1, -1],
#     [-1, -1, -1, -1, -1, -1, -1, -1],
#     [1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1],
# ]

# current_pieces = [
#     ["Rook", "Knight", "Bishop", "Queen", "King", "Bishop", "Knight", "Rook"],
#     ["Pawn", "Pawn", "Pawn", "Pawn", "Pawn", "Pawn", "Pawn", "Pawn"],
#     ["", "", "", "", "", "", "", ""],
#     ["", "", "", "", "", "", "", ""],
#     ["", "Pawn", "", "", "", "", "", ""],
#     ["", "", "", "", "", "", "", ""],
#     ["Pawn", "", "Pawn", "Pawn", "Pawn", "Pawn", "Pawn", "Pawn"],
#     ["Rook", "Knight", "Bishop", "Queen", "King", "Bishop", "Knight", "Rook"]
# ]

# current_colours = [
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [-1, -1, -1, -1, -1, -1, -1, -1],
#     [-1, -1, -1, -1, -1, -1, -1, -1],
#     [-1, 1, -1, -1, -1, -1, -1, -1],
#     [-1, -1, -1, -1, -1, -1, -1, -1],
#     [1, -1, 1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1],
# ]
# board = getBoard(colours, pieces)
# myPosition = Position(board=board, score=0, wc=[True, True], bc=[True, True], ep=0, kp=0)
# # print(myPosition)

# for m in myPosition.gen_moves():
#     new_Position = myPosition.move(m)
#     # print(new_Position)
#     # print("\n")

# old_board_repr = [
#     [7, 11, 8, 9, 10, 8, 11, 7],
#     [6, 6, 6, 6, 6, 6, 6, 6],
#     [-1, -1, -1, -1, -1, -1, -1, -1],
#     [-1, -1, -1, -1, -1, -1, -1, -1],
#     [-1, -1, -1, -1, -1, -1, -1, -1],
#     [-1, -1, -1, -1, -1, -1, -1, -1],
#     [1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 3, 2, 5, 4, 2, 3, 1],
# ]

# new_board_repr = [
#     [7, 11, 8, 9, 10, 8, 11, 7],
#     [6, 6, 6, 6, 6, 6, 6, 6],
#     [-1, -1, -1, -1, -1, -1, -1, -1],
#     [-1, -1, -1, -1, -1, -1, -1, -1],
#     [-1, -1, -1, -1, 1, -1, -1, -1],
#     [-1, -1, -1, -1, -1, -1, -1, -1],
#     [1, 1, 1, 1, -1, 1, 1, 1],
#     [1, 3, 2, 5, 4, 2, 3, 1],
# ]

# old_board = getBoard(old_board_repr)
# current_board = getBoard(new_board_repr)
# oldPosition = Position(board=old_board, score=0, wc=[True, True], bc=[True, True], ep=0, kp=0)
# currentPosition = Position(board=current_board, score=0, wc=[True, True], bc=[True, True], ep=0, kp=0)
# print(oldPosition)
# print(currentPosition)


#inputs:
#   old: the previous state of the board. i.e. the board right before the move was played 
# a list of lists, with numbers to represent pieces. The starting position should look like this:
# old = [
#     [7, 11, 8, 9, 10, 8, 11, 7],
#     [6, 6, 6, 6, 6, 6, 6, 6],
#     [-1, -1, -1, -1, -1, -1, -1, -1],
#     [-1, -1, -1, -1, -1, -1, -1, -1],
#     [-1, -1, -1, -1, -1, -1, -1, -1],
#     [-1, -1, -1, -1, -1, -1, -1, -1],
#     [1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 3, 2, 5, 4, 2, 3, 1],
# ]

#  new: the current state of the board. i.e. a move has just been played

def isOneLegalMoveAway(old, new):
    old_board = getBoard(old)
    current_board = getBoard(new)
    
    oldPosition = Position(board=old_board, score=0, wc=[True, True], bc=[True, True], ep=0, kp=0)
    currentPosition = Position(board=current_board, score=0, wc=[True, True], bc=[True, True], ep=0, kp=0)
        
    moveIsLegal = False
    # A legal move can be made by EITHER player and this function will still call it legal
    # regardless of whose turn it actually is
    # This means that we don't have to store and pass around whose turn it is to play
    for m in oldPosition.gen_moves():
        legalPosition = oldPosition.move(m)
        if currentPosition == legalPosition:
            moveIsLegal = True
    oldPosition.rotate() # switch whose turn it is
    for m in oldPosition.gen_moves():
        legalPosition = oldPosition.move(m)
        if currentPosition == legalPosition:
            moveIsLegal = True
    
    return moveIsLegal
