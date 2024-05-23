import chess
import chess.svg
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cairosvg
import gurobipy as gp
import os
import shutil
from gurobipy import GRB
from io import BytesIO
from numpy import sign
from sys import argv
from time import time
from itertools import combinations

DIRECTORY = 'imgs_{}'.format(int(time()))

if os.path.exists(DIRECTORY):
    shutil.rmtree(DIRECTORY)
os.makedirs(DIRECTORY)

gp.setParam("MIPFocus", 2)  # Go for optimality
gp.setParam("Presolve", 2)  # Takes an hour with pin constraints otherwise
# gp.setParam("Cuts", 2)  # Takes an hour with pin constraints otherwise
# gp.setParam("ZeroHalfCuts", 2) # Overwrites Cuts parameter
# gp.setParam("Heuristics", 0)
gp.setParam('Nodefilestart', 0.1)  # Start caching disk to tree in order to not run out of memory
gp.setParam('PoolSolutions', 2_000_000_000)  # No reason to limit this
gp.setParam('PoolSearchMode', 2)  # Systematically search the entire solutions space
gp.setParam('PoolGap', 0.0)  # Only retain optimal solutions
# gp.setParam('Cutoff', 218)  # Enter current record here to make things faster

MOVES_MUST_BE_CAPTURES = False
ALLOW_PROMOTIONS = True  # Whether you can have more than 1 queen, 2 knights, ...
VALUE_PROMOTIONS_FOUR_TIMES = True

# Activate/Deactivate constraints
KINGS_NOT_IN_CHECK = ["white", "black"]  # These kings may not be attacked
ENABLE_KING_WALK_INTO_FIRE = True  # White king may not step onto square that is attacked
ENABLE_ROYAL_CUDDLING = True  # White king may not stand next to black king
ENABLE_MOVE_LEGALITY = True  # Basic move legality, no reason to disable this
ENABLE_NO_MOVING_THROUGH = True  # Can piece move if other pieces stand in the way
ENABLE_PINS = False  # Very expensive and slow, better to filter these out later on
ENABLE_NO_GROUP_CAMPING = True  # at most one piece at a square, no reason to disable this
ENFORCE_ANY_COUNTS = True  # enforce piece counts, otherwise solver has complete freedom except for kings
ENABLE_MIP_START = False  # give current record to solver, increases speed
REINFORCE_RELAXATION = True  # redundant cuts which make solving much faster
PIECES_DEACTIVATE = []  # pieces not allowed to use. Use to make the program finish sooner to test things

# To check how the relaxation looks
VTYPE = GRB.CONTINUOUS if False else GRB.INTEGER
LOW_LIMIT = 1e-4

PIECES = ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king']
COLORS = ["white", "black"]

fens = set()
# Graphically display the bord or save it to file. Add real score. Collects position with correct score
def plot_chess_board(pieces, moves=[], filename=None, score=None):

    global fens

    def coordinate_to_square(row, col):
        return chess.square(col, row)

    board = chess.Board(None)  # Empty board

    # Place pieces on the board
    for piece, pos in pieces:
        square = coordinate_to_square(*pos)
        board.set_piece_at(square, chess.Piece.from_symbol(piece))

    # Determine real score
    real_move_count = 0
    for move in board.legal_moves:        
        if not MOVES_MUST_BE_CAPTURES or board.is_capture(move):
            if not move.promotion or VALUE_PROMOTIONS_FOUR_TIMES:
                real_move_count += 1
            else:
                real_move_count += 0.25  # each promotion occurs 4 times but should only be counted once
    real_move_count = round(real_move_count)
    
    # Convert input moves to python-chess moves
    gurobi_moves = []
    for r, c, R, C, piece in moves:
        from_square = coordinate_to_square(r, c)
        to_square = coordinate_to_square(R, C)
        if (piece == "pawn" and r == 6):
            for promotion in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
                move = chess.Move(from_square, to_square, promotion=promotion)
                gurobi_moves.append(move.uci())
        else:
            move = chess.Move(from_square, to_square, promotion=None)
            gurobi_moves.append(move.uci())
            

    real_moves = [m.uci() for m in board.legal_moves]

    # Determine how gurobi's solution differs from the actual moves
    moves_illegal = [move for move in gurobi_moves if move not in real_moves]
    moves_missed = [move for move in real_moves if move not in gurobi_moves]

    for m in moves_illegal + moves_missed:
        assert m not in moves_illegal or m not in moves_missed

    # Adapt filename to indicate correctness of gurobi's score and real score.
    if score:
        assert isinstance(score, int)
        if real_move_count != score:
            filename = '{}_wrong_{}_but_{}.png'.format(filename, score, real_move_count)
            # print('Position has wrong move count {} but should have {}'.format(score, real_move_count))
            # return  # Do not save wrong positions
        else:
            filename = '{}_correct_{}.png'.format(filename, real_move_count)

        if real_move_count == score:
            fens.add(board.fen())

    # Write result to file if requested
    if filename:
        tf = open(filename+'.txt', 'w')

        if len(moves) > 0:
            tf.write('{}\n'.format(board.fen()))
            tf.write('{} gurobi and {} real moves\n'.format(len(gurobi_moves), len(real_moves)))
            tf.write('Illegal moves:\n')
            for m in moves_illegal:
                tf.write('    {}\n'.format(m))

            tf.write('Missed moves:\n')
            for m in moves_missed:
                tf.write('    {}\n'.format(m))

        tf.close()

    # Create an SVG image of the board
    svg = chess.svg.board(board, arrows=[], squares=[])

    # Save it file if filename is not None
    if filename:
        # Save the SVG to a file directly
        with open(filename, "w") as f:
            f.write(svg)
    else:
        # Convert SVG to a format that matplotlib can handle
        png_output = BytesIO()
        cairosvg.svg2png(bytestring=svg.encode('utf-8'), write_to=png_output)
        png_output.seek(0)  # Rewind the file pointer to the start
        image = mpimg.imread(png_output, format='PNG')
        plt.imshow(image)
        plt.axis('off')  # Hide the axes
        plt.show()

callback_count = 0
def mycallback(model, where):
    global callback_count
    if where == GRB.Callback.MIPSOL:
        current_obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)  # Get the best objective value found so far
        if model._best_obj is None or current_obj >= model._best_obj:
            model._best_obj = current_obj  # Update the best known objective value
            pieces_for_plotting = []
            chessboard = [["-" for _ in range(8)] for _ in range(8)]
            maps = {"pawn": "p", "knight": "n", "bishop": "b", "rook": "r", "queen": "q", "king": "k"}

            for k, v in model._p.items():
                if model.cbGetSolution(v) > 0.5:
                    row, col, color, piece = k
                    letter = maps[piece]
                    if color == "white":
                        letter = letter.upper()
                    chessboard[row][col] = letter
                    pieces_for_plotting.append((letter, (row, col)))

            moves = []
            for k, v in model._m.items():
                if model.cbGetSolution(v) > 0.5:
                    moves.append(k)

            if current_obj > 0:  # For some reason Gurobi is weird like this
                callback_count += 1  # started at 0
                fn = os.path.join(DIRECTORY, "{}".format(callback_count, int(current_obj)))
                plot_chess_board(pieces_for_plotting, moves=moves, filename=fn, score=round(current_obj))

# Make sure coordinates are within chess board
def ok(r, c):
    return 0 <= r <= 7 and 0 <= c <= 7

# For iterating through all squares
def squares():
    for r in range(8):
        for c in range(8):
            yield r, c

# Generate all moves for this piece at this position which do not leave the board
def moves(r, c, piece):
    assert ok(r, c)

    if piece == "rook":
        return rook_moves(r, c)
    elif piece == "bishop":
        return bishop_moves(r, c)
    elif piece == "queen":
        return rook_moves(r, c) + bishop_moves(r, c)
    elif piece == "knight":
        return knight_moves(r, c)
    elif piece == "king":
        return king_moves(r, c)

    assert False

def rook_moves(r, c):
    ms = []
    for s in range(1, 9):
        for rs, cs in [(-s, 0), (0, -s), (s, 0), (0, s)]:
            if ok(r+rs, c+cs):
                ms.append((rs, cs))                
    return ms

def bishop_moves(r, c):
    ms = []
    for s in range(1, 9):
        for rs, cs in [(-s, -s), (-s, s), (s, -s), (s, s)]:
            if ok(r+rs, c+cs):
                ms.append((rs, cs))
    return ms

def knight_moves(r, c):
    ms = []
    for rs in [-2, -1, 1, 2]:
        for cs in [-2, -1, 1, 2]:
            if abs(rs) != abs(cs):
                if ok(r+rs, c+cs):
                    ms.append((rs, cs))
    return ms

def king_moves(r, c):
    ms = []
    for rs in [-1, 0, 1]:
        for cs in [-1, 0, 1]:
            if (rs != 0 or cs != 0) and ok(r+rs, c+cs):
                ms.append((rs, cs))
    if (r, c) in [(0, 4)]:  # e1, not caring about black castling
        ms += [(r, c-2), (r, c+2)]
    return ms

# Is the move from this to that square a move of the given piece (white)?
# Only care about white moves. Black moves only matter for attacks and there is a seperate function for that
def is_move(r, c, R, C, piece):
    assert ok(r, c)
    assert ok(R, C)

    if piece == "pawn":
        return r not in [0, 7] and abs(C-c) <= 1 and R-r == 1 or C==c and r==1 and R==3
    elif piece == "knight":
        return (abs(r-R), abs(c-C)) in [(1, 2), (2,1)]
    elif piece == "bishop":
        return abs(r-R) == abs(c-C)
    elif piece == "rook":
        return (r == R) != (c == C)
    elif piece == "queen":
        return is_move(r, c, R, C, "bishop") or is_move(r, c, R, C, "rook")
    elif piece == "king":
        return max(abs(R-r), abs(C-c)) == 1 or (r, c) in [(0, 4)] and (R, C) in [(r, c-2), (r, c+2)]
    else:
        assert False

# Does the given move represent an attack from the given piece on (r, c) on the square (R, C)?
def is_attacking(r, c, R, C, piece, color="white"):
    assert ok(r, c)
    assert ok(R, C)

    if piece == "pawn":
        return r not in [0, 7] and abs(C-c) == 1 and R-r == (1 if color == "white" else -1)
    elif piece == "knight":
        return (abs(r-R), abs(c-C)) in [(1, 2), (2,1)]
    elif piece == "bishop":
        return abs(r-R) == abs(c-C)
    elif piece == "rook":
        return (r == R) != (c == C)
    elif piece == "queen":
        return is_move(r, c, R, C, "bishop") or is_move(r, c, R, C, "rook")
    elif piece == "king":
        return max(abs(R-r), abs(C-c)) == 1
    else:
        assert False

# Generate a list of all squares inbetween but not including (r, c) and (R, C).
# Inbetween = like bishop/rook
def inbetween(r, c, R, C):
    assert ok(r, c)
    assert ok(R, C)

    sgn_r = sign(R-r)
    sgn_c = sign(C-c)
    inb = []
    if is_move(r, c, R, C, "rook"):
        if r == R:
            return [(r, x) for x in range(c, C+sgn_c, sgn_c)][1:-1]
        else:
            return [(x, c) for x in range(r, R+sgn_r, sgn_r)][1:-1]
    elif is_move(r, c, R, C, "bishop"):
        return [(r+s*sgn_r, c+s*sgn_c) for s in range(1, abs(R-r))]
    elif is_move(r, c, R, C, "knight") or is_move(r, c, R, C, "king"):
        return []  # Nothing inbetween for knight and king
    else:
        assert false


def create_and_solve():
    model = gp.Model("chess")
    obj = 0

    print('Building model... this might take a few minutes')

    # variable m[r, c, R, C, piece] is true if and only if that white piece can move from (r, c) to (R, C)
    print("Creating move variables")
    m = dict()
    for r, c in squares():
        for R, C in squares():
            if (r, c) == (R, C):
                continue
            for piece in PIECES:
                white_promotion = (r == 6 and R == 7 and piece == "pawn" and abs(c-C) <= 1)
                white_pawn_move = r not in [0, 6, 7] and is_move(r, c, R, C, "pawn") and piece == "pawn"
                knight = is_move(r, c, R, C, "knight") and piece == "knight"
                rook_or_queen = is_move(r, c, R, C, "rook") and piece in ["rook", "queen"]
                bishop_or_queen = is_move(r, c, R, C, "bishop") and piece in ["bishop", "queen"]
                king = is_move(r, c, R, C, "king") and piece == "king"

                if white_promotion or white_pawn_move or knight or rook_or_queen or bishop_or_queen or king:
                    v = model.addVar(vtype=VTYPE, lb=0, ub=1, name="m_{}_{}_{}_{}_{}".format(r, c, R, C, piece))
                    m[r, c, R, C, piece] = v
                    obj += v * (1 if not white_promotion or not VALUE_PROMOTIONS_FOUR_TIMES else 4)  # promotion is four different moves

    # 218 moves with promotion
    RECORD_218 = [
        (0, 1, "white", "king"),
        (0, 3, "white", "bishop"),
        (0, 4, "white", "bishop"),
        (0, 5, "white" ,"knight"),
        (0, 6, "white" ,"knight"),
        (0, 7, "black" ,"king"),
        (1, 1, "white", "queen"),
        (1, 6, "white", "rook"),
        (1, 7, "black", "pawn"),
        (2, 3, "white", "queen"),
        (3, 0, "white", "queen"),
        (3, 5, "white", "queen"),
        (4, 2, "white", "queen"),
        (4, 7, "white", "rook"),
        (5, 4, "white", "queen"),
        (6, 1, "white", "queen"),
        (6, 6, "white", "queen"),
        (7, 3, "white", "queen")
    ]

    # 144 moves without promotions
    RECORD_144 = [
        (0, 2, "white", "rook"),
        (1, 3, "white", "pawn"),
        (1, 4, "white", "bishop"),
        (1, 5, "white", "pawn"),
        (1, 6, "black", "king"),
        (1, 7, "white", "pawn"),
        (2, 4, "black", "pawn"),
        (2, 6, "black", "pawn"),
        (3, 0, "white", "rook"),
        (4, 3, "white", "knight"),
        (4, 4, "white", "bishop"),
        (4, 5, "white", "knight"),
        (4, 6, "white", "king"),
        (5, 1, "white", "queen"),
        (6, 1, "white", "pawn"),
        (6, 3, "white", "pawn"),
        (6, 5, "white", "pawn"),
        (6, 7, "white", "pawn"),
        (7, 0, "black", "knight"),
        (7, 2, "black", "rook"),
        (7, 4, "black", "rook"),
        (7, 6, "black", "bishop")
    ]

    # 74 captures (with or without promotions?)
    RECORD_74 = [
        (0, 4, "white", "bishop"),
        (0, 5, "black", "queen"),
        (1, 0, "black", "king"),
        (2, 3, "white", "bishop"),
        (2, 4, "white", "knight"),
        (2, 5, "white", "queen"),
        (2, 6, "black", "bishop"),
        (2, 7, "white", "queen"),
        (3, 2, "black", "pawn"),
        (3, 3, "white", "king"),
        (3, 4, "black", "pawn"),
        (4, 2, "white", "rook"),
        (4, 3, "black", "knight"),
        (4, 4, "white", "queen"),
        (4, 5, "black", "rook"),
        (4, 6, "white", "rook"),
        (4, 7, "black", "queen"),
        (5, 1, "white", "knight"),
        (5, 3, "white", "knight"),
        (5, 5, "white", "knight"),
        (6, 1, "white", "pawn"),
        (6, 3, "white", "pawn"),
        (6, 5, "white", "pawn"),
        (6, 7, "white", "pawn"),
        (7, 0, "black", "rook"),
        (7, 2, "black", "knight"),
        (7, 4, "black", "knight"),
        (7, 6, "black", "bishop")
    ]

    # p[r, c, color, piece] = 1 if and only if there is a piece of said color on (r, c)
    print("Creating piece variables")
    p = dict()
    for r, c in squares():
        for color in COLORS:
            for piece in PIECES:
                v = model.addVar(vtype=VTYPE, lb=0, ub=1, name="p_{}_{}_{}_{}".format(r, c, color, piece))
                p[r, c, color, piece] = v
                if piece == "pawn" and r in [0, 7]:
                    model.addConstr(v == 0, 'deactivate_piece_{}_{}_{}_{}'.format(r, c, color, piece))

                if ENABLE_MIP_START:
                    if (r, c, color, piece) in RECORD_74:
                        v.Start = 1
                        # model.addConstr(v == 1, 'recursive_{}_{}_{}_{}'.format(r, c, color, piece))
                    else:
                        v.Start = 0
                        # model.addConstr(v == 0, 'recursive_{}_{}_{}_{}'.format(r, c, color, piece))

    # Neither king should be in check
    for color in KINGS_NOT_IN_CHECK:
        assert color in ["white", "black"]
        print("{} king shall not be in check".format(color))
        other_color = 'black' if color == 'white' else 'white'
        
        for r, c in squares():
            var_king = p[r, c, color, "king"]

            # Ban pawns that give check
            rs = -1 if color == 'black' else 1
            for cs in [-1, 1]:
                if ok(r+rs, c+cs):
                    var_pawn = p[r+rs, c+cs, other_color, "pawn"]
                    model.addConstr(var_king + var_pawn <= 1, '{}_king_on_{}_{}_not_in_check_by_pawn_on_{}_{}'.format(color, r, c, r+rs, c+cs))

            # Ban other king standing next
            if color != "black" or "white" not in KINGS_NOT_IN_CHECK:  # Do this for black if not already done for white. Otherwise don't do it twice.
                for rs in [-1, 0, 1]:
                    for cs in [-1, 0, 1]:
                        if ok(r+rs, c+cs) and (rs != 0 or cs != 0):
                            var_other_king = p[r+rs, c+cs, other_color, "king"]
                            model.addConstr(var_king + var_other_king <= 1, '{}_king_on_{}_{}_not_next_to_king_on_{}_{}'.format(color, r, c, r+rs, c+cs))

            # Ban knights that give check
            for rs, cs in moves(r, c, "knight"):
                var_knight = p[r+rs, c+cs, other_color, "knight"]
                model.addConstr(var_king + var_knight <= 1, '{}_king_on_{}_{}_not_in_check_by_knight_on_{}_{}'.format(color, r, c, r+rs, c+cs))

            straights = [(rs, cs, pie) for rs, cs in moves(r, c, "rook") for pie in ["rook", "queen"]]
            straights += [(rs, cs, pie) for rs, cs in moves(r, c, "bishop") for pie in ["bishop", "queen"]]

            # Ban straight pieces
            for rs, cs, pie in straights:
                var_pie = p[r+rs, c+cs, other_color, pie]
                vars_empty = [p[ri, ci, col, piece] for ri, ci in inbetween(r, c, r+rs, c+cs) for piece in PIECES for col in COLORS]
                su = var_pie + var_king
                for v in vars_empty:
                    su += 1 - v
                model.addConstr(su <= len(vars_empty) + 1, '{}_king_on_{}_{}_not_in_check_by_{}_on_{}_{}'.format(color, r, c, pie, r+rs, c+cs))  # At least one of those has to be false

    if ENABLE_MOVE_LEGALITY:
        print("Moves shall be kinda legal")
        # Enforce basic legality (without considering pins, exact castling, en passant etc.)
        for r, c, R, C, piece in m:
            var_move_possible = m[r, c, R, C, piece]
            var_there = p[r, c, "white", piece]

            model.addConstr(var_there >= var_move_possible, 'no_move_{}_{}_to_{}_{}_without_{}'.format(r, c, R, C, piece))  # move is only possible if a piece is there
            
            if MOVES_MUST_BE_CAPTURES:
                # If move could be capture, only allow with target
                if is_attacking(r, c, R, C, piece):
                    vars_targets = [p[R, C, "black", piece_target]  for piece_target in ["pawn", "knight", "bishop", "rook", "queen"]]
                    model.addConstr(var_move_possible <= sum(vars_targets))  # Can't be capture if there's nothing to capture
                else:  # If move can never be a capture (pawn move) ban it entirely
                    model.addConstr(var_move_possible == 0, 'ban_capture_moves_that_cant_be_captures_{}_{}_{}_{}_{}'.format(r, c, R, C, piece))

            if ENABLE_NO_MOVING_THROUGH:
                if piece == "pawn":
                    assert is_move(r, c, R, C, "pawn")
                    if r == 4 and R == 5 and abs(c-C) == 1: # TODO finish this block
                        vars_targets = [p[R, C, "black", piece_target] for piece_target in ["pawn", "knight", "bishop", "rook", "queen"]]
                        var_neighbor_pawn = p[r, C, "black", "pawn"]
                        # capture only possible if at least one of the targets is present or neighbor pawn for en passant present
                        model.addConstr(var_move_possible <= sum(vars_targets) + var_neighbor_pawn, 'no_non_en_passant_or_en_passant_pawn_capture_from_{}_{}_without_target_on_{}_{}'.format(r, c, R, C))
                        pass  # possible en passant capture, just allow it so I don't have to check it :D (relaxed en passant)
                    elif abs(c-C) == 1:  # capture
                        vars_targets = [p[R, C, "black", piece_target] for piece_target in ["pawn", "knight", "bishop", "rook", "queen"]]
                        # capture only possible if at least one of the targets is present
                        model.addConstr(var_move_possible <= sum(vars_targets), 'no_non_en_passant_pawn_capture_from_{}_{}_without_target_on_{}_{}'.format(r, c, R, C))
                    elif c == C:  # not a capture
                        vars_blocking = [(p[R, C, color, piece_target], R, C, piece_target, color) for piece_target in PIECES for color in COLORS]
                        if R-r == 2:  # double move, also consider blockading pieces on second square
                            vars_blocking += [(p[R-1, C, color, piece_target], R-1, C, piece_target, color) for piece_target in PIECES for color in COLORS]
                        # There can only be one (at most). Either move is possible, then there is no blockading piece. Or there is a piece and thus the move is not valid.
                        for v, rb, cb, piece_target, color in vars_blocking:
                            model.addConstr(var_move_possible + v <= 1, 'no_pawn_move_{}_{}_to_{}_{}_if_{}_{}_on_{}_{}'.format(r, c, R, C, color, piece_target, rb, cb))
                    else:
                        assert False
                elif piece in ["knight", "bishop", "rook", "queen", "king"]:
                    assert is_move(r, c, R, C, piece)
                    vars_blocking = [p[R, C, "white", piece_target] for piece_target in PIECES]
                    vars_inbetween = [p[ri, ci, color, piece_target] for ri, ci in inbetween(r, c, R, C) for piece_target in PIECES for color in COLORS]
                    for v in vars_blocking + vars_inbetween:
                        model.addConstr(var_move_possible + v <= 1)  # Either move is legal or there is something in the way. (Or neither of the two)
                    if piece == "king" and r == 0 and c == 4:  # king still on e1, having castling options
                        if R == 0 and C == 6:  # wants to castle kingside
                            model.addConstr(var_move_possible <= p[0, 7, "white", "rook"], 'no_castling_without_kingside_rook')  # Don't castle without a rook being present
                        elif R == 0 and C == 2:  # wants to castle queenside
                            model.addConstr(var_move_possible <= p[0, 0, "white", "rook"], 'no_castling_without_queenside_rook')
                else:
                    assert False

    if ENABLE_KING_WALK_INTO_FIRE:
        print("King may not walk into fire")
        attackers = ["pawn", "knight", "bishop", "rook", "queen", "king"]
    elif ENABLE_ROYAL_CUDDLING:
        print("King shall not cuddle royally")
        attackers = ["king"]
    else:
        attackers = []

    # Do not walk the king onto endangered squares
    for r, c, R, C, piece in m:
        if piece != "king":
            continue
        var_king_move = m[r, c, R, C, "king"]
        # For all attackers
        for ra, ca in squares():
            if (r, c) == (ra, ca):
                continue
            for piece_attacker in attackers:
                if ra in [0, 7] and piece_attacker == "pawn" or not is_attacking(ra, ca, R, C, piece_attacker, color="black"):
                    continue
                var_attacker_there = p[ra, ca, "black", piece_attacker]
                vars_inbetween = [p[ri, ci, color, piece_target] for ri, ci in inbetween(ra, ca, R, C) for piece_target in PIECES for color in COLORS]
                model.addConstr(var_king_move + var_attacker_there + sum([(1 - v) for v in vars_inbetween]) <= len(vars_inbetween) + 1, 'no_walk_into_fire')  # Can't be all true at the same time

    if ENABLE_PINS:
        print("You may not abandon thy king")
        # Do not move pieces out of pins (not including horizontally removing two pawns via en passant)
        for r, c, R, C, piece in m:
            if piece == "king":
                continue
            print(r, c, R, C, piece, end=' '*50+'\r')
            var_move = m[r, c, R, C, piece]
            # for all king positions
            for rk, ck in squares():
                if (rk, ck) in [(r, c), (R, C)]:
                    continue
                var_king_there = p[rk, ck, "white", "king"]
                # For all attackers
                for ra, ca in squares():
                    if (ra, ca) in [(r, c), (R, C), (rk, ck)]:
                        continue
                    for piece_attacker in ["pawn", "knight", "bishop", "rook", "queen", "king"]:
                        if ra in [0, 7] and piece_attacker == "pawn" or not is_attacking(ra, ca, R, C, piece_attacker):
                            continue

                        # Are we actually leaving the pin?
                        if (R, C) in inbetween(ra, ca, R, C):
                            continue

                        # Are we the only one inbetween?
                        var_attacker_there = p[ra, ca, "black", piece_attacker]
                        vars_inbetween = [p[ri, ci, color, piece_target] for ri, ci in inbetween(ra, ca, R, C) for piece_target in PIECES for color in COLORS if (ri, ci) != (r, c)]                    
                        constr = model.addConstr(1 - var_move + 1 - var_king_there + 1 - var_attacker_there + sum(vars_inbetween) >= 1, 'do_not_leave_pin')  # At least one part has to be wrong
                        constr.Lazy = 1  # don't check pin constraints all the time
        
        print()

    if ENABLE_NO_GROUP_CAMPING:
        print("No group camping on squares")
        # At most one piece per square
        for r, c in squares():
            pieces_of_that_square = [p[r, c, color, piece] for color in COLORS for piece in PIECES]
            model.addConstr(sum(pieces_of_that_square) <= 1, 'no_group_camping')

    print("Keep count")
    # Enforcing the number of each type of piece
    for color in ['white', 'black']:
        pawns = [p[r, c, color, 'pawn'] for r, c in squares()]
        knights = [p[r, c, color, 'knight'] for r, c in squares()]
        bishops = [p[r, c, color, 'bishop'] for r, c in squares()]
        rooks = [p[r, c, color, 'rook'] for r, c in squares()]
        queens = [p[r, c, color, 'queen'] for r, c in squares()]
        kings = [p[r, c, color, 'king'] for r, c in squares()]
        everybody = pawns+knights+bishops+rooks+queens+kings

        var_count_prom_knights = model.addVar(vtype=VTYPE, lb=0, ub=8, name="var_count_prom_{}_{}s".format(color, "knight"))
        var_count_prom_bishops = model.addVar(vtype=VTYPE, lb=0, ub=8, name="var_count_prom_{}_{}s".format(color, "bishop"))
        var_count_prom_rooks = model.addVar(vtype=VTYPE, lb=0, ub=8, name="var_count_prom_{}_{}s".format(color, "rook"))
        var_count_prom_queens = model.addVar(vtype=VTYPE, lb=0, ub=8, name="var_count_prom_{}_{}s".format(color, "queen"))

        if ENFORCE_ANY_COUNTS:
            model.addConstr(var_count_prom_knights >= sum(knights) - 2, 'promoted_knights_count_{}'.format(color))
            model.addConstr(var_count_prom_bishops >= sum(bishops) - 2, 'promoted_bishops_count_{}'.format(color))
            model.addConstr(var_count_prom_rooks >= sum(rooks) - 2, 'promoted_rooks_count_{}'.format(color))
            model.addConstr(var_count_prom_queens >= sum(queens) - 1, 'promoted_queens_count_{}'.format(color))

            sum_counts_promoted = var_count_prom_knights + var_count_prom_bishops + var_count_prom_rooks + var_count_prom_queens

            if ALLOW_PROMOTIONS:
                model.addConstr(sum_counts_promoted + sum(pawns) <= 8, 'pawns_and_promoted_atmost_8_{}'.format(color))
            else:
                model.addConstr(sum_counts_promoted == 0, 'no_promotion_counts_{}'.format(color))
                for sq in [1, 0]:
                    vars_bishops = [p[r, c, color, "bishop"] for r, c in squares() if (r+c)%2==sq]
                    model.addConstr(sum(vars_bishops) <= 1, 'at_most_one_{}_bishop_on_{}_squares'.format(color, 'light' if sq==1 else 'dark'))

            if color in COLORS:
                model.addConstr(sum(pawns) <= 8)
                model.addConstr(sum(everybody) <= 16)
                # Rest enforced via promotion?

        model.addConstr(sum(kings) == 1)

    if REINFORCE_RELAXATION:
        print("Reinforcing relaxation, only one white piece in total moving from one direction")
        # Reinforce relaxation by preventing multiple partial pieces from moving onto the same square from the same direction
        for R, C in squares():  # target square
            # The four rook lines and the four bishop diagonals
            for sig_r in [-1, 0, 1]:
                for sig_c in [-1, 0, 1]:
                    if sig_r == 0 and sig_c == 0:
                        continue
                    non_queen = "bishop" if abs(sig_r)==abs(sig_c) else "rook"
                    su = 0
                    r, c = R+sig_r, C+sig_c
                    while ok(r, c):
                        su += m[r, c, R, C, non_queen] + m[r, c, R, C, "queen"]
                        if is_move(r, c, R, C, "king"):
                            su += m[r, c, R, C, "king"]
                        if r not in [0, 7] and is_move(r, c, R, C, "pawn"):
                            su += m[r, c, R, C, "pawn"]
                        r += sig_r
                        c += sig_c
                    model.addConstr(su <= 1, 'one_attacker_from_one_direction_only')

    for piece in PIECES_DEACTIVATE:
        for r, c, color, piece2 in p:
            if piece == piece2:
                model.addConstr(p[r, c, color, piece2] == 0)

    model.setObjective(obj, GRB.MAXIMIZE)

    print("Writing model to file")
    model.write("chess.lp")

    print("Here we go")
    model._best_obj = None
    model._p = p
    model._m = m
    model.optimize(mycallback)

    print("Let's see what happened...'")
    if model.status == GRB.INFEASIBLE:
        return

    count_true = 0
    total = 0

    for k, v in m.items():
        if v.X > LOW_LIMIT:
            count_true += v.X
        total += 1

    print("Model: {} of {} were true".format(count_true, total))
            
    for k, v in m.items():
        if v.X > LOW_LIMIT:
            print('m {} {}'.format(k, v.X))
    print('-' * 50)
    for k, v in p.items():
        if v.X > LOW_LIMIT:
            print('p {} {}'.format(k, v.X))

    chessboard = [["-" for _ in range(8)] for _ in range(8)]

    maps = {
        "pawn": "p",
        "knight": "n",
        "bishop": "b",
        "rook": "r",
        "queen": "q",
        "king": "k"
    }

    pieces_for_plotting = []

    for k, v in p.items():
        if v.X > LOW_LIMIT:
            row, col, color, piece = k
            letter = maps[piece]
            if color == "white":
                letter = letter.upper()
            chessboard[row][col] = letter
            pieces_for_plotting.append((letter, (row, col)))

    # Print the chessboard
    for row in chessboard[::-1]:
        print(' '.join(row))

    print('Writing found solutions to fen.txt')

    with open(os.path.join(DIRECTORY, 'fen.txt'), 'w') as f:
        for fe in sorted(fens):
            f.write(fe+'\n')

    # fn = os.path.join(DIRECTORY, "{}.png".format('final.png'))
    # plot_chess_board(pieces_for_plotting, filename=fn)
    plot_chess_board(pieces_for_plotting)

    print('Score {}'.format(count_true))


if __name__ == "__main__":
    create_and_solve()
