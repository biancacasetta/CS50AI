"""
Tic Tac Toe Player
"""

import copy
import math

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    X_quantity = sum(row.count(X) for row in board)
    O_quantity = sum(row.count(O) for row in board)    

    if X_quantity <= O_quantity:
        return X
    else:
        return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    possible_actions = set()
    for row_index, row in enumerate(board):
        for cell_index, cell in enumerate(row):
            if cell == EMPTY:
                possible_actions.add((row_index, cell_index))
    
    return possible_actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    if action not in actions(board):
        raise Exception("Invalid action: cell is not empty")

    new_board = copy.deepcopy(board)
    new_board[action[0]][action[1]] = player(board)

    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    winner = None

    winning_positions = [[(0, 0), (0, 1), (0, 2)],
                         [(1, 0), (1, 1), (1, 2)],
                         [(2, 0), (2, 1), (2, 2)],
                         [(0, 0), (1, 0), (2, 0)],
                         [(0, 1), (1, 1), (2, 1)],
                         [(0, 2), (1, 2), (2, 2)],
                         [(0, 0), (1, 1), (2, 2)],
                         [(0, 2), (1, 1), (2, 0)]]
    
    for wp in winning_positions:
        cell1 = board[wp[0][0]][wp[0][1]]
        cell2 = board[wp[1][0]][wp[1][1]]
        cell3 = board[wp[2][0]][wp[2][1]]

        if cell1 is not None and cell1 == cell2 == cell3:
            winner = cell1
            break

    return winner


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) is not None or not any(EMPTY in row for row in board):
        return True
    else:
        return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None

    possible_actions = actions(board)
    current_player = player(board)

    if current_player is X:
        max = max_value(board)

        for action in possible_actions:
            if max == min_value(result(board, action)):
                return action
    else:
        min = min_value(board)

        for action in possible_actions:
            if min == max_value(result(board, action)):
                return action        


def max_value(state):
    max_value = -math.inf
    if terminal(state):
        return utility(state)
    
    for action in actions(state):
        max_value = max(max_value, min_value(result(state, action)))

    return max_value


def min_value(state):
    min_value = math.inf
    if terminal(state):
        return utility(state)
    
    for action in actions(state):
        min_value = min(min_value, max_value(result(state, action)))

    return min_value