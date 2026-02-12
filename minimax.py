# minimax.py
from __future__ import annotations
from typing import List, Tuple, Dict, Optional

# board_abs: X=+1, O=-1, vide=0

WINS = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
]

def check_winner_abs(board: List[int]) -> int:
    for a, b, c in WINS:
        s = board[a] + board[b] + board[c]
        if s == 3:
            return 1
        if s == -3:
            return -1
    return 0

def is_full(board: List[int]) -> bool:
    return all(v != 0 for v in board)

def available_moves(board: List[int]) -> List[int]:
    return [i for i, v in enumerate(board) if v == 0]

def _minimax_value(board: Tuple[int, ...], player: int, memo: Dict[Tuple[Tuple[int, ...], int], int]) -> int:
    """
    Retourne la valeur du board du point de vue de 'player' (le joueur qui doit jouer maintenant),
    avec score: +1 = victoire de 'player', -1 = défaite, 0 = nul.
    """
    key = (board, player)
    if key in memo:
        return memo[key]

    b = list(board)
    winner = check_winner_abs(b)
    if winner != 0:
        # si winner == player => +1, sinon -1
        memo[key] = 1 if winner == player else -1
        return memo[key]
    if is_full(b):
        memo[key] = 0
        return 0

    best = -2  # plus petit que -1
    for mv in available_moves(b):
        b2 = b[:]
        b2[mv] = player
        # tour adverse => valeur pour player = - valeur pour l'adversaire
        val = -_minimax_value(tuple(b2), -player, memo)
        if val > best:
            best = val
            if best == 1:
                break  # on ne peut pas faire mieux
    memo[key] = best
    return best

def minimax_best_move(board_abs: List[int], player: int) -> int:
    """
    Renvoie le meilleur coup (optimal) pour 'player' (+1 pour X, -1 pour O).
    """
    memo: Dict[Tuple[Tuple[int, ...], int], int] = {}
    best_move = None
    best_val = -2
    for mv in available_moves(board_abs):
        b2 = board_abs[:]
        b2[mv] = player
        val = -_minimax_value(tuple(b2), -player, memo)
        if val > best_val:
            best_val = val
            best_move = mv
            if best_val == 1:
                break
    if best_move is None:
        raise ValueError("Aucun coup possible")
    return best_move
