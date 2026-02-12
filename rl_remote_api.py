# rl_remote_api.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn

from rl import QLearningAgent

app = FastAPI()

agent = QLearningAgent(qtable_path="qtable.pkl")  # charge ton modèle
# optionnel: s'assurer qu'il n'explore jamais côté API
agent.epsilon = 0.0

class MoveReq(BaseModel):
    board: List[str]   # ["X","O"," ",...]
    you_are: str       # "X" or "O"

def board_to_abs(board: List[str]) -> list[int]:
    out = []
    for v in board:
        if v == "X":
            out.append(1)
        elif v == "O":
            out.append(-1)
        else:
            out.append(0)
    return out

def abs_to_state(board_abs: list[int], player_abs: int):
    """
    Même représentation que ton projet principal :
    - player_abs = +1 (X) ou -1 (O)
    - on transforme en perspective joueur courant: board * player_abs
    Ici on suppose que ton rl.py expose déjà abs_to_state,
    sinon on peut recoder la fonction.
    """
    from rl import abs_to_state as _abs_to_state
    return _abs_to_state(board_abs, player_abs)

def legal_moves_abs(board_abs: list[int]) -> list[int]:
    return [i for i, v in enumerate(board_abs) if v == 0]

@app.post("/move")
def move(req: MoveReq):
    board_abs = board_to_abs(req.board)
    player_abs = 1 if req.you_are == "X" else -1

    moves = legal_moves_abs(board_abs)
    if not moves:
        return {"idx": 0}

    s = abs_to_state(board_abs, player_abs)
    idx = agent.choose_action(s, epsilon_override=0.0)

    # sécurité si jamais
    if idx not in moves:
        idx = moves[0]

    return {"idx": idx}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9100)
