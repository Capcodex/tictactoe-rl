# rl.py
from __future__ import annotations
from dataclasses import dataclass
import random
import pickle
from typing import Dict, List, Tuple, Optional

State = Tuple[int, ...]  # -1,0,1 du point de vue du joueur courant
QTable = Dict[State, List[float]]

# ----------------- Morpion (absolu) -----------------
def check_winner_abs(board_abs: List[int]) -> int:
    wins = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        (0, 4, 8), (2, 4, 6),
    ]
    for a, b, c in wins:
        s = board_abs[a] + board_abs[b] + board_abs[c]
        if s == 3:
            return 1
        if s == -3:
            return -1
    return 0

def is_full_abs(board_abs: List[int]) -> bool:
    return all(v != 0 for v in board_abs)

# ----------------- Perspective RL -----------------
def available_actions_state(state: State) -> List[int]:
    return [i for i, v in enumerate(state) if v == 0]

def abs_to_state(board_abs: List[int], current_player_abs: int) -> State:
    """
    Convertit board_abs (X=+1, O=-1) en state du point de vue du joueur courant:
    joueur courant = +1, adversaire = -1
    """
    factor = 1 if current_player_abs == 1 else -1
    return tuple(v * factor for v in board_abs)

# ----------------- Symétries (canonicalisation + mapping actions) -----------------
# transform t : state_t[j] = state[t[j]]
# action original -> transformé : a' = inv[t][a] (où t[a'] == a)
# action transformé -> original : a = t[a']

_TRANSFORMS: List[Tuple[int, ...]] = [
    (0, 1, 2, 3, 4, 5, 6, 7, 8),          # identité
    (6, 3, 0, 7, 4, 1, 8, 5, 2),          # rotation 90
    (8, 7, 6, 5, 4, 3, 2, 1, 0),          # rotation 180
    (2, 5, 8, 1, 4, 7, 0, 3, 6),          # rotation 270
    (2, 1, 0, 5, 4, 3, 8, 7, 6),          # miroir vertical
    (6, 7, 8, 3, 4, 5, 0, 1, 2),          # miroir horizontal
    (0, 3, 6, 1, 4, 7, 2, 5, 8),          # diag principale
    (8, 5, 2, 7, 4, 1, 6, 3, 0),          # diag secondaire
]

_INV_POS: List[List[int]] = []
for t in _TRANSFORMS:
    inv = [0] * 9
    for new_idx, old_idx in enumerate(t):
        inv[old_idx] = new_idx
    _INV_POS.append(inv)

def transform_state(state: State, t: Tuple[int, ...]) -> State:
    return tuple(state[i] for i in t)

def canonicalize(state: State) -> Tuple[State, int]:
    best = None
    best_k = 0
    for k, t in enumerate(_TRANSFORMS):
        s2 = transform_state(state, t)
        if best is None or s2 < best:
            best = s2
            best_k = k
    return best, best_k

def action_to_canonical(action: int, transform_id: int) -> int:
    return _INV_POS[transform_id][action]

def action_from_canonical(action_c: int, transform_id: int) -> int:
    return _TRANSFORMS[transform_id][action_c]

# ----------------- Agent Q-learning (zéro-somme) -----------------
@dataclass
class QLearningAgent:
    alpha: float = 0.25
    gamma: float = 0.95

    epsilon: float = 0.20
    epsilon_min: float = 0.02
    epsilon_decay: float = 0.9995

    qtable_path: str = "qtable.pkl"
    q: QTable = None

    def __post_init__(self):
        if self.q is None:
            self.q = {}
        self.load()

    def load(self) -> None:
        try:
            with open(self.qtable_path, "rb") as f:
                self.q = pickle.load(f)
        except FileNotFoundError:
            self.q = {}
        except Exception:
            self.q = {}

    def save(self) -> None:
        with open(self.qtable_path, "wb") as f:
            pickle.dump(self.q, f)

    def _ensure_state(self, s_canon: State) -> None:
        if s_canon not in self.q:
            self.q[s_canon] = [0.0] * 9

    def choose_action(self, s: State, epsilon_override: Optional[float] = None) -> int:
        """
        Décide dans le repère canonique, renvoie une action dans le repère original.
        """
        actions = available_actions_state(s)
        if not actions:
            raise ValueError("Aucune action possible")

        eps = self.epsilon if epsilon_override is None else float(epsilon_override)

        s_c, k = canonicalize(s)
        self._ensure_state(s_c)
        qvals = self.q[s_c]

        actions_c = [action_to_canonical(a, k) for a in actions]

        if random.random() < eps:
            a_c = random.choice(actions_c)
            return action_from_canonical(a_c, k)

        best_ac = max(actions_c, key=lambda ac: qvals[ac])
        return action_from_canonical(best_ac, k)

    def update(self, s: State, a: int, r: float, s_next: Optional[State], terminal: bool) -> None:
        """
        Update zéro-somme :
        s_next est l'état du JOUEUR SUIVANT (adversaire). Donc la valeur pour moi est l'opposé :
          target = r - gamma * max Q(s_next, a_next)
        """
        s_c, k = canonicalize(s)
        a_c = action_to_canonical(a, k)
        self._ensure_state(s_c)
        q_s = self.q[s_c]

        target = r
        if not terminal and s_next is not None:
            s2_c, k2 = canonicalize(s_next)
            self._ensure_state(s2_c)
            q_s2 = self.q[s2_c]

            acts2 = available_actions_state(s_next)
            if acts2:
                acts2_c = [action_to_canonical(x, k2) for x in acts2]
                target -= self.gamma * max(q_s2[ac] for ac in acts2_c)

        q_s[a_c] = q_s[a_c] + self.alpha * (target - q_s[a_c])

    def decay_epsilon(self) -> None:
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min

    def self_play(self, episodes: int = 500) -> Dict[str, float]:
        """
        Self-play équilibré : on alterne qui commence (X puis O).
        """
        x_wins = 0
        o_wins = 0
        draws = 0
        total_moves = 0

        for ep in range(episodes):
            board_abs = [0] * 9
            current = 1 if (ep % 2 == 0) else -1  # alternance

            moves = 0
            while True:
                s = abs_to_state(board_abs, current)
                actions = available_actions_state(s)
                if not actions:
                    draws += 1
                    break

                a = self.choose_action(s)  # exploration
                board_abs[a] = current
                moves += 1

                winner = check_winner_abs(board_abs)
                if winner != 0:
                    self.update(s, a, r=1.0, s_next=None, terminal=True)
                    if winner == 1:
                        x_wins += 1
                    else:
                        o_wins += 1
                    break

                if is_full_abs(board_abs):
                    self.update(s, a, r=0.0, s_next=None, terminal=True)
                    draws += 1
                    break

                next_player = -current
                s_next = abs_to_state(board_abs, next_player)
                self.update(s, a, r=0.0, s_next=s_next, terminal=False)
                current = next_player

            total_moves += moves
            self.decay_epsilon()

        total = max(1, episodes)
        return {
            "episodes": float(episodes),
            "x_win_rate": x_wins / total,
            "o_win_rate": o_wins / total,
            "draw_rate": draws / total,
            "avg_moves": total_moves / total,
            "epsilon": float(self.epsilon),
            "qtable_states": float(len(self.q)),
        }

    def train_vs_minimax(self, episodes: int = 500) -> Dict[str, float]:
        """
        Entraîne l'agent contre Minimax (optimal).
        - On alterne le joueur qui commence (X/O)
        - On alterne le symbole contrôlé par l'agent (agent en X puis agent en O)
        - On met à jour Q uniquement sur les coups de l'agent
        """
        from minimax import minimax_best_move

        agent_wins = 0
        agent_losses = 0
        draws = 0
        total_moves = 0

        for ep in range(episodes):
            board_abs = [0] * 9

            # alternance du starter
            current = 1 if (ep % 2 == 0) else -1

            # alternance du mark de l'agent (tous les 2 épisodes)
            agent_mark = 1 if ((ep // 2) % 2 == 0) else -1

            last_agent_s: Optional[State] = None
            last_agent_a: Optional[int] = None

            moves = 0
            while True:
                winner = check_winner_abs(board_abs)
                if winner != 0:
                    # si minimax vient de gagner, il faut punir le dernier coup agent
                    if last_agent_s is not None and last_agent_a is not None:
                        r = 1.0 if winner == agent_mark else -1.0
                        self.update(last_agent_s, last_agent_a, r=r, s_next=None, terminal=True)
                    if winner == agent_mark:
                        agent_wins += 1
                    else:
                        agent_losses += 1
                    break

                if is_full_abs(board_abs):
                    if last_agent_s is not None and last_agent_a is not None:
                        self.update(last_agent_s, last_agent_a, r=0.0, s_next=None, terminal=True)
                    draws += 1
                    break

                if current == agent_mark:
                    # coup agent
                    s = abs_to_state(board_abs, current)
                    a = self.choose_action(s)  # exploration via epsilon
                    board_abs[a] = current
                    moves += 1

                    winner2 = check_winner_abs(board_abs)
                    if winner2 != 0:
                        self.update(s, a, r=1.0, s_next=None, terminal=True)
                        agent_wins += 1
                        break

                    if is_full_abs(board_abs):
                        self.update(s, a, r=0.0, s_next=None, terminal=True)
                        draws += 1
                        break

                    # non terminal -> tour minimax
                    s_next = abs_to_state(board_abs, -current)
                    self.update(s, a, r=0.0, s_next=s_next, terminal=False)

                    last_agent_s, last_agent_a = s, a
                    current = -current
                else:
                    # coup minimax
                    mv = minimax_best_move(board_abs, current)
                    board_abs[mv] = current
                    moves += 1
                    current = -current

            total_moves += moves
            self.decay_epsilon()

        total = max(1, episodes)
        return {
            "episodes": float(episodes),
            "agent_win_rate": agent_wins / total,
            "agent_loss_rate": agent_losses / total,
            "draw_rate": draws / total,
            "avg_moves": total_moves / total,
            "epsilon": float(self.epsilon),
            "qtable_states": float(len(self.q)),
        }
