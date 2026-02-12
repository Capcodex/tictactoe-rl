# app.py
from __future__ import annotations
from flask import Flask, jsonify, request, send_from_directory
import uuid
from typing import Dict, Any, Optional, Tuple
import requests

from rl import QLearningAgent, check_winner_abs, is_full_abs, abs_to_state
from minimax import minimax_best_move
from flask_cors import CORS

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

app = Flask(__name__, static_folder="static", static_url_path="")

agent = QLearningAgent(qtable_path="qtable.pkl")

GAMES: Dict[str, Dict[str, Any]] = {}

STATS: Dict[str, Any] = {
    "games_total": 0,
    "bot_wins": 0,
    "human_wins": 0,
    "draws": 0,

    "selfplay_episodes_total": 0,
    "selfplay_x_wins": 0,
    "selfplay_o_wins": 0,
    "selfplay_draws": 0,

    "minimax_episodes_total": 0,
    "minimax_wins": 0,
    "minimax_losses": 0,
    "minimax_draws": 0,
}


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


# ------------------ EPSILON CONTROL ------------------
@app.get("/api/epsilon")
def get_epsilon():
    return jsonify({"ok": True, "epsilon": float(agent.epsilon), "epsilon_min": float(agent.epsilon_min)})


@app.post("/api/epsilon")
def set_epsilon():
    data = request.get_json(force=True) if request.data else {}
    try:
        eps = float(data.get("epsilon", agent.epsilon))
    except Exception:
        return jsonify({"ok": False, "error": "epsilon invalide"}), 400

    eps = max(0.0, min(1.0, eps))
    agent.epsilon = eps
    agent.save()
    return jsonify({"ok": True, "epsilon": float(agent.epsilon)})


# ------------------ GAME API (humain vs bot) ------------------
@app.get("/api/new")
def new_game():
    bot = (request.args.get("bot", "rl") or "rl").lower()
    if bot not in ("rl", "minimax", "remote"):
        bot = "rl"

    human_as = (request.args.get("human_as") or "X").upper()
    human_mark = 1 if human_as == "X" else -1
    bot_mark = -human_mark

    remote_url = (request.args.get("remote_url") or "").strip().rstrip("/")

    # strict remote
    if bot == "remote" and not remote_url:
        return jsonify({"error": "Remote API sélectionnée mais l'URL est vide."}), 400

    gid = str(uuid.uuid4())
    game = {
        "id": gid,
        "board": [0] * 9,
        "turn": 1,
        "bot_kind": bot,
        "bot_mark": bot_mark,
        "human_mark": human_mark,
        "remote_url": remote_url,

        "done": False,
        "winner": 0,
        "counted": False,

        "last_bot_s": None,
        "last_bot_a": None,

        "error": "",
    }
    GAMES[gid] = game

    if game["turn"] == game["bot_mark"] and not game["done"]:
        _bot_move(game)

    return jsonify(_public_game(game))


@app.post("/api/move")
def human_move():
    data = request.get_json(force=True)
    gid = data.get("game_id")
    pos = int(data.get("pos"))

    if gid not in GAMES:
        return jsonify({"error": "Partie introuvable"}), 404

    game = GAMES[gid]
    if game["done"]:
        return jsonify(_public_game(game))

    if game.get("error"):
        game["done"] = True
        game["winner"] = 0
        _maybe_count_game_end(game)
        return jsonify(_public_game(game))

    human_mark = game["human_mark"]

    if game["turn"] != human_mark:
        return jsonify({"error": "Ce n'est pas à toi de jouer"}), 400

    if pos < 0 or pos > 8 or game["board"][pos] != 0:
        return jsonify({"error": "Coup invalide"}), 400

    game["board"][pos] = human_mark
    _update_terminal(game)

    if game["done"]:
        _credit_last_rl_if_needed(game)
        return jsonify(_public_game(game))

    game["turn"] *= -1
    _bot_move(game)

    return jsonify(_public_game(game))


@app.post("/api/train")
def train():
    data = request.get_json(force=True) if request.data else {}
    episodes = int(data.get("episodes", 1000))
    episodes = max(1, min(episodes, 50000))

    if "epsilon" in data:
        try:
            eps = float(data["epsilon"])
            agent.epsilon = max(0.0, min(1.0, eps))
        except Exception:
            pass

    mode = (data.get("mode") or "selfplay").lower()
    if mode not in ("selfplay", "minimax"):
        mode = "selfplay"

    if mode == "minimax":
        stats = agent.train_vs_minimax(episodes=episodes)
    else:
        stats = agent.self_play(episodes=episodes)

    agent.save()

    eps_count = int(stats.get("episodes", episodes))

    if mode == "selfplay":
        STATS["selfplay_episodes_total"] += eps_count
        STATS["selfplay_x_wins"] += int(round(stats.get("x_win_rate", 0.0) * eps_count))
        STATS["selfplay_o_wins"] += int(round(stats.get("o_win_rate", 0.0) * eps_count))
        STATS["selfplay_draws"] += int(round(stats.get("draw_rate", 0.0) * eps_count))

    if mode == "minimax":
        STATS["minimax_episodes_total"] += eps_count
        STATS["minimax_wins"] += int(round(stats.get("agent_win_rate", 0.0) * eps_count))
        STATS["minimax_losses"] += int(round(stats.get("agent_loss_rate", 0.0) * eps_count))
        STATS["minimax_draws"] += int(round(stats.get("draw_rate", 0.0) * eps_count))

    return jsonify({"ok": True, "mode": mode, "stats": stats, "global_stats": STATS, "epsilon": float(agent.epsilon)})


@app.get("/api/state")
def state():
    gid = request.args.get("game_id")
    if not gid or gid not in GAMES:
        return jsonify({"error": "Partie introuvable"}), 404
    return jsonify(_public_game(GAMES[gid]))


# ------------------ ARENA API (bot vs bot) ------------------
@app.post("/api/arena")
def arena():
    """
    Body JSON:
      - x: "rl" | "minimax" | "remote"
      - o: "rl" | "minimax" | "remote"
      - remote_url: obligatoire si x ou o == "remote"
      - games: int (1..5000)
    """
    data = request.get_json(force=True) if request.data else {}

    x_kind = (data.get("x") or "remote").lower()
    o_kind = (data.get("o") or "rl").lower()
    if x_kind not in ("rl", "minimax", "remote"):
        x_kind = "rl"
    if o_kind not in ("rl", "minimax", "remote"):
        o_kind = "rl"

    remote_url = (data.get("remote_url") or "").strip().rstrip("/")
    if ("remote" in (x_kind, o_kind)) and not remote_url:
        return jsonify({"ok": False, "error": "remote_url requis (Remote API sélectionnée)."}), 400

    games = int(data.get("games", 50))
    games = max(1, min(games, 5000))

    results = {
        "games": games,
        "x": x_kind,
        "o": o_kind,
        "x_wins": 0,
        "o_wins": 0,
        "draws": 0,
        "avg_moves": 0.0,
        "errors": 0,
        "last_error": "",
    }

    total_moves = 0

    for _ in range(games):
        board_abs = [0] * 9
        turn = 1  # X
        moves = 0
        error_msg = ""

        while True:
            winner = check_winner_abs(board_abs)
            if winner != 0:
                if winner == 1:
                    results["x_wins"] += 1
                else:
                    results["o_wins"] += 1
                break
            if is_full_abs(board_abs):
                results["draws"] += 1
                break

            player_kind = x_kind if turn == 1 else o_kind
            idx, err = _choose_bot_move_arena(
                board_abs=board_abs,
                player_abs=turn,
                kind=player_kind,
                remote_url=remote_url,
            )

            if err:
                error_msg = err
                break

            board_abs[idx] = turn
            moves += 1
            turn *= -1

        total_moves += moves

        if error_msg:
            results["errors"] += 1
            results["last_error"] = error_msg
            results["draws"] += 1

    results["avg_moves"] = (total_moves / games) if games else 0.0
    return jsonify({"ok": True, "result": results})


def _choose_bot_move_arena(
    board_abs: list[int],
    player_abs: int,
    kind: str,
    remote_url: str,
) -> Tuple[int, str]:
    if kind == "minimax":
        return minimax_best_move(board_abs, player_abs), ""

    if kind == "rl":
        # évaluation : greedy, pas d'exploration
        s = abs_to_state(board_abs, player_abs)
        idx = agent.choose_action(s, epsilon_override=0.0)
        if idx < 0 or idx > 8 or board_abs[idx] != 0:
            return 0, "RL a produit un coup illégal (inattendu)."
        return idx, ""

    # remote
    idx = _remote_move_board(board_abs, player_abs, remote_url)
    if idx is None:
        return 0, "Remote API injoignable (timeout/réponse invalide)."
    if idx < 0 or idx > 8 or board_abs[idx] != 0:
        return 0, "Remote API a renvoyé un coup illégal."
    return idx, ""


# ------------------ Helpers ------------------
def _public_game(game: Dict[str, Any]) -> Dict[str, Any]:
    def cell(v: int) -> str:
        return "X" if v == 1 else ("O" if v == -1 else "")

    if game["bot_kind"] == "minimax":
        bot_name = "Minimax"
    elif game["bot_kind"] == "remote":
        bot_name = "Remote API"
    else:
        bot_name = "RL"

    return {
        "id": game["id"],
        "board": [cell(v) for v in game["board"]],
        "done": game["done"],
        "winner": "X" if game["winner"] == 1 else ("O" if game["winner"] == -1 else ""),
        "bot_kind": game["bot_kind"],
        "bot_name": bot_name,
        "bot": "X" if game["bot_mark"] == 1 else "O",
        "human": "X" if game["human_mark"] == 1 else "O",
        "turn": "X" if game["turn"] == 1 else "O",
        "epsilon": float(agent.epsilon),
        "global_stats": STATS,
        "remote_url": game.get("remote_url", ""),
        "error": game.get("error", ""),
    }


def _maybe_count_game_end(game: Dict[str, Any]) -> None:
    if not game["done"] or game.get("counted"):
        return

    STATS["games_total"] += 1

    if game["winner"] == 0:
        STATS["draws"] += 1
    elif game["winner"] == game["bot_mark"]:
        STATS["bot_wins"] += 1
    else:
        STATS["human_wins"] += 1

    game["counted"] = True


def _update_terminal(game: Dict[str, Any]) -> None:
    w = check_winner_abs(game["board"])
    if w != 0:
        game["done"] = True
        game["winner"] = w
        _maybe_count_game_end(game)
        return
    if is_full_abs(game["board"]):
        game["done"] = True
        game["winner"] = 0
        _maybe_count_game_end(game)


def _credit_last_rl_if_needed(game: Dict[str, Any]) -> None:
    if game["bot_kind"] != "rl":
        return

    last_s = game.get("last_bot_s")
    last_a = game.get("last_bot_a")
    if last_s is None or last_a is None:
        return

    if game["winner"] == 0:
        r = 0.0
    elif game["winner"] == game["bot_mark"]:
        r = 1.0
    else:
        r = -1.0

    agent.update(last_s, last_a, r=r, s_next=None, terminal=True)
    agent.save()

    game["last_bot_s"] = None
    game["last_bot_a"] = None


def _board_to_remote_payload(board_abs: list[int], player_abs: int) -> Dict[str, Any]:
    board = []
    for v in board_abs:
        if v == 1:
            board.append("X")
        elif v == -1:
            board.append("O")
        else:
            board.append(" ")
    you_are = "X" if player_abs == 1 else "O"
    return {"board": board, "you_are": you_are}


def _remote_move_board(board_abs: list[int], player_abs: int, remote_url: str) -> Optional[int]:
    base = (remote_url or "").strip().rstrip("/")
    if not base:
        return None
    url = f"{base}/move"
    payload = _board_to_remote_payload(board_abs, player_abs)
    try:
        resp = requests.post(url, json=payload, timeout=2.5)
        if resp.status_code != 200:
            return None
        data = resp.json()
        return int(data.get("idx"))
    except Exception:
        return None


def _remote_move(game: Dict[str, Any]) -> Optional[int]:
    return _remote_move_board(game["board"], game["bot_mark"], game.get("remote_url", ""))


def _bot_move(game: Dict[str, Any]) -> None:
    if game["done"]:
        return

    bot_mark = game["bot_mark"]
    if game["turn"] != bot_mark:
        return

    if game["bot_kind"] == "minimax":
        a = minimax_best_move(game["board"], bot_mark)
        game["board"][a] = bot_mark
        _update_terminal(game)
        if not game["done"]:
            game["turn"] *= -1
        return

    if game["bot_kind"] == "remote":
        a = _remote_move(game)
        if a is None:
            game["error"] = "Remote API injoignable (timeout/réponse invalide)."
            game["done"] = True
            game["winner"] = 0
            _maybe_count_game_end(game)
            return

        if a < 0 or a > 8 or game["board"][a] != 0:
            game["error"] = "Remote API a renvoyé un coup illégal."
            game["done"] = True
            game["winner"] = 0
            _maybe_count_game_end(game)
            return

        game["board"][a] = bot_mark
        _update_terminal(game)
        if not game["done"]:
            game["turn"] *= -1
        return

    # RL
    s = abs_to_state(game["board"], bot_mark)
    a = agent.choose_action(s, epsilon_override=0.0)

    game["last_bot_s"] = s
    game["last_bot_a"] = a

    game["board"][a] = bot_mark
    _update_terminal(game)

    if game["done"]:
        if game["winner"] == bot_mark:
            r = 1.0
        elif game["winner"] == 0:
            r = 0.0
        else:
            r = -1.0

        agent.update(s, a, r=r, s_next=None, terminal=True)
        agent.decay_epsilon()
        agent.save()

        game["last_bot_s"] = None
        game["last_bot_a"] = None
        return

    next_player = -bot_mark
    s_next = abs_to_state(game["board"], next_player)
    agent.update(s, a, r=0.0, s_next=s_next, terminal=False)
    agent.decay_epsilon()
    agent.save()

    game["turn"] *= -1


if __name__ == "__main__":
    agent.self_play(episodes=2000)
    agent.save()
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
