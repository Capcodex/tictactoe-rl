let gameId = null;
let state = null;

const boardEl = document.getElementById("board");
const msgEl = document.getElementById("msg");

const botNameEl = document.getElementById("botName");
const humanMarkEl = document.getElementById("humanMark");
const botMarkEl = document.getElementById("botMark");
const turnEl = document.getElementById("turn");
const epsEl = document.getElementById("eps");

function setMsg(text, kind="") {
  msgEl.className = "msg " + kind;
  msgEl.textContent = text;
}
function pct(x) { return (x * 100).toFixed(1) + "%"; }

function renderGlobalStats(gs) {
  if (!gs) return;
  document.getElementById("gGames").textContent = gs.games_total ?? "-";
  document.getElementById("gBot").textContent = gs.bot_wins ?? "-";
  document.getElementById("gHuman").textContent = gs.human_wins ?? "-";
  document.getElementById("gDraws").textContent = gs.draws ?? "-";
}

function renderLastTrain(mode, s) {
  if (!s) return;
  document.getElementById("tMode").textContent = mode || "-";
  document.getElementById("stEpisodes").textContent = (s.episodes ?? "-").toString();
  document.getElementById("stEps").textContent = (s.epsilon ?? 0).toFixed(3);
  document.getElementById("stD").textContent = s.draw_rate != null ? pct(s.draw_rate) : "-";
  document.getElementById("stM").textContent = s.avg_moves != null ? s.avg_moves.toFixed(2) : "-";
  document.getElementById("stQ").textContent = s.qtable_states != null ? Math.round(s.qtable_states) : "-";

  const hintEl = document.getElementById("trainHint");
  if (mode === "minimax") {
    document.getElementById("stA").textContent = s.agent_win_rate != null ? pct(s.agent_win_rate) : "-";
    document.getElementById("stB").textContent = s.agent_loss_rate != null ? pct(s.agent_loss_rate) : "-";
    hintEl.textContent = "Score A = victoires agent, Score B = défaites agent. Contre Minimax, viser surtout beaucoup de nuls.";
  } else {
    document.getElementById("stA").textContent = s.x_win_rate != null ? pct(s.x_win_rate) : "-";
    document.getElementById("stB").textContent = s.o_win_rate != null ? pct(s.o_win_rate) : "-";
    hintEl.textContent = "Score A = X win, Score B = O win (self-play). À terme, le taux de nuls devrait monter fortement.";
  }
}

function renderArena(r) {
  if (!r) return;
  document.getElementById("aX").textContent = r.x_wins ?? "-";
  document.getElementById("aO").textContent = r.o_wins ?? "-";
  document.getElementById("aD").textContent = r.draws ?? "-";
  document.getElementById("aM").textContent = r.avg_moves != null ? r.avg_moves.toFixed(2) : "-";
  document.getElementById("aE").textContent = r.errors ?? "-";

  const hint = document.getElementById("arenaHint");
  if (r.errors > 0) {
    hint.textContent = `Dernière erreur: ${r.last_error || "(non précisée)"}`;
  } else {
    hint.textContent = `Matchs: ${r.games} | X=${r.x} vs O=${r.o}`;
  }
}

function render() {
  if (!state) return;

  botNameEl.textContent = state.bot_name ?? "?";
  humanMarkEl.textContent = state.human ?? "?";
  botMarkEl.textContent = state.bot ?? "?";
  turnEl.textContent = state.turn ?? "?";
  epsEl.textContent = (state.epsilon ?? 0).toFixed(3);

  renderGlobalStats(state.global_stats);

  boardEl.innerHTML = "";
  state.board.forEach((v, i) => {
    const d = document.createElement("div");
    d.className = "cell " + (v === "X" ? "x" : (v === "O" ? "o" : ""));
    if (state.done || v !== "") d.classList.add("disabled");
    d.textContent = v;
    d.addEventListener("click", () => onClickCell(i));
    boardEl.appendChild(d);
  });

  if (state.error) {
    setMsg(state.error, "bad");
    return;
  }

  if (state.done) {
    if (state.winner === "") {
      setMsg("Match nul.", "");
    } else {
      const botWon = (state.winner === state.bot);
      setMsg(`Victoire de ${state.winner}.`, botWon ? "bad" : "good");
    }
  } else {
    setMsg("Clique sur une case vide pour jouer.", "");
  }
}

async function refreshEpsilonUI() {
  const res = await fetch("/api/epsilon");
  const data = await res.json();
  if (data.ok) {
    document.getElementById("epsilonVal").value = (data.epsilon ?? 0.2).toFixed(2);
    epsEl.textContent = (data.epsilon ?? 0).toFixed(3);
  }
}

async function applyEpsilon() {
  const eps = parseFloat(document.getElementById("epsilonVal").value || "0.2");
  const res = await fetch("/api/epsilon", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({ epsilon: eps })
  });
  const data = await res.json();
  if (data.ok) {
    setMsg(`ε appliqué: ${data.epsilon.toFixed(3)}`, "good");
    await refreshEpsilonUI();
  } else {
    setMsg(data.error || "Erreur epsilon", "bad");
  }
}

function buildNewGameUrl(humanAs) {
  const opponent = document.getElementById("opponent").value;
  const remoteUrl = (document.getElementById("remoteUrl").value || "").trim();

  if (opponent === "remote" && !remoteUrl) {
    setMsg("Tu as choisi Remote API mais l’URL est vide.", "bad");
    return null;
  }

  let url = `/api/new?bot=${encodeURIComponent(opponent)}&human_as=${encodeURIComponent(humanAs)}`;
  if (opponent === "remote") {
    url += `&remote_url=${encodeURIComponent(remoteUrl)}`;
  }
  return url;
}

async function newGame(humanAs="X") {
  const url = buildNewGameUrl(humanAs);
  if (!url) return;

  const res = await fetch(url);
  const data = await res.json();

  if (!res.ok) {
    setMsg(data.error || "Erreur création partie.", "bad");
    return;
  }

  state = data;
  gameId = state.id;
  render();
}

async function onClickCell(pos) {
  if (!state || state.done) return;
  if (state.turn === state.bot) return;
  if (state.board[pos] !== "") return;

  const res = await fetch("/api/move", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({ game_id: gameId, pos })
  });

  const data = await res.json();
  if (data.error) {
    setMsg(data.error, "bad");
    return;
  }
  state = data;
  render();
}

async function train() {
  const episodes = parseInt(document.getElementById("episodes").value || "1000", 10);
  const mode = document.getElementById("mode").value;
  const eps = parseFloat(document.getElementById("epsilonVal").value || "0.2");

  setMsg(`Entraînement en cours (${mode}, ${episodes} épisodes, ε=${eps})…`, "");
  const res = await fetch("/api/train", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({ episodes, mode, epsilon: eps })
  });

  const data = await res.json();
  if (!data.ok) {
    setMsg("Erreur entraînement.", "bad");
    return;
  }

  renderLastTrain(data.mode || mode, data.stats);
  if (data.global_stats) renderGlobalStats(data.global_stats);

  const dr = data.stats?.draw_rate ?? 0;
  setMsg(`OK (${data.mode}). ε=${(data.epsilon ?? 0).toFixed(3)} | nuls≈${pct(dr)}`, "good");
  await refreshEpsilonUI();
}

async function runArena() {
  const x = document.getElementById("arenaX").value;
  const o = document.getElementById("arenaO").value;
  const games = parseInt(document.getElementById("arenaGames").value || "200", 10);
  const remoteUrl = (document.getElementById("remoteUrl").value || "").trim();

  if ((x === "remote" || o === "remote") && !remoteUrl) {
    setMsg("Arena: Remote sélectionné mais l’URL est vide.", "bad");
    return;
  }

  setMsg(`Arena en cours: X=${x} vs O=${o} (${games} parties)…`, "");

  const res = await fetch("/api/arena", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({ x, o, games, remote_url: remoteUrl })
  });
  const data = await res.json();

  if (!res.ok || !data.ok) {
    setMsg(data.error || "Erreur arena.", "bad");
    return;
  }

  renderArena(data.result);
  setMsg(`Arena OK: Xwins=${data.result.x_wins}, Owins=${data.result.o_wins}, nuls=${data.result.draws}`, "good");
}

document.getElementById("playX").addEventListener("click", () => newGame("X"));
document.getElementById("playO").addEventListener("click", () => newGame("O"));
document.getElementById("trainBtn").addEventListener("click", train);
document.getElementById("setEpsBtn").addEventListener("click", applyEpsilon);
document.getElementById("arenaBtn").addEventListener("click", runArena);

refreshEpsilonUI().then(() => newGame("X"));
