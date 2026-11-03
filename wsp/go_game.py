"""9x9 Go engine and policy networks for Experiment 6 (Go from scratch).

The rules mirror the task prompt in `runs/budget_*/message.txt` of
`upstream/exp6_go_from_scratch`:

* 9x9 board, rows 0 (top)..8 (bottom), columns 0 (left)..8 (right).
* Black moves first, players alternate; a move places a stone on an empty point
  or passes.
* A group with no adjacent empty point is captured and removed.
* Suicide is illegal; simple ko (immediate single-stone recapture) is illegal
  for one move only.
* A stone placement that repeats an earlier board position (stones only, pgx's
  positional superko) ends the game at once and the player who made it loses.
* The game ends after two consecutive passes or after 162 moves in total.
* Area scoring: stones plus empty points surrounded only by one player's stones;
  White receives a 7.5-point komi. Higher score wins (no draws).

The policy networks are plain feed-forward ReLU networks over the 243-number
input (3 planes over the 81 points, from the mover's point of view) and 82
outputs (81 points + pass). Illegal moves are masked and the move is sampled
from the softmax of the remaining logits at temperature 1.
"""

import json
from pathlib import Path

import numpy as np

BLACK = 1
WHITE = 2
EMPTY = 0

SIZE = 9
N = SIZE * SIZE  # 81 board points
PASS = N  # move 81 means pass
NUM_OUTPUTS = N + 1  # 82 logits
KOMI = 7.5
MAX_MOVES = 162

_TENSOR_SUFFIXES = ("W", "b")


def other(color):
    """The opposite colour."""
    return BLACK if color == WHITE else WHITE


def _neighbors(row, col):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = row + dr, col + dc
        if 0 <= rr < SIZE and 0 <= cc < SIZE:
            yield rr, cc


def _group_liberties(board, row, col):
    """Number of distinct empty points adjacent to the group at (row, col)."""
    color = board[row, col]
    if color == EMPTY:
        return 0
    seen = np.zeros((SIZE, SIZE), dtype=bool)
    stack = [(row, col)]
    seen[row, col] = True
    liberties = set()
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors(r, c):
            v = board[nr, nc]
            if v == EMPTY:
                liberties.add((nr, nc))
            elif v == color and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return len(liberties)


def _group_points(board, row, col):
    """Flat indices of every stone in the group at (row, col)."""
    color = board[row, col]
    if color == EMPTY:
        return []
    seen = np.zeros((SIZE, SIZE), dtype=bool)
    stack = [(row, col)]
    seen[row, col] = True
    points = []
    while stack:
        r, c = stack.pop()
        points.append(r * SIZE + c)
        for nr, nc in _neighbors(r, c):
            if board[nr, nc] == color and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return points


def _remove_group(board, row, col):
    """Remove the whole group at (row, col); return the number removed."""
    color = board[row, col]
    if color == EMPTY:
        return 0
    stack = [(row, col)]
    board[row, col] = EMPTY
    removed = 0
    while stack:
        r, c = stack.pop()
        removed += 1
        for nr, nc in _neighbors(r, c):
            if board[nr, nc] == color:
                board[nr, nc] = EMPTY
                stack.append((nr, nc))
    return removed


def encode(game):
    """The 243-number input from the point of view of the player to move.

    plane 0: mover's stones, plane 1: opponent's stones, plane 2: 1 everywhere
    when White is to move (0 when Black is). Index (row*9+col)*3+plane.
    """
    x = np.zeros(3 * N, dtype=np.float32)
    flat = game.board.reshape(-1)
    mover = game.to_move
    opp = other(mover)
    x[0::3] = (flat == mover).astype(np.float32)
    x[1::3] = (flat == opp).astype(np.float32)
    if mover == WHITE:
        x[2::3] = 1.0
    return x


def sample_move(logits, legal_mask, rng, temperature=1.0):
    """Sample a move from the softmax of the legal logits at the given temperature."""
    legal = np.asarray(legal_mask, dtype=bool)
    if not legal.any():
        return int(PASS)
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    z = z - np.max(z[legal])
    e = np.zeros(NUM_OUTPUTS, dtype=np.float64)
    e[legal] = np.exp(z[legal])
    p = e / e.sum()
    return int(rng.choice(NUM_OUTPUTS, p=p))


class GoGame:
    """A 9x9 Go position with capture, suicide, simple ko, superko and scoring."""

    def __init__(self, komi=KOMI, max_moves=MAX_MOVES):
        self.komi = float(komi)
        self.max_moves = int(max_moves)
        self.reset()

    def reset(self):
        self.board = np.zeros((SIZE, SIZE), dtype=np.int8)
        self.to_move = BLACK
        self.ko = -1
        self.seen = {self.board.tobytes()}
        self.passes = 0
        self.move_count = 0
        self.done = False
        self.winner = EMPTY
        self.end_reason = None
        self.captured = {BLACK: 0, WHITE: 0}

    def set_board(self, board, to_move, move_count=0, passes=0, ko=-1):
        """Load an arbitrary position (test/debug helper)."""
        self.board = np.asarray(board, dtype=np.int8).copy()
        self.to_move = int(to_move)
        self.move_count = int(move_count)
        self.passes = int(passes)
        self.ko = int(ko)
        self.done = False
        self.winner = EMPTY
        self.end_reason = None
        self.seen = {self.board.tobytes()}

    def _simulate_place(self, idx):
        """Board after placing a stone at idx and captures, or (None, 0, []) for suicide."""
        row, col = divmod(int(idx), SIZE)
        color = self.to_move
        board = self.board.copy()
        board[row, col] = color
        opp = other(color)
        captured = 0
        captured_points = []
        for nr, nc in _neighbors(row, col):
            if board[nr, nc] == opp and _group_liberties(board, nr, nc) == 0:
                points = _group_points(board, nr, nc)
                captured += len(points)
                captured_points.extend(points)
                _remove_group(board, nr, nc)
        if _group_liberties(board, row, col) == 0:
            return None, 0, []
        return board, captured, captured_points

    def _compute_ko(self, idx, captured, captured_points):
        """pgx simple ko: after capturing a single stone whose recapture point is the
        placed stone's only liberty, that point is illegal for the next move only."""
        if captured != 1 or not captured_points:
            return -1
        row, col = divmod(int(idx), SIZE)
        opp = other(self.to_move)
        for nr, nc in _neighbors(row, col):
            if self.board[nr, nc] != opp:
                return -1
        return int(captured_points[0])

    def _legal_place(self, idx):
        """True if placing a stone at idx is legal (empty, not suicide, not simple ko)."""
        if self.board.flat[int(idx)] != EMPTY:
            return False
        if int(idx) == self.ko:
            return False  # simple ko
        board, _captured, _points = self._simulate_place(idx)
        return board is not None

    def legal_mask(self):
        """Boolean mask over the 82 outputs; pass is always legal."""
        mask = np.zeros(NUM_OUTPUTS, dtype=bool)
        mask[PASS] = True
        if self.done:
            return mask
        for idx in range(N):
            if self.board.flat[idx] == EMPTY and self._legal_place(idx):
                mask[idx] = True
        return mask

    def apply(self, move):
        """Play a move (0..80 point, 81 pass). Raises ValueError if illegal."""
        if self.done:
            raise ValueError("game is already over")

        if int(move) == PASS:
            self.ko = -1
            self.passes += 1
            self.move_count += 1
            self.to_move = other(self.to_move)
            self.seen.add(self.board.tobytes())
            if self.passes >= 2:
                self._end("two_passes")
            elif self.move_count >= self.max_moves:
                self._end("move_limit")
            return

        idx = int(move)
        if not 0 <= idx < N:
            raise ValueError(f"move out of range: {move}")
        if self.board.flat[idx] != EMPTY:
            raise ValueError(f"occupied point: {idx}")
        if idx == self.ko:
            raise ValueError(f"simple ko at {idx}")

        board, captured, captured_points = self._simulate_place(idx)
        if board is None:
            raise ValueError(f"suicide at {idx}")

        color = self.to_move
        board_bytes = board.tobytes()
        repetition = board_bytes in self.seen
        ko = self._compute_ko(idx, captured, captured_points)

        self.board = board
        self.captured[color] += captured
        self.ko = ko
        self.to_move = other(color)
        self.passes = 0
        self.move_count += 1
        self.seen.add(board_bytes)

        # Positional superko (pgx): a stone placement that recreates an earlier
        # board ends the game at once and the player who made the move loses.
        if repetition:
            self._end("repetition", winner=other(color))
            return
        if self.move_count >= self.max_moves:
            self._end("move_limit")

    def _end(self, reason, winner=None):
        self.done = True
        self.end_reason = reason
        if winner is not None:
            self.winner = winner
        else:
            black, white = self.scores()
            self.winner = BLACK if black > white else WHITE

    def _territory(self, color):
        """Empty points whose every adjacent stone belongs to `color`."""
        visited = np.zeros((SIZE, SIZE), dtype=bool)
        count = 0
        for r in range(SIZE):
            for c in range(SIZE):
                if self.board[r, c] != EMPTY or visited[r, c]:
                    continue
                stack = [(r, c)]
                visited[r, c] = True
                region = 0
                adjacent = set()
                while stack:
                    rr, cc = stack.pop()
                    region += 1
                    for nr, nc in _neighbors(rr, cc):
                        v = self.board[nr, nc]
                        if v == EMPTY:
                            if not visited[nr, nc]:
                                visited[nr, nc] = True
                                stack.append((nr, nc))
                        else:
                            adjacent.add(int(v))
                if adjacent == {color}:
                    count += region
        return count

    def scores(self):
        """Area scores as (black, white); white includes komi."""
        black = int((self.board == BLACK).sum()) + self._territory(BLACK)
        white = int((self.board == WHITE).sum()) + self._territory(WHITE) + self.komi
        return float(black), float(white)


class PolicyNet:
    """A ReLU policy network loaded from the published `.npy` + `arch.json` layout.

    `arch.json` holds `{"depth": D, "width": W}`; hidden layers are L1..LD (each
    W units) and the output layer is `out`. The written networks use D=6, W=512;
    the small hidden PPO opponents use D=2, W=32.
    """

    def __init__(self, tensors, depth, width):
        self.tensors = dict(tensors)
        self.depth = int(depth)
        self.width = int(width)

    @classmethod
    def load(cls, path):
        path = Path(path)
        arch = json.loads((path / "arch.json").read_text())
        depth, width = int(arch["depth"]), int(arch["width"])
        tensors = {}
        for k in range(1, depth + 1):
            for suffix in _TENSOR_SUFFIXES:
                name = f"L{k}.{suffix}"
                tensors[name] = np.load(path / f"{name}.npy")
        for suffix in _TENSOR_SUFFIXES:
            tensors[f"out.{suffix}"] = np.load(path / f"out.{suffix}.npy")
        return cls(tensors, depth, width)

    def logits(self, x):
        h = x
        for k in range(1, self.depth + 1):
            h = np.maximum(h @ self.tensors[f"L{k}.W"] + self.tensors[f"L{k}.b"], 0.0)
        return h @ self.tensors["out.W"] + self.tensors["out.b"]

    def move(self, game, rng, temperature=1.0):
        """Sample a legal move for `game` from this policy."""
        logits = self.logits(encode(game))
        return sample_move(logits, game.legal_mask(), rng, temperature)


def small_ppo_opponent(path):
    """Load the hidden evaluation opponent: a 243 -> 32 -> 32 -> 82 ReLU policy.

    The hidden PPO baselines (seeds 0 and 2) have two 32-unit hidden layers and
    are stored in the same `.npy` + `arch.json` layout. They are not published in
    this repository, but this loader is here so `go_nn` can use them when they
    are available.
    """
    net = PolicyNet.load(path)
    if (net.depth, net.width) != (2, 32):
        raise ValueError(
            f"expected a 243->32->32->82 PPO opponent, got depth={net.depth} width={net.width}"
        )
    return net


def play_game(submission, opponent, submission_color, rng):
    """Play one game to completion; returns the finished GoGame."""
    game = GoGame()
    while not game.done:
        mover = submission if game.to_move == submission_color else opponent
        game.apply(mover.move(game, rng))
    return game


def h2h(submission, opponent, games=2000, seed=31337):
    """Head-to-head win rate of `submission` vs `opponent`.

    The submission is Black in half the games (games alternate). Both policies
    sample moves. Returns a summary dict matching the published h2h JSON fields.
    """
    rng = np.random.RandomState(seed)
    wins = 0
    wins_as_black = wins_as_white = 0
    black_games = white_games = 0
    margins = []
    lengths = []
    ends = {"two_passes": 0, "repetition": 0, "move_limit": 0}

    for g in range(games):
        sub_color = BLACK if g % 2 == 0 else WHITE
        game = play_game(submission, opponent, sub_color, rng)

        sub_won = game.winner == sub_color
        wins += int(sub_won)
        if sub_color == BLACK:
            black_games += 1
            wins_as_black += int(sub_won)
        else:
            white_games += 1
            wins_as_white += int(sub_won)

        black_score, white_score = game.scores()
        sub_score = black_score if sub_color == BLACK else white_score
        opp_score = white_score if sub_color == BLACK else black_score
        margins.append(sub_score - opp_score)
        lengths.append(game.move_count)
        if game.end_reason is not None:
            ends[game.end_reason] = ends.get(game.end_reason, 0) + 1

    return {
        "games": games,
        "score": wins / games,
        "wins": wins,
        "losses": games - wins,
        "score_as_black": wins_as_black / black_games if black_games else 0.0,
        "score_as_white": wins_as_white / white_games if white_games else 0.0,
        "mean_margin": float(np.mean(margins)) if margins else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "end_two_passes": ends.get("two_passes", 0),
        "end_repetition": ends.get("repetition", 0),
        "end_move_limit": ends.get("move_limit", 0),
    }
