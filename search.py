import chess
import time
import threading
from typing import Dict, List, Optional, Tuple
from evaluation import Evaluator

MATE_SCORE = 100000
MAX_DEPTH = 20
QS_DEPTH = 4
NULL_MOVE_REDUCTION = 3
LMP_DEPTH = 3
LMP_LIMIT = 4

class TTEntry:
    def __init__(self, score: int, depth: int, flag: str, move: Optional[chess.Move]):
        self.score = score
        self.depth = depth
        self.flag = flag
        self.move = move


class TranspositionTable:
    """Thread-safe transposition table with sharded locks for concurrent access."""

    def __init__(self, size_mb: int = 64):
        self.size = (size_mb * 1024 * 1024) // 32
        self.table: Dict[int, TTEntry] = {}
        self.hits = 0
        self.misses = 0
        self.num_shards = 16
        self.shards = [threading.Lock() for _ in range(self.num_shards)]

    def _get_shard_index(self, key: int) -> int:
        return key % self.num_shards

    def _get_lock(self, key: int) -> threading.Lock:
        return self.shards[self._get_shard_index(key)]

    def store(self, key: int, score: int, depth: int, flag: str, move: Optional[chess.Move]):
        lock = self._get_lock(key)
        with lock:
            if len(self.table) < self.size:
                self.table[key] = TTEntry(score, depth, flag, move)

    def probe(self, key: int, depth: int, alpha: int, beta: int) -> Tuple[Optional[TTEntry], bool]:
        lock = self._get_lock(key)
        with lock:
            if key not in self.table:
                self.misses += 1
                return None, False

            entry = self.table[key]

            if entry.depth < depth:
                self.misses += 1
                return entry, False

            self.hits += 1

            if entry.flag == 'exact':
                return entry, True
            elif entry.flag == 'lower' and entry.score >= beta:
                return entry, True
            elif entry.flag == 'upper' and entry.score <= alpha:
                return entry, True

            return entry, False


class HistoryTable:
    """History heuristic for move ordering."""

    def __init__(self):
        self.table = {}
        self.max_score = 1

    def update(self, color: chess.Color, move: chess.Move, depth: int):
        key = (color, move.from_square, move.to_square)
        bonus = depth * depth
        self.table[key] = self.table.get(key, 0) + bonus
        self.max_score = max(self.max_score, self.table[key])

    def get_score(self, color: chess.Color, move: chess.Move) -> int:
        key = (color, move.from_square, move.to_square)
        return self.table.get(key, 0)

    def age(self):
        for key in self.table:
            self.table[key] //= 2
        self.max_score //= 2


class KillerMoves:
    """Killer move heuristic for ordering quiet moves."""

    def __init__(self, max_depth: int = 64):
        self.killers = [[] for _ in range(max_depth)]

    def update(self, ply: int, move: chess.Move):
        if move in self.killers[ply]:
            return

        self.killers[ply].insert(0, move)
        if len(self.killers[ply]) > 2:
            self.killers[ply].pop()

    def is_killer(self, ply: int, move: chess.Move) -> bool:
        return move in self.killers[ply]


def order_moves(board: chess.Board, moves: List[chess.Move], tt_move: Optional[chess.Move],
               history: HistoryTable, killers: KillerMoves, ply: int) -> List[Tuple[int, chess.Move]]:
    ordered = []

    for move in moves:
        score = 0

        if tt_move and move == tt_move:
            score = 1000000

        elif board.is_capture(move):
            to_piece = board.piece_at(move.to_square)
            from_piece = board.piece_at(move.from_square)
            if to_piece and from_piece:
                victim = to_piece.piece_type
                attacker = from_piece.piece_type
                score = 100000 + victim * 10 - attacker

                if move.promotion:
                    score += 10000 + move.promotion * 10

        elif killers.is_killer(ply, move):
            score = 90000

        else:
            score = history.get_score(board.turn, move) * 1000 // max(1, history.max_score)

        if board.is_castling(move):
            score += 50000

        ordered.append((score, move))

    ordered.sort(key=lambda x: x[0], reverse=True)
    return ordered


def see(board: chess.Board, move: chess.Move) -> int:
    if not board.is_capture(move):
        return 0

    from_piece = board.piece_at(move.from_square)
    to_piece = board.piece_at(move.to_square)

    if from_piece is None or to_piece is None:
        return 0

    captured_value = PIECE_VALUES[to_piece.piece_type]

    board.push(move)

    attackers = board.attackers(not from_piece.color, move.to_square)
    min_attacker_value = float('inf')

    for square in attackers:
        piece = board.piece_at(square)
        if piece and piece.color != from_piece.color:
            if PIECE_VALUES[piece.piece_type] < min_attacker_value:
                min_attacker_value = PIECE_VALUES[piece.piece_type]

    board.pop()

    if min_attacker_value == float('inf'):
        return captured_value

    return int(captured_value - min_attacker_value)


PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}


class Search:
    """Chess engine search with alpha-beta pruning."""

    def __init__(self):
        self.evaluator = Evaluator()
        self.tt = TranspositionTable(size_mb=64)
        self.history = HistoryTable()
        self.killers = KillerMoves()
        self.nodes = 0
        self.start_time = 0
        self.max_time = 0
        self.stop_event = threading.Event()
        self.num_threads = 1

    def search(self, board: chess.Board, depth: int, time_limit: float = 1.0, num_threads: int = 1) -> chess.Move:
        """Main search function with iterative deepening and Lazy SMP."""
        self.start_time = time.time()
        self.max_time = time_limit
        self.stop_event.clear()
        self.nodes = 0
        self.num_threads = num_threads

        if num_threads == 1:
            return self._search_single_threaded(board, depth, time_limit)

        return self._search_lazy_smp(board, depth, time_limit, num_threads)

    def _search_single_threaded(self, board: chess.Board, depth: int, time_limit: float) -> chess.Move:
        """Single-threaded search for backward compatibility."""
        best_move = None
        best_score = 0

        for d in range(1, depth + 1):
            if self.stop_event.is_set():
                break

            lower = -MATE_SCORE
            upper = MATE_SCORE
            gamma = 0

            while lower < upper - 1:
                if self.stop_event.is_set():
                    break

                beta = gamma if gamma == lower else gamma + 1
                score, move = self._alpha_beta(board, d, beta - 1, beta, 0, True)

                if score >= beta:
                    lower = score
                    gamma = score
                    if move:
                        best_move = move
                        best_score = score
                else:
                    upper = score
                    gamma = score

            elapsed = time.time() - self.start_time
            self._report_info(d, best_score, elapsed, best_move)

            if elapsed >= self.max_time * 0.8:
                break

        self.history.age()

        return best_move if best_move else list(board.legal_moves)[0]

    def _search_lazy_smp(self, board: chess.Board, depth: int, time_limit: float, num_threads: int) -> chess.Move:
        """Lazy SMP multi-threaded search."""
        import concurrent.futures
        results = []
        boards = [board.copy() for _ in range(num_threads)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(self._search_single_threaded, b, depth, time_limit) for b in boards]
            for future in concurrent.futures.as_completed(futures):
                if self.stop_event.is_set():
                    break
                try:
                    move = future.result()
                    if move:
                        results.append(move)
                except Exception as e:
                    pass

        return results[0] if results else list(board.legal_moves)[0]

    def _alpha_beta(self, board: chess.Board, depth: int, alpha: int, beta: int,
                    ply: int, root: bool = False) -> Tuple[int, Optional[chess.Move]]:
        """Alpha-beta search with all optimizations."""
        self.nodes += 1

        if not root and time.time() - self.start_time > self.max_time * 0.95:
            self.stop_event.set()
            return 0, None

        if not root and board.can_claim_draw():
            return 0, None

        if depth == 0:
            return self._quiescence(board, alpha, beta), None

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            if board.is_checkmate():
                return -MATE_SCORE + ply, None
            return 0, None

        board_key = self._board_hash(board)
        tt_entry, tt_hit = self.tt.probe(board_key, depth, alpha, beta)

        if tt_hit and not root and tt_entry:
            if tt_entry.flag == 'exact':
                return tt_entry.score, tt_entry.move
            elif tt_entry.flag == 'lower':
                alpha = max(alpha, tt_entry.score)
            elif tt_entry.flag == 'upper':
                beta = min(beta, tt_entry.score)

            if alpha >= beta:
                return tt_entry.score, tt_entry.move

        if depth >= 3 and not board.is_check():
            board.push(chess.Move.null())
            null_score, _ = self._alpha_beta(board, depth - 1 - NULL_MOVE_REDUCTION, -beta, -beta + 1, ply + 1)
            null_score = -null_score
            board.pop()

            if null_score >= beta:
                return beta, None

        tt_move = tt_entry.move if tt_entry else None
        ordered_moves = order_moves(board, legal_moves, tt_move, self.history, self.killers, ply)

        best_move = None
        searched_moves = 0

        lmp_limit = LMP_LIMIT if depth <= LMP_DEPTH else 999

        for score, move in ordered_moves:
            if self.stop_event.is_set():
                break

            if board.is_capture(move) and see(board, move) < 0 and searched_moves > 4:
                continue

            if searched_moves >= lmp_limit and not board.is_capture(move) and not board.is_castling(move):
                continue

            searched_moves += 1

            board.push(move)

            if searched_moves == 1:
                score, _ = self._alpha_beta(board, depth - 1, -beta, -alpha, ply + 1)
            else:
                score, _ = self._alpha_beta(board, depth - 1, -alpha - 1, -alpha, ply + 1)
                if score > alpha and alpha < beta - 1:
                    score, _ = self._alpha_beta(board, depth - 1, -beta, -alpha, ply + 1)

            score = -score
            board.pop()

            if score > alpha:
                alpha = score
                best_move = move

                if not board.is_capture(move) and not board.is_castling(move):
                    self.killers.update(ply, move)

            if alpha >= beta:
                if not board.is_capture(move):
                    self.history.update(board.turn, move, depth)

                flag = 'exact' if alpha >= beta else 'upper'
                self.tt.store(board_key, alpha, depth, flag, best_move)
                return alpha, best_move

        flag = 'exact' if alpha >= beta else 'lower'
        self.tt.store(board_key, alpha, depth, flag, best_move)

        return alpha, best_move

    def _quiescence(self, board: chess.Board, alpha: int, beta: int) -> int:
        """Quiescence search to reach quiet positions."""
        self.nodes += 1

        stand_pat = self.evaluator.evaluate(board)
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        moves = [m for m in board.legal_moves if board.is_capture(m) or m.promotion]

        if not moves:
            return alpha

        ordered = []
        for move in moves:
            score = see(board, move)
            if move.promotion:
                score += PIECE_VALUES[move.promotion]
            ordered.append((score, move))

        ordered.sort(key=lambda x: x[0], reverse=True)

        for score, move in ordered:
            if score < -50:
                break

            board.push(move)
            q_score = -self._quiescence(board, -beta, -alpha)
            board.pop()

            if q_score >= beta:
                return beta
            if q_score > alpha:
                alpha = q_score

        return alpha

    def _board_hash(self, board: chess.Board) -> int:
        return hash(board.fen())

    def _report_info(self, depth: int, score: int, elapsed: float, move: Optional[chess.Move]):
        cp_score = score
        if abs(score) >= MATE_SCORE - 100:
            mate_in = (MATE_SCORE - abs(score)) // 2
            if score > 0:
                print(f"info depth {depth} score mate {mate_in} time {int(elapsed*1000)} nodes {self.nodes} nps {int(self.nodes/elapsed) if elapsed > 0 else 0} pv {move.uci() if move else ''}")
            else:
                print(f"info depth {depth} score mate -{mate_in} time {int(elapsed*1000)} nodes {self.nodes} nps {int(self.nodes/elapsed) if elapsed > 0 else 0} pv {move.uci() if move else ''}")
        else:
            print(f"info depth {depth} score cp {cp_score} time {int(elapsed*1000)} nodes {self.nodes} nps {int(self.nodes/elapsed) if elapsed > 0 else 0} pv {move.uci() if move else ''}")


def search_best_move(board: chess.Board, depth: int = 10, time_limit: float = 1.0) -> chess.Move:
    engine = Search()
    return engine.search(board, depth, time_limit)
