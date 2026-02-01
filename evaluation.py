import chess
from typing import Dict, List

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

PAWN_TABLE = [
    0,  0,  0,  0,  0,  0,  0,  0,
    80, 80, 80, 80, 80, 80, 80, 80,
    20, 20, 30, 40, 40, 30, 20, 20,
    5,  5, 10, 25, 25, 10,  5,  5,
    0,  0,  0, 20, 20,  0,  0,  0,
    5, -5,-10,  0,  0,-10, -5,  5,
    5, 10, 10,-20,-20, 10, 10,  5,
    0,  0,  0,  0,  0,  0,  0,  0
]

KNIGHT_TABLE = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  5,  5,  0,-40,
    -30,  0, 10, 20, 20, 15,  0,-30,
    -30,  5, 15, 25, 25, 15,  5,-30,
    -30,  5, 20, 25, 25, 20,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
    -50,-40,-30,-30,-30,-30,-40,-50
]

BISHOP_TABLE = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  5,  5,  5,  0,  0,-10,
    -10,  5, 10, 15, 15, 10,  5,-10,
    -10,  5, 10, 15, 20, 15, 10,  5,
    -10,  5, 15, 20, 20, 15, 10,  5,
    -10,  5, 15, 20, 20, 15,  5,  0,
    -10,  5, 10, 15, 10,  5,  0,  5,
    -20,-10,-10,-10,-10,-10,-10,-20
]

ROOK_TABLE = [
    0,  0,  0,  0,  0,  0,  0,  0,
    5, 10, 15, 20, 20, 15, 10,  5,
    -5,  5, 10, 15, 15, 10,  5, -5,
    -5,  5, 10, 15, 15, 10,  5, -5,
    -5,  5, 10, 15, 15, 10,  5, -5,
    -5,  5, 10, 15, 15, 10,  5, -5,
    -5,  5, 10, 15, 15, 10,  5, -5,
    0,  0,  0,  5,  5,  0,  0,  0
]

QUEEN_TABLE = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  5,  5,  0,  0,-10,
    -10,  5, 10, 10,  5,  0,  0,-10,
    -5,  0,  5,  5,  5,  0, -5, -5,
    0,  0,  5, 10, 10,  5,  0, -5,
    -10,  5, 10, 10, 10,  5,  0,  0,
    -10,  0,  5,  0,  0,  0,  0,  0,
    -20,-10,-10, -5, -5,-10,-10,-20
]

KING_TABLE_MG = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20
]

KING_TABLE_EG = [
    -50,-40,-30,-20,-10,-20,-30,-40,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 40, 30, 20,-10,
    -30,-10, 20, 40, 50, 40, 30, 20,
    -30,-10, 20, 40, 50, 40, 30, 20,
    -30,-20,  0,  0,  0,  0,-20,-30,
    -50,-30,-30,-30,-30,-30,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-30
]

PIECE_SQUARE_TABLES = {
    chess.PAWN: PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
    chess.ROOK: ROOK_TABLE,
    chess.QUEEN: QUEEN_TABLE
}

BISHOP_PAIR_BONUS = 30
KING_SAFETY_BONUS = 20
DOUBLED_PAWN_PENALTY = 10
ISOLATED_PAWN_PENALTY = 20
BACKWARD_PAWN_PENALTY = 10
PASSED_PAWN_BONUS = 20
CONNECTED_ROOKS_BONUS = 20
ROOK_ON_SEVENTH_BONUS = 20
MOBILITY_BONUS = 4


class Evaluator:
    """Advanced chess position evaluator with multiple features."""

    def __init__(self):
        self.game_phase_weight = 0.0

    def evaluate(self, board: chess.Board) -> int:
        """
        Evaluate a chess position.
        Returns positive for white advantage, negative for black advantage.
        """
        if board.is_checkmate():
            return -100000 + (board.fullmove_number * 10) if board.turn else 100000 - (board.fullmove_number * 10)
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            return 0

        self.game_phase_weight = self._calculate_game_phase(board)

        score = self._evaluate_material(board)
        score += self._evaluate_position(board)
        score += self._evaluate_king_safety(board)
        score += self._evaluate_pawn_structure(board)
        score += self._evaluate_mobility(board)
        score += self._evaluate_piece_coordination(board)

        return score

    def _calculate_game_phase(self, board: chess.Board) -> float:
        """Calculate game phase from 0 (opening) to 1 (endgame)."""
        total_material = 0
        max_material = 3950

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type != chess.KING:
                    total_material += PIECE_VALUES[piece.piece_type]

        return max(0, 1 - (total_material / max_material))

    def _evaluate_material(self, board: chess.Board) -> int:
        """Evaluate material difference."""
        white_material = 0
        black_material = 0

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = PIECE_VALUES[piece.piece_type]
                if piece.color == chess.WHITE:
                    white_material += value
                else:
                    black_material += value

        return white_material - black_material

    def _evaluate_position(self, board: chess.Board) -> int:
        """Evaluate position using piece-square tables."""
        score = 0

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type in PIECE_SQUARE_TABLES:
                table = PIECE_SQUARE_TABLES[piece.piece_type]

                if piece.color == chess.WHITE:
                    score += table[square]
                else:
                    score -= table[chess.square_mirror(square)]

        score += self._evaluate_king_position(board)

        return score

    def _evaluate_king_position(self, board: chess.Board) -> int:
        """Evaluate king position based on game phase."""
        score = 0

        table = [
            int(KING_TABLE_MG[i] * (1 - self.game_phase_weight) + KING_TABLE_EG[i] * self.game_phase_weight)
            for i in range(64)
        ]

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.KING:
                if piece.color == chess.WHITE:
                    score += table[square]
                else:
                    score -= table[chess.square_mirror(square)]

        return score

    def _evaluate_king_safety(self, board: chess.Board) -> int:
        """Evaluate king safety."""
        score = 0

        for color in [chess.WHITE, chess.BLACK]:
            king_square = board.king(color)
            if king_square is None:
                continue

            pawn_shield_score = 0
            king_file = chess.square_file(king_square)
            king_rank = chess.square_rank(king_square)

            for file_offset in [-1, 0, 1]:
                file = max(0, min(7, king_file + file_offset))
                if color == chess.WHITE:
                    rank = king_rank + 1
                    if rank < 8:
                        square = chess.square(file, rank)
                        pawn = board.piece_at(square)
                        if pawn and pawn.piece_type == chess.PAWN and pawn.color == color:
                            pawn_shield_score += KING_SAFETY_BONUS
                else:
                    rank = king_rank - 1
                    if rank >= 0:
                        square = chess.square(file, rank)
                        pawn = board.piece_at(square)
                        if pawn and pawn.piece_type == chess.PAWN and pawn.color == color:
                            pawn_shield_score += KING_SAFETY_BONUS

            if color == chess.WHITE:
                score += pawn_shield_score
            else:
                score -= pawn_shield_score

        return score

    def _evaluate_pawn_structure(self, board: chess.Board) -> int:
        """Evaluate pawn structure."""
        score = 0

        white_pawns = [[] for _ in range(8)]
        black_pawns = [[] for _ in range(8)]

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN:
                file = chess.square_file(square)
                rank = chess.square_rank(square)
                if piece.color == chess.WHITE:
                    white_pawns[file].append(rank)
                else:
                    black_pawns[file].append(rank)

        for file in range(8):
            if len(white_pawns[file]) > 1:
                score -= DOUBLED_PAWN_PENALTY * (len(white_pawns[file]) - 1)
            if len(black_pawns[file]) > 1:
                score += DOUBLED_PAWN_PENALTY * (len(black_pawns[file]) - 1)

            white_has_pawn = len(white_pawns[file]) > 0
            left_has_pawn = file > 0 and len(white_pawns[file - 1]) > 0
            right_has_pawn = file < 7 and len(white_pawns[file + 1]) > 0

            if white_has_pawn and not left_has_pawn and not right_has_pawn:
                score -= ISOLATED_PAWN_PENALTY

            black_has_pawn = len(black_pawns[file]) > 0
            left_has_pawn = file > 0 and len(black_pawns[file - 1]) > 0
            right_has_pawn = file < 7 and len(black_pawns[file + 1]) > 0

            if black_has_pawn and not left_has_pawn and not right_has_pawn:
                score += ISOLATED_PAWN_PENALTY

        return score

    def _evaluate_mobility(self, board: chess.Board) -> int:
        """Evaluate piece mobility."""
        score = 0
        prev_turn = board.turn

        board.turn = chess.WHITE
        white_mobility = len(list(board.legal_moves))

        board.turn = chess.BLACK
        black_mobility = len(list(board.legal_moves))

        board.turn = prev_turn

        return (white_mobility - black_mobility) * MOBILITY_BONUS

    def _evaluate_piece_coordination(self, board: chess.Board) -> int:
        """Evaluate piece coordination and special patterns."""
        score = 0

        white_bishops = 0
        black_bishops = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.BISHOP:
                if piece.color == chess.WHITE:
                    white_bishops += 1
                else:
                    black_bishops += 1

        if white_bishops >= 2:
            score += BISHOP_PAIR_BONUS
        if black_bishops >= 2:
            score -= BISHOP_PAIR_BONUS

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.ROOK:
                rank = chess.square_rank(square)
                if piece.color == chess.WHITE and rank == 6:
                    score += ROOK_ON_SEVENTH_BONUS
                elif piece.color == chess.BLACK and rank == 1:
                    score -= ROOK_ON_SEVENTH_BONUS

        for square_a in chess.SQUARES:
            for square_b in chess.SQUARES:
                if square_a >= square_b:
                    continue
                piece_a = board.piece_at(square_a)
                piece_b = board.piece_at(square_b)

                if piece_a and piece_b and piece_a.piece_type == chess.ROOK and piece_b.piece_type == chess.ROOK:
                    if piece_a.color == piece_b.color:
                        if chess.square_file(square_a) == chess.square_file(square_b):
                            clear = True
                            start = min(square_a, square_b) + 8
                            end = max(square_a, square_b)
                            for sq in range(start, end, 8):
                                if board.piece_at(sq):
                                    clear = False
                                    break
                            if clear:
                                if piece_a.color == chess.WHITE:
                                    score += CONNECTED_ROOKS_BONUS
                                else:
                                    score -= CONNECTED_ROOKS_BONUS

        return score


def evaluate(board: chess.Board) -> int:
    """Convenience function for position evaluation."""
    evaluator = Evaluator()
    return evaluator.evaluate(board)
