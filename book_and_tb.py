import chess
import chess.polyglot
import chess.syzygy
import os
from typing import Optional


class BookAndTB:
    """Opening book and endgame tablebase handler."""

    def __init__(self, book_path: Optional[str] = None, tb_path: Optional[str] = None):
        self.book_path = book_path or self.get_default_book_path()
        self.tb_path = tb_path or self.get_default_tb_path()
        self.book = self._load_book()
        self.tb = self._load_tb()

    def get_default_book_path(self) -> str:
        """Get default opening book path."""
        possible_paths = [
            "book.bin",
            os.path.expanduser("~/.local/share/chess/book.bin"),
            "/usr/share/chess/book.bin",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return "book.bin"

    def get_default_tb_path(self) -> str:
        """Get default tablebase path."""
        possible_paths = [
            "syzygy",
            os.path.expanduser("~/.local/share/chess/syzygy"),
            "/usr/share/chess/syzygy",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return "syzygy"

    def _load_book(self):
        """Load Polyglot opening book."""
        try:
            if os.path.exists(self.book_path):
                return chess.polyglot.open_reader(self.book_path)
        except Exception:
            pass
        return None

    def _load_tb(self):
        """Load Syzygy tablebase."""
        try:
            if os.path.exists(self.tb_path):
                return chess.syzygy.open_tablebase(self.tb_path)
        except Exception:
            pass
        return None

    def get_book_move(self, board: chess.Board) -> Optional[chess.Move]:
        """Get best move from opening book."""
        if not self.book:
            return None

        try:
            entry = self.book.find(board)
            if entry:
                return entry.move()
        except Exception:
            pass
        return None

    def get_best_move(self, board: chess.Board) -> Optional[chess.Move]:
        """Get best move from tablebase (endgame only)."""
        if not self.tb:
            return None

        try:
            # Only use TB in endgame positions (few pieces)
            piece_count = len(board.piece_map())
            if piece_count > 6:
                return None

            wdl = self.tb.get_wdl(board)
            if wdl:
                for move in board.legal_moves:
                    board.push(move)
                    wdl_after = self.tb.get_wdl(board)
                    board.pop()

                    if wdl_after:
                        # Find the best move (look for win or draw)
                        if wdl_after >= 0:
                            return move
        except Exception:
            pass
        return None


def get_default_book_path() -> str:
    """Get default opening book path."""
    possible_paths = [
        "book.bin",
        os.path.expanduser("~/.local/share/chess/book.bin"),
        "/usr/share/chess/book.bin",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return "book.bin"


def get_default_tb_path() -> str:
    """Get default tablebase path."""
    possible_paths = [
        "syzygy",
        os.path.expanduser("~/.local/share/chess/syzygy"),
        "/usr/share/chess/syzygy",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return "syzygy"
