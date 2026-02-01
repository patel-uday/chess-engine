import sys
import chess
from search import Search, MATE_SCORE
from book_and_tb import BookAndTB, get_default_book_path, get_default_tb_path

class UCIEngine:
    """UCI chess engine interface."""

    def __init__(self):
        self.board = chess.Board()
        self.search = Search()
        self.depth = 10
        self.time_limit = 1.0
        self.book_tb = BookAndTB()
        self.use_book = True
        self.use_tb = True
        self.num_threads = 1

    def run(self):
        """Main UCI loop."""
        print("id name PyChessEngine")
        print("id author Sisyphus")
        print("option name Hash type spin default 64 min 1 max 1024")
        print("option name Depth type spin default 10 min 1 max 20")
        print("option name Threads type spin default 1 min 1 max 16")
        print("option name OwnBook type check default true")
        print("option name UseBook type check default true")
        print("option name UseTB type check default true")
        print("uciok")
        sys.stdout.flush()

        while True:
            line = sys.stdin.readline().strip()
            if not line:
                continue

            self._handle_command(line)

    def _handle_command(self, command: str):
        """Handle UCI command."""
        parts = command.split()
        cmd = parts[0].lower()

        if cmd == "uci":
            self._cmd_uci()
        elif cmd == "isready":
            self._cmd_isready()
        elif cmd == "ucinewgame":
            self._cmd_ucinewgame()
        elif cmd == "position":
            self._cmd_position(parts[1:])
        elif cmd == "go":
            self._cmd_go(parts[1:])
        elif cmd == "stop":
            self._cmd_stop()
        elif cmd == "setoption":
            self._cmd_setoption(parts[1:])
        elif cmd == "quit":
            sys.exit(0)

    def _cmd_uci(self):
        """Handle uci command."""
        print("id name PyChessEngine")
        print("id author Sisyphus")
        print("uciok")
        sys.stdout.flush()

    def _cmd_isready(self):
        """Handle isready command."""
        print("readyok")
        sys.stdout.flush()

    def _cmd_ucinewgame(self):
        """Handle ucinewgame command."""
        self.board = chess.Board()
        self.search = Search()
        self.book_tb = BookAndTB()

    def _cmd_position(self, args: list):
        """Handle position command."""
        if not args:
            return

        if args[0] == "startpos":
            self.board = chess.Board()
            args = args[1:]
        elif args[0] == "fen":
            fen_str = " ".join(args[1:7])
            self.board.set_fen(fen_str)
            args = args[7:]

        if args and args[0] == "moves":
            for move_str in args[1:]:
                try:
                    move = self.board.parse_uci(move_str)
                    self.board.push(move)
                except ValueError:
                    pass

    def _cmd_go(self, args: list):
        """Handle go command."""
        self.search.stop_event.clear()

        depth = self.depth
        time_limit = self.time_limit

        for i in range(0, len(args), 2):
            if i + 1 >= len(args):
                break

            param = args[i]
            value = args[i + 1]

            if param == "depth":
                depth = int(value)
            elif param == "movetime":
                time_limit = int(value) / 1000.0
            elif param == "wtime":
                white_time = int(value) / 1000.0
                if self.board.turn == chess.WHITE:
                    time_limit = min(time_limit, white_time / 40)
            elif param == "btime":
                black_time = int(value) / 1000.0
                if self.board.turn == chess.BLACK:
                    time_limit = min(time_limit, black_time / 40)
            elif param == "movestogo":
                time_limit = time_limit / max(1, int(value))
            elif param == "infinite":
                time_limit = 999999.0

        legal_moves = list(self.board.legal_moves)

        if len(legal_moves) == 0:
            print("bestmove (none)")
            sys.stdout.flush()
            return

        if self.use_book:
            book_move = self.book_tb.get_book_move(self.board)
            if book_move:
                print(f"bestmove {book_move.uci()}")
                sys.stdout.flush()
                return

        if self.use_tb:
            tb_move = self.book_tb.get_best_move(self.board)
            if tb_move:
                print(f"bestmove {tb_move.uci()}")
                sys.stdout.flush()
                return

        best_move = self.search.search(self.board, depth, time_limit, self.num_threads)

        if best_move:
            print(f"bestmove {best_move.uci()}")
        else:
            print("bestmove (none)")

        sys.stdout.flush()

    def _cmd_stop(self):
        """Handle stop command."""
        self.search.stop_event.set()

    def _cmd_setoption(self, args: list):
        """Handle setoption command."""
        if len(args) < 4 or args[0].lower() != "name":
            return

        name = args[1]
        if args[2].lower() == "value":
            value = args[3]

            if name == "Hash":
                self.search.tt = Search().tt
            elif name == "Depth":
                self.depth = int(value)
            elif name == "Threads":
                self.num_threads = int(value)
            elif name == "UseBook":
                self.use_book = value.lower() == "true"
            elif name == "UseTB":
                self.use_tb = value.lower() == "true"

        print("readyok")
        sys.stdout.flush()


def main():
    """Main entry point."""
    engine = UCIEngine()
    try:
        engine.run()
    except KeyboardInterrupt:
        print("\nquit")
        sys.stdout.flush()
        sys.exit(0)


if __name__ == "__main__":
    main()
