# chess-engine

This project is a simple chess engine written in Python.  
I built this as a learning project to understand how chess engines work and to try creating one on my own.

The engine supports basic position evaluation, move searching, and can be connected to chess GUIs using the UCI protocol.

---

## Features

- Chess position evaluation using material values and piece-square tables
- Move search using alpha-beta pruning
- Basic move ordering for better search results
- Opening book support (Polyglot format)
- Endgame tablebase support (Syzygy)
- UCI protocol support (can run in most chess GUIs)

---

## Requirements

- Python 3.7+
- `python-chess` library

Install dependencies:
```bash
pip install python-chess
