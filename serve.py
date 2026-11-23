"""Local / Windows launcher: ONE process, several threads (waitress).

Run:  python serve.py        (on Render the start command uses gunicorn instead)

One process keeps the in-memory face index and the per-session state
(busy flags, rate limits, "already marked" sets) shared between threads.
There is no model in this process any more, so it is small and fast to start.
"""

import logging

from waitress import serve

from app import app
from config import Config

if __name__ == "__main__":
    logging.getLogger("waitress").setLevel(logging.INFO)
    serve(
        app,
        host=Config.HOST,
        port=Config.PORT,
        threads=Config.WAITRESS_THREADS,
        max_request_body_size=Config.MAX_CONTENT_LENGTH,
    )
