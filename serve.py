"""Production-style launcher: ONE process, several threads (waitress).

Run:  python serve.py

Why one process? Every worker process would load its own copy of the Facenet
model (hundreds of MB of RAM) and its own embedding cache, and the in-memory
session state (busy flags, rate limits, "already marked" sets) would no longer
be shared. Threads share all of that, and the inference lock keeps TensorFlow
safe. Face recognition is CPU bound, so extra processes would not make a
laptop faster anyway.
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
