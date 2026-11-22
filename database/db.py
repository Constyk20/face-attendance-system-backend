"""Single PyMongo connection used by the rest of the backend."""

import logging
import threading

from pymongo import MongoClient
from pymongo.errors import PyMongoError

from config import Config

log = logging.getLogger(__name__)

_client = None
_client_lock = threading.Lock()


def get_client():
    """Create the MongoClient once and reuse it (it is thread-safe and pools sockets)."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = MongoClient(
                    Config.MONGODB_URI,
                    appName=Config.MONGODB_APP_NAME,
                    maxPoolSize=Config.MONGODB_MAX_POOL_SIZE,
                    minPoolSize=0,
                    connectTimeoutMS=Config.MONGODB_CONNECT_TIMEOUT_MS,
                    serverSelectionTimeoutMS=Config.MONGODB_SERVER_SELECTION_TIMEOUT_MS,
                    socketTimeoutMS=Config.MONGODB_SOCKET_TIMEOUT_MS,
                    retryWrites=True,
                )
    return _client


def get_db():
    """Return the application database."""
    return get_client()[Config.MONGODB_DB_NAME]


def ping_database():
    """Return (ok, message). Used by the health endpoint."""
    try:
        get_client().admin.command("ping")
        return True, "connected"
    except PyMongoError:
        # The real error can contain host names, so it only goes to the log.
        log.warning("Database ping failed", exc_info=True)
        return False, "database unreachable"
