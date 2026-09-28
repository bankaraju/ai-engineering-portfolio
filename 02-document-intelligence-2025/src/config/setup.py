"""Minimal stand-in for the original `config.setup` module.

The 2025 platform had a larger setup module (not included here) that built
the shared Postgres connection, Google API clients and the embedding model.
This stub provides only the names the published files use, reading every
setting from environment variables (see `.env.example`). Heavy clients are
created lazily on first use so that importing a module does not require a
running database or cloud credentials.
"""
import logging
import os

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("rag_platform")


def db_params() -> dict:
    """Postgres connection parameters from the environment."""
    return {
        "dbname": os.getenv("PGDATABASE", "annual_reports_db"),
        "user": os.getenv("PGUSER", "postgres"),
        "password": os.getenv("PGPASSWORD", ""),
        "host": os.getenv("PGHOST", "localhost"),
        "port": os.getenv("PGPORT", "5432"),
    }


class _LazyConnection:
    """Opens the psycopg2 connection on first attribute access."""

    def __init__(self):
        self._conn = None

    def __getattr__(self, name):
        if self._conn is None:
            import psycopg2
            self._conn = psycopg2.connect(**db_params())
        return getattr(self._conn, name)


class _Settings:
    def __init__(self):
        self.conn = _LazyConnection()
        self.audit_logger = logging.getLogger("rag_platform.audit")
        # projects/<project>/locations/<location>/processors/<processor_id>
        self.processor_name = os.getenv("DOCAI_PROCESSOR_NAME", "")
        self._drive_service = None
        self._docai_client = None
        self._embed_model = None

    @property
    def drive_service(self):
        if self._drive_service is None:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build
            creds = service_account.Credentials.from_service_account_file(
                os.getenv("GOOGLE_APPLICATION_CREDENTIALS", ""),
                scopes=["https://www.googleapis.com/auth/drive.readonly"],
            )
            self._drive_service = build("drive", "v3", credentials=creds)
        return self._drive_service

    @property
    def docai_client(self):
        if self._docai_client is None:
            from google.cloud import documentai_v1 as documentai
            self._docai_client = documentai.DocumentProcessorServiceClient()
        return self._docai_client

    @property
    def embed_model(self):
        if self._embed_model is None:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            self._embed_model = HuggingFaceEmbedding(
                model_name=os.getenv("EMBED_MODEL", "BAAI/bge-large-en-v1.5")
            )
        return self._embed_model


config = _Settings()
