"""
Annual Reports RAG System - LlamaIndex + PostgreSQL + local models via Ollama
(BGE-M3 embeddings, DeepSeek-R1 distill for answers). May 2025 scaffold:
ingestion and query paths only, no evaluation. Settings from environment.
"""
import os
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
import logging

class AnnualReportsRAG:
    def __init__(self, 
                 db_host=None, 
                 db_name=None,
                 db_user=None, 
                 db_password=None,
                 db_port=None):
        
        # Database connection (arguments override environment)
        self.db = {
            "host": db_host or os.getenv("PGHOST", "localhost"),
            "database": db_name or os.getenv("PGDATABASE", "annual_reports_db"),
            "user": db_user or os.getenv("PGUSER", "postgres"),
            "password": db_password if db_password is not None else os.getenv("PGPASSWORD", ""),
            "port": int(db_port or os.getenv("PGPORT", "5432")),
        }
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        # Initialize local models via Ollama
        self.setup_local_models()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize vector store
        self.setup_vector_store()
    
    def setup_local_models(self):
        """Setup local models for embeddings and LLM"""
        # Use BGE-M3 for embeddings (best for financial docs)
        self.embed_model = OllamaEmbedding(
            model_name=os.getenv("OLLAMA_EMBED_MODEL", "bge-m3"),
            base_url=self.ollama_url
        )
        
        # Use DeepSeek for reasoning (excellent for financial analysis)
        self.llm = Ollama(
            model=os.getenv("OLLAMA_LLM_MODEL", "deepseek-r1-distill-llama-8b"),
            base_url=self.ollama_url,
            temperature=0.1
        )
        
        # Configure LlamaIndex settings
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm
    
    def setup_vector_store(self):
        """Setup PostgreSQL vector store"""
        self.vector_store = PGVectorStore.from_params(
            database=self.db["database"],
            host=self.db["host"],
            password=self.db["password"],
            port=self.db["port"],
            user=self.db["user"],
            table_name="annual_reports_embeddings",
            embed_dim=1024  # BGE-M3 dimension
        )
    
    def add_annual_report(self, text_content, metadata):
        """Add annual report content to vector store"""
        # Create document with metadata
        document = Document(
            text=text_content,
            metadata=metadata  # company, year, section, etc.
        )
        
        # Create index and add document
        index = VectorStoreIndex.from_vector_store(self.vector_store)
        index.insert(document)
        
        self.logger.info(f"Added annual report: {metadata}")
    
    def query_reports(self, query, filters=None):
        """Query annual reports with natural language"""
        index = VectorStoreIndex.from_vector_store(self.vector_store)
        query_engine = index.as_query_engine()
        
        response = query_engine.query(query)
        return response

if __name__ == "__main__":
    rag = AnnualReportsRAG()
    print("Annual Reports RAG system initialized!")
