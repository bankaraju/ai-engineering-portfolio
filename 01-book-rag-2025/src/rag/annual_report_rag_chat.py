#!/usr/bin/env python3
"""
Annual-report RAG chat service (May 2025).

The book-RAG service was pointed at bank annual reports / earnings material in
late May 2025. This is a trimmed copy of that service's core path:

    JSON documents -> filename metadata + content flags -> SentenceSplitter
    chunking -> BGE embeddings in pgvector -> retrieval -> query-type filtering
    -> Claude answer -> rule-based follow-up questions

Trimmed from the original (~1,200 lines): the embedded HTML/CSS front end,
live market-data lookups, a hard-coded bank universe and a canned demo-mode
answer were removed. Database, data path, API key and model ids are read from
environment variables (see .env.example). Requires PostgreSQL with pgvector and
an ANTHROPIC_API_KEY; it does not run offline.
"""

import os
import re
import json
import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

# LlamaIndex core
from llama_index.core import (
    VectorStoreIndex,
    Document,
    StorageContext,
    Settings
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import anthropic

# Database
import psycopg2
from psycopg2.extras import RealDictCursor

# FastAPI
from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Config:
    # Database
    DB_HOST = os.getenv("PGHOST", "localhost")
    DB_PORT = int(os.getenv("PGPORT", "5432"))
    DB_NAME = os.getenv("PGDATABASE", "annual_reports_db")
    DB_USER = os.getenv("PGUSER", "postgres")
    DB_PASSWORD = os.getenv("PGPASSWORD", "")

    # Data: a directory of pre-extracted JSON documents for one issuer
    DATA_PATH = os.getenv("REPORTS_DATA_DIR", "./data/reports")
    BANK_NAME = os.getenv("REPORTS_ISSUER_NAME", "Example Bank")

    # API Keys
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

    # Models
    BGE_MODEL = os.getenv("RAG_EMBED_MODEL", "BAAI/bge-large-en-v1.5")
    EMBED_DIM = int(os.getenv("RAG_EMBED_DIM", "1024"))
    CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "your-claude-model-id")


class CleanBankingPlatform:
    """Retrieval + Claude analysis over a bank's reports."""

    def __init__(self):
        self.setup_complete = False
        self.index = None
        self.anthropic_client = None

    async def initialize(self):
        logger.info("Initializing platform")
        await self._setup_database()
        await self._setup_llamaindex()
        await self._setup_vector_store()
        await self._process_banking_documents()
        self.setup_complete = True
        logger.info("Platform ready")

    async def _setup_database(self):
        """Setup PostgreSQL with pgvector"""
        self.db_connection = psycopg2.connect(
            host=Config.DB_HOST,
            port=Config.DB_PORT,
            database=Config.DB_NAME,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD,
            cursor_factory=RealDictCursor
        )
        with self.db_connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            self.db_connection.commit()
        logger.info("Database ready")

    async def _setup_llamaindex(self):
        """Configure LlamaIndex settings"""
        self.embed_model = HuggingFaceEmbedding(
            model_name=Config.BGE_MODEL,
            trust_remote_code=True
        )
        if not Config.ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self.anthropic_client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)

        Settings.embed_model = self.embed_model
        Settings.chunk_size = 1024
        Settings.chunk_overlap = 100
        logger.info("LlamaIndex configured")

    async def _setup_vector_store(self):
        """Setup pgvector store (this service used an HNSW index)"""
        self.vector_store = PGVectorStore.from_params(
            database=Config.DB_NAME,
            host=Config.DB_HOST,
            password=Config.DB_PASSWORD,
            port=Config.DB_PORT,
            user=Config.DB_USER,
            table_name="clean_banking_embeddings",
            embed_dim=Config.EMBED_DIM,
            hnsw_kwargs={
                "hnsw_m": 16,
                "hnsw_ef_construction": 64,
                "hnsw_ef_search": 40,
            }
        )
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        logger.info("Vector store ready")

    async def _process_banking_documents(self):
        """Load documents, chunk with SentenceSplitter and index"""
        files = list(Path(Config.DATA_PATH).glob("*.json"))
        logger.info(f"Found {len(files)} documents in {Config.DATA_PATH}")
        if not files:
            raise RuntimeError(f"No documents found in {Config.DATA_PATH}")

        documents = []
        for file_path in files:
            doc = await self._load_banking_document(file_path, Config.BANK_NAME)
            if doc:
                documents.append(doc)
        if not documents:
            raise RuntimeError("No valid documents loaded")
        logger.info(f"Loaded {len(documents)} documents")

        node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
        self.index = VectorStoreIndex.from_documents(
            documents,
            storage_context=self.storage_context,
            transformations=[node_parser],
            show_progress=True
        )
        logger.info("Documents indexed")

    async def _load_banking_document(self, file_path: Path, bank_name: str) -> Optional[Document]:
        """Load a document with metadata derived from filename and content"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            text_content = self._extract_text_from_json(data)
            if not text_content or len(text_content) < 100:
                return None

            metadata = {
                "bank_name": bank_name,
                "filename": file_path.name,
                "document_type": self._determine_doc_type(file_path.name),
                "quarter": self._extract_quarter(file_path.name),
                "year": self._extract_year(file_path.name),
                "has_management_commentary": self._detect_management_content(text_content),
                "has_guidance": self._detect_guidance_content(text_content),
                "has_financial_metrics": self._detect_financial_metrics(text_content),
                "processed_date": datetime.now().isoformat(),
                "text_length": len(text_content)
            }
            return Document(text=text_content, metadata=metadata)

        except Exception as e:
            logger.error(f"Failed to load {file_path.name}: {str(e)}")
            return None

    def _extract_text_from_json(self, data: Dict) -> str:
        """Extract text with priority for management content"""
        text_parts = []
        if isinstance(data, dict):
            priority_fields = ["combined_text", "text", "content", "management_commentary"]
            for field in priority_fields:
                if field in data and isinstance(data[field], str) and len(data[field]) > 100:
                    text_parts.append(data[field])

            if "chunks_processed" in data and isinstance(data["chunks_processed"], list):
                for chunk in data["chunks_processed"]:
                    if isinstance(chunk, dict) and "text" in chunk:
                        text_parts.append(str(chunk["text"]))
        return "\n\n".join(text_parts)

    def _determine_doc_type(self, filename: str) -> str:
        filename_lower = filename.lower()
        if "earnings" in filename_lower or "call" in filename_lower:
            return "Earnings Call"
        elif "presentation" in filename_lower:
            return "Investor Presentation"
        elif "annual" in filename_lower:
            return "Annual Report"
        return "Financial Document"

    def _extract_quarter(self, filename: str) -> Optional[str]:
        match = re.search(r'Q(\d)', filename)
        return f"Q{match.group(1)}" if match else None

    def _extract_year(self, filename: str) -> Optional[str]:
        match = re.search(r'FY(\d{2,4})', filename)
        return f"FY{match.group(1)}" if match else None

    def _detect_management_content(self, text: str) -> bool:
        management_keywords = [
            "management", "ceo", "md", "director", "guidance", "outlook",
            "strategy", "vision", "execution", "leadership", "commentary"
        ]
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in management_keywords)

    def _detect_guidance_content(self, text: str) -> bool:
        guidance_keywords = [
            "guidance", "outlook", "forecast", "expect", "target", "projection",
            "fy25", "next year", "going forward", "future"
        ]
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in guidance_keywords)

    def _detect_financial_metrics(self, text: str) -> bool:
        return any(term in text.lower() for term in [
            "nii", "nim", "roe", "roa", "casa", "credit growth", "provision",
            "npa", "tier 1", "capital adequacy"
        ])

    async def analyze_query(self, query: str, analysis_mode: str = "comprehensive") -> Dict[str, Any]:
        """Retrieve, filter by query type, and answer with Claude"""
        if not self.setup_complete or not self.index:
            raise RuntimeError("Platform not ready")

        logger.info(f"Analyzing: {query} (mode: {analysis_mode})")
        retriever = self.index.as_retriever(similarity_top_k=5)
        relevant_nodes = await retriever.aretrieve(query)

        filtered_nodes = self._filter_nodes_by_query_type(query, relevant_nodes)
        context_chunks = [node.text for node in filtered_nodes]
        context_text = "\n\n---\n\n".join(context_chunks[:3])
        sources = self._prepare_sources(filtered_nodes)

        response = await self._generate_claude_analysis(query, context_text, analysis_mode)
        return {
            "response": response,
            "sources": sources,
            "analysis_mode": analysis_mode,
            "query": query,
            "timestamp": datetime.now().isoformat()
        }

    def _filter_nodes_by_query_type(self, query: str, nodes) -> list:
        """Prefer chunks whose metadata flags match the query type"""
        query_lower = query.lower()
        if any(term in query_lower for term in ["management", "guidance", "leadership", "strategy"]):
            management_nodes = [n for n in nodes if n.metadata.get("has_management_commentary", False)]
            if management_nodes:
                return management_nodes
        elif any(term in query_lower for term in ["nii", "growth", "margin", "ratio"]):
            financial_nodes = [n for n in nodes if n.metadata.get("has_financial_metrics", False)]
            if financial_nodes:
                return financial_nodes
        return nodes

    def _prepare_sources(self, nodes) -> List[Dict]:
        sources = []
        for node in nodes:
            sources.append({
                "document": node.metadata.get("document_type", "Unknown"),
                "filename": node.metadata.get("filename"),
                "quarter": node.metadata.get("quarter", "N/A"),
                "year": node.metadata.get("year", "N/A"),
                "bank": node.metadata.get("bank_name", "Unknown"),
                "has_management": node.metadata.get("has_management_commentary", False),
                "has_guidance": node.metadata.get("has_guidance", False),
                "relevance": round(node.score, 3) if getattr(node, "score", None) is not None else None
            })
        return sources

    async def _generate_claude_analysis(self, query: str, context: str, mode: str) -> str:
        """Generate the answer with Claude from the retrieved context"""
        management_prompt = f"""
        You are an investment analyst covering the Indian banking sector, with a focus on
        management assessment and strategic analysis.

        Query: {query}

        Banking Data Context:
        {context}

        Provide analysis covering:
        1. MANAGEMENT SCORECARD - guidance vs delivery, execution, transparency
        2. FINANCIAL PERFORMANCE - key metrics with numbers, QoQ and YoY trends
        3. STRATEGIC POSITIONING - competitive position, digital progress, risk management
        4. INVESTMENT PERSPECTIVE - key risks and catalysts

        Use only data points present in the context and say when the context is insufficient.
        """
        try:
            response = self.anthropic_client.messages.create(
                model=Config.CLAUDE_MODEL,
                max_tokens=2500,
                messages=[{"role": "user", "content": management_prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Claude API error: {str(e)}")
            return f"Analysis unavailable: {str(e)}"

    def generate_follow_up_questions(self, query: str, response: str) -> List[str]:
        """Rule-based contextual follow-ups"""
        follow_ups = []
        query_lower = query.lower()
        if any(term in query_lower for term in ["management", "guidance", "leadership"]):
            follow_ups.extend([
                "Analyze management's track record of guidance delivery vs actuals",
                "Compare leadership effectiveness across peer banks",
                "Evaluate strategic execution capabilities and succession planning"
            ])
        elif any(term in query_lower for term in ["growth", "performance", "financial"]):
            follow_ups.extend([
                "What is management's outlook and forward guidance?",
                "How does leadership assess competitive positioning?",
                "Analyze management commentary on strategic priorities"
            ])
        else:
            follow_ups.extend([
                "Assess management scorecard and leadership effectiveness",
                "Analyze strategic vision and execution track record",
                "Compare management quality vs peer banks"
            ])
        return follow_ups[:3]


# FastAPI application
app = FastAPI(title="Annual Report RAG Chat", version="2.0.0")
platform: Optional[CleanBankingPlatform] = None


async def initialize_platform() -> bool:
    global platform
    try:
        platform = CleanBankingPlatform()
        await platform.initialize()
        return True
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        platform = None
        return False


class ChatRequest(BaseModel):
    message: str
    session_id: str
    analysis_mode: str = "comprehensive"


@app.post("/api/banking/chat")
async def banking_chat(request: ChatRequest):
    global platform
    if not platform and not await initialize_platform():
        return {"response": "Platform initialization failed. Check logs.",
                "follow_up_questions": [], "sources": [], "analysis_mode": "error"}
    try:
        result = await platform.analyze_query(request.message, request.analysis_mode)
        follow_ups = platform.generate_follow_up_questions(request.message, result["response"])
        return {
            "response": result["response"],
            "follow_up_questions": follow_ups,
            "sources": result["sources"],
            "analysis_mode": result["analysis_mode"],
            "timestamp": result["timestamp"]
        }
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return {"response": f"Analysis error: {str(e)}", "follow_up_questions": [],
                "sources": [], "analysis_mode": "error"}


@app.get("/api/banking/status")
async def platform_status():
    return {"ready": bool(platform and platform.setup_complete)}


async def _cli(question: str):
    if not await initialize_platform():
        raise SystemExit(1)
    result = await platform.analyze_query(question)
    print(result["response"])
    for s in result["sources"]:
        print("source:", s)
    for q in platform.generate_follow_up_questions(question, result["response"]):
        print("follow-up:", q)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Annual-report RAG chat")
    parser.add_argument("--ask", help="Ask one question from the CLI instead of serving")
    parser.add_argument("--port", type=int, default=int(os.getenv("RAG_PORT", "8002")))
    args = parser.parse_args()
    if args.ask:
        asyncio.run(_cli(args.ask))
    else:
        import uvicorn
        uvicorn.run(app, host=os.getenv("RAG_HOST", "127.0.0.1"), port=args.port, log_level="info")
