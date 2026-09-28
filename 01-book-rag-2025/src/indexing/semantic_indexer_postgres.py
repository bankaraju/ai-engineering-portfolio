#!/usr/bin/env python3
"""
Semantic indexing for the book corpus using PostgreSQL + pgvector.

BGE-small embeddings (384-d) via LlamaIndex, stored in a pgvector table with an
ivfflat cosine index. Connection settings come from environment variables
(see .env.example); content directories are passed on the command line.
"""

import psycopg2
import psycopg2.extras
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
import json
import os

class PostgreSQLSemanticIndexer:
    def __init__(self, data_paths=None):
        print("Initializing PostgreSQL Semantic Indexer...")
        
        # Database connection (from environment)
        self.db_params = {
            "host": os.getenv("PGHOST", "localhost"),
            "database": os.getenv("PGDATABASE", "book_rag"),
            "user": os.getenv("PGUSER", "postgres"),
            "password": os.getenv("PGPASSWORD", ""),
            "port": int(os.getenv("PGPORT", "5432")),
        }
        self.data_paths = data_paths or []
        
        # Use local embeddings
        print("Loading HuggingFace embedding model...")
        self.embed_model = HuggingFaceEmbedding(
            model_name=os.getenv("BOOK_EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
        Settings.embed_model = self.embed_model
        
        # Create vector store table
        self.setup_vector_table()
        
        print("✅ PostgreSQL Indexer initialized successfully!")
        
    def setup_vector_table(self):
        """Create vector storage table in PostgreSQL"""
        try:
            conn = psycopg2.connect(**self.db_params)
            cur = conn.cursor()
            
            # Enable vector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create embeddings table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS document_embeddings (
                    id SERIAL PRIMARY KEY,
                    content_id TEXT,
                    title TEXT,
                    content TEXT,
                    metadata JSONB,
                    embedding vector(384)  -- 384 dimensions for BAAI/bge-small-en-v1.5
                );
            """)
            
            # Create index for fast similarity search
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_embedding_cosine 
                ON document_embeddings USING ivfflat (embedding vector_cosine_ops);
            """)
            
            conn.commit()
            cur.close()
            conn.close()
            
            print("✅ Vector table setup complete")
            
        except Exception as e:
            print(f"❌ Error setting up vector table: {e}")
    
    def load_content_from_postgres(self):
        """Load existing content from PostgreSQL"""
        print("Loading content from PostgreSQL...")
        
        try:
            conn = psycopg2.connect(**self.db_params)
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Get all sections with content
            cur.execute("""
                SELECT 
                    s.id,
                    s.title,
                    s.content,
                    c.title as chapter_title,
                    c.number as chapter_number
                FROM sections s
                JOIN chapters c ON s.chapter_id = c.id
                WHERE s.content IS NOT NULL AND LENGTH(s.content) > 50
                ORDER BY c.number, s.id;
            """)
            
            rows = cur.fetchall()
            
            documents = []
            for row in rows:
                doc = Document(
                    text=f"Title: {row['title']}\n\nContent: {row['content']}",
                    metadata={
                        "id": str(row['id']),
                        "title": row['title'],
                        "chapter": row['chapter_title'],
                        "chapter_number": str(row['chapter_number']),
                        "type": "section"
                    }
                )
                documents.append(doc)
            
            cur.close()
            conn.close()
            
            print(f"✅ Loaded {len(documents)} documents from PostgreSQL")
            return documents
            
        except Exception as e:
            print(f"❌ Error loading from PostgreSQL: {e}")
            return []
    
    def load_content_from_files(self):
        """Load content from JSON files as fallback"""
        print("Loading content from files...")
        
        documents = []
        
        # Content directories supplied on the command line
        for path in self.data_paths:
            if os.path.exists(path):
                json_files = [f for f in os.listdir(path) if f.endswith('.json')]
                print(f"Found {len(json_files)} JSON files in {path}")
                
                for json_file in json_files[:15]:  # Process first 15 files
                    try:
                        with open(os.path.join(path, json_file), 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # Handle different JSON structures
                        self._process_json_data(data, json_file, documents)
                        
                    except Exception as e:
                        print(f"⚠️  Could not process {json_file}: {e}")
                        continue
        
        print(f"✅ Loaded {len(documents)} documents from files")
        return documents
    
    def _process_json_data(self, data, filename, documents):
        """Process JSON data and create documents"""
        if isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    text_content, title = self._extract_text_and_title(item)
                    if text_content.strip():
                        doc = Document(
                            text=text_content.strip(),
                            metadata={
                                "id": f"{filename}_{i}",
                                "title": title,
                                "source": filename,
                                "type": "file_content"
                            }
                        )
                        documents.append(doc)
                        
        elif isinstance(data, dict):
            text_content, title = self._extract_text_and_title(data)
            if text_content.strip():
                doc = Document(
                    text=text_content.strip(),
                    metadata={
                        "id": filename,
                        "title": title,
                        "source": filename,
                        "type": "file_content"
                    }
                )
                documents.append(doc)
    
    def _extract_text_and_title(self, item):
        """Extract text content and title from JSON item"""
        text_content = ""
        title = ""
        
        # Extract text content from various fields
        for field in ['content', 'question', 'answer', 'text', 'description']:
            if field in item and item[field]:
                if field in ['question', 'answer']:
                    text_content += f"{field.title()}: {item[field]}\n"
                else:
                    text_content += str(item[field]) + "\n"
        
        # Extract title
        for field in ['title', 'question', 'topic', 'name']:
            if field in item and item[field]:
                title = str(item[field])[:100]  # Limit title length
                break
        
        return text_content, title or "Untitled"
    
    def create_semantic_index(self):
        """Create semantic index using PostgreSQL pgvector"""
        print("Creating semantic index with PostgreSQL...")
        
        # Try loading from PostgreSQL first
        documents = self.load_content_from_postgres()
        
        # Fallback to files if no PostgreSQL content
        if not documents:
            print("No PostgreSQL content found, loading from files...")
            documents = self.load_content_from_files()
        
        if not documents:
            print("❌ No documents loaded!")
            return None
        
        print(f"Processing {len(documents)} documents...")
        
        try:
            # Create PG Vector Store
            vector_store = PGVectorStore.from_params(
                database=self.db_params["database"],
                host=self.db_params["host"],
                password=self.db_params["password"],
                port=self.db_params["port"],
                user=self.db_params["user"],
                table_name="document_embeddings",
                embed_dim=384  # BAAI/bge-small-en-v1.5 embedding dimension
            )
            
            # Create the index
            index = VectorStoreIndex.from_documents(
                documents, 
                vector_store=vector_store
            )
            
            print("✅ Semantic index created in PostgreSQL!")
            
            # Test the index
            self.test_index(index)
            
            return index
            
        except Exception as e:
            print(f"❌ Error creating index: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def test_index(self, index):
        """Test the semantic search"""
        print("\n🧪 Testing semantic search...")
        
        query_engine = index.as_query_engine(similarity_top_k=3)
        
        test_queries = ["risk management", "portfolio construction", "investment strategy"]
        
        for query in test_queries:
            try:
                print(f"\n🔍 Testing: '{query}'")
                response = query_engine.query(query)
                print(f"✅ Success!")
                print(f"   Response: {str(response)[:150]}...")
                
            except Exception as e:
                print(f"❌ Failed: {e}")

if __name__ == "__main__":
    print("🚀 Starting PostgreSQL Semantic Indexer")
    print("="*60)
    
    import argparse
    parser = argparse.ArgumentParser(description="Build a pgvector semantic index")
    parser.add_argument("data_paths", nargs="*",
                        help="Fallback directories of JSON content files")
    args = parser.parse_args()
    indexer = PostgreSQLSemanticIndexer(data_paths=args.data_paths)
    index = indexer.create_semantic_index()
    
    if index:
        print("\n🎉 SUCCESS! Semantic search ready in PostgreSQL!")
    else:
        print("\n❌ FAILED! Check errors above.")
