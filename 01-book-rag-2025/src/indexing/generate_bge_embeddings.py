# generate_bge_embeddings.py
"""Backfill BGE embeddings into existing pgvector tables and add an ivfflat
cosine index per table. Tables are passed on the command line; connection
settings come from environment variables (see .env.example)."""
import os
import argparse
import psycopg2
from psycopg2.extras import RealDictCursor
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
import time

def generate_embeddings_for_all_banks(embedding_tables):
    print("="*60)
    print("GENERATING BGE EMBEDDINGS FOR ALL BANKS")
    print("="*60)
    
    # Initialize BGE model
    print("\n1. Loading BGE model...")
    model = SentenceTransformer(os.getenv("BANK_EMBED_MODEL", "BAAI/bge-base-en-v1.5"))
    print("✅ BGE model loaded")
    
    # Connect to database
    conn = psycopg2.connect(
        dbname=os.getenv("PGDATABASE", "annual_reports_db"),
        user=os.getenv("PGUSER", "postgres"),
        password=os.getenv("PGPASSWORD", ""),
        host=os.getenv("PGHOST", "localhost"),
        port=int(os.getenv("PGPORT", "5432")),
    )
    cur = conn.cursor()
    
    # First, check vector dimension
    print("\n2. Setting up vector dimension...")
    try:
        # Test embedding dimension
        test_embedding = model.encode("test", normalize_embeddings=True)
        dimension = len(test_embedding)
        print(f"✅ BGE embedding dimension: {dimension}")
        
        # Update vector dimension if needed
        cur.execute(f"SET LOCAL vector.pgvector_compatibility = on")
        conn.commit()
    except Exception as e:
        print(f"Note: {e}")
    
    # Embedding tables (one per issuer) are passed in by the caller

    # Process each table
    for table_name in embedding_tables:
        bank_name = table_name.replace('data_', '').replace('_embeddings', '').upper()
        print(f"\n3. Processing {bank_name}...")
        
        try:
            # Get all records without embeddings
            cur.execute(f"""
                SELECT id, text 
                FROM {table_name}
                WHERE text IS NOT NULL AND text != ''
                ORDER BY id
            """)
            
            records = cur.fetchall()
            print(f"   Found {len(records)} text chunks")
            
            # Process in batches
            batch_size = 100
            updated_count = 0
            
            for i in tqdm(range(0, len(records), batch_size), desc=f"   Embedding {bank_name}"):
                batch = records[i:i+batch_size]
                
                # Extract texts and ids
                ids = [record[0] for record in batch]
                texts = [record[1] for record in batch]
                
                # Generate embeddings
                embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
                
                # Update database
                for record_id, embedding in zip(ids, embeddings):
                    try:
                        # Convert to list for PostgreSQL
                        embedding_list = embedding.tolist()
                        
                        cur.execute(f"""
                            UPDATE {table_name}
                            SET embedding = %s::vector
                            WHERE id = %s
                        """, (embedding_list, record_id))
                        
                        updated_count += 1
                        
                    except Exception as e:
                        print(f"\n❌ Error updating record {record_id}: {e}")
                        # Try alternative approach
                        try:
                            cur.execute(f"""
                                ALTER TABLE {table_name} 
                                ALTER COLUMN embedding TYPE vector({dimension})
                            """)
                            conn.commit()
                            # Retry the update
                            cur.execute(f"""
                                UPDATE {table_name}
                                SET embedding = %s::vector
                                WHERE id = %s
                            """, (embedding_list, record_id))
                        except:
                            pass
                
                # Commit batch
                conn.commit()
                
                # Small delay to avoid overloading
                time.sleep(0.1)
            
            print(f"   ✅ Updated {updated_count} embeddings for {bank_name}")
            
            # Create index for this table
            print(f"   Creating index for {bank_name}...")
            try:
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{table_name}_embedding 
                    ON {table_name} USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                """)
                conn.commit()
                print(f"   ✅ Index created")
            except Exception as e:
                print(f"   Note: Index might already exist - {e}")
                
        except Exception as e:
            print(f"❌ Error processing {bank_name}: {e}")
            conn.rollback()
            continue
    
    conn.close()
    print("\n✅ EMBEDDING GENERATION COMPLETE!")

if __name__ == "__main__":
    # Check if sentence-transformers is installed
    try:
        import sentence_transformers
        parser = argparse.ArgumentParser(description="Backfill BGE embeddings into pgvector tables")
        parser.add_argument("tables", nargs="+", help="Table names, e.g. data_bank_a_embeddings")
        args = parser.parse_args()
        import re
        bad = [t for t in args.tables if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", t)]
        if bad:
            raise SystemExit(f"Invalid table name(s): {bad}")
        generate_embeddings_for_all_banks(args.tables)
    except ImportError:
        print("Please install sentence-transformers first:")
        print("pip install sentence-transformers")
