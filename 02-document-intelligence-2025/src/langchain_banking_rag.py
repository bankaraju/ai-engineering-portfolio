# langchain_banking_rag.py
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.chains import RetrievalQA
from langchain.memory import ConversationSummaryBufferMemory
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import os
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "")

class BankingRAGSystem:
    def __init__(self):
        # Initialize BGE embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # PostgreSQL connection string
        self.connection_string = os.getenv("PGVECTOR_CONNECTION_STRING", "postgresql://postgres@localhost:5432/market_intelligence")
        
        # Initialize PGVector store
        self.vector_store = PGVector(
            connection_string=self.connection_string,
            embedding_function=self.embeddings,
            collection_name="banking_financial_data",
            distance_strategy="cosine"
        )
        
        # Initialize Claude (model id from ANTHROPIC_MODEL)
        self.llm = ChatAnthropic(
            model=ANTHROPIC_MODEL,
            temperature=0,
            max_tokens=4096,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        
        # Banking-specific prompt
        self.prompt_template = """You are a senior banking analyst with deep expertise in financial analysis.

Context from database:
{context}

Current Question: {question}

Instructions:
1. Use ONLY the data provided in the context
2. If data is missing, explicitly state what's needed
3. Provide specific numbers and calculations
4. Follow institutional research report standards
5. Never make up or estimate numbers not in the context

Professional Banking Analysis:"""
        
        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )
    
    def add_financial_data_to_vectorstore(self, symbol: str, data: Dict):
        """Add financial data to PGVector"""
        documents = []
        
        # Create comprehensive document for each financial period
        for year in range(2022, 2025):
            for quarter in range(1, 5):
                doc_text = f"""
                {symbol} Financial Data {year} Q{quarter}:
                Revenue: {data.get(f'revenue_{year}_q{quarter}', 'Not Available')}
                PAT (Profit After Tax): {data.get(f'pat_{year}_q{quarter}', 'Not Available')}
                EBITDA: {data.get(f'ebitda_{year}_q{quarter}', 'Not Available')}
                Net Interest Margin: {data.get(f'nim_{year}_q{quarter}', 'Not Available')}
                Cost to Income Ratio: {data.get(f'cost_income_{year}_q{quarter}', 'Not Available')}
                Gross NPA: {data.get(f'gnpa_{year}_q{quarter}', 'Not Available')}
                Net NPA: {data.get(f'nnpa_{year}_q{quarter}', 'Not Available')}
                ROE: {data.get(f'roe_{year}_q{quarter}', 'Not Available')}
                ROA: {data.get(f'roa_{year}_q{quarter}', 'Not Available')}
                Capital Adequacy Ratio: {data.get(f'car_{year}_q{quarter}', 'Not Available')}
                Book Value per Share: {data.get(f'book_value_{year}_q{quarter}', 'Not Available')}
                """
                
                metadata = {
                    'symbol': symbol,
                    'year': year,
                    'quarter': quarter,
                    'source': 'annual_report',
                    'data_type': 'financial_statement'
                }
                
                documents.append(Document(
                    page_content=doc_text.strip(),
                    metadata=metadata
                ))
        
        # Add documents to vector store
        self.vector_store.add_documents(documents)
        logger.info(f"Added {len(documents)} documents for {symbol}")
    
    def query_financial_data(self, query: str, symbol: str = None) -> str:
        """Query financial data using RAG"""
        # Add symbol filter if provided
        if symbol:
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 5,
                    "filter": {"symbol": symbol}
                }
            )
        else:
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt}
        )
        
        # Execute query
        result = qa_chain({"query": query})
        
        return {
            'answer': result['result'],
            'source_documents': result['source_documents']
        }
    
    def calculate_dcf_with_rag(self, symbol: str) -> Dict:
        """Calculate DCF using RAG to retrieve all necessary data"""
        dcf_query = f"""
        For {symbol}, provide the following data for DCF calculation:
        1. Latest annual revenue and last 3 years revenue
        2. Latest PAT (Profit After Tax) and historical PAT
        3. EBITDA margins
        4. Book value per share
        5. Number of shares outstanding
        6. Industry average PE ratio
        7. Beta and cost of equity
        Calculate the revenue CAGR and provide all numbers clearly.
        """
        
        result = self.query_financial_data(dcf_query, symbol)
        
        # Parse the response to extract numbers
        # This would be enhanced with structured output parsing
        return {
            'rag_response': result['answer'],
            'sources': [doc.metadata for doc in result['source_documents']]
        }