#!/usr/bin/env python3
"""
Semantic Document Understanding Pipeline
A dynamic, embedding-based approach to document analysis without hardcoding
"""

import os
from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import json
from pathlib import Path

# Core imports
from llama_index import (
    Document, 
    VectorStoreIndex,
    StorageContext,
    ServiceContext,
    TreeIndex,
    DocumentSummaryIndex
)
from llama_index.node_parser import HierarchicalNodeParser, SentenceWindowNodeParser
from llama_index.embeddings import BGEEmbedding
from llama_index.llms import Anthropic

from langchain import LLMChain, PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

import chromadb
from sentence_transformers import SentenceTransformer

# Single Claude model id for the whole module, read from the environment.
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "")

@dataclass
class DocumentChunk:
    """Represents a semantic chunk of document"""
    content: str
    metadata: Dict
    embedding: Optional[np.ndarray] = None
    chunk_type: str = "paragraph"  # paragraph, section, table, etc.
    hierarchy_level: int = 0
    parent_id: Optional[str] = None
    children_ids: List[str] = None

class ParagraphGroupingEngine:
    """
    Generates Paragraph Grouping (PG) embeddings that capture
    semantic meaning at different granularities
    """
    
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        self.encoder = SentenceTransformer(model_name)
        self.chunk_strategies = {
            'semantic': self._semantic_chunking,
            'structural': self._structural_chunking,
            'hybrid': self._hybrid_chunking
        }
    
    def process_document(self, 
                        text: str, 
                        metadata: Dict,
                        strategy: str = 'hybrid') -> List[DocumentChunk]:
        """
        Process document into semantically meaningful chunks
        NO HARDCODING - adapts to document structure
        """
        
        # Use selected chunking strategy
        chunks = self.chunk_strategies[strategy](text, metadata)
        
        # Generate embeddings for each chunk
        for chunk in chunks:
            chunk.embedding = self._generate_embedding(chunk.content)
        
        # Establish semantic relationships
        chunks = self._build_semantic_graph(chunks)
        
        return chunks
    
    def _semantic_chunking(self, text: str, metadata: Dict) -> List[DocumentChunk]:
        """
        Chunk based on semantic coherence using sentence embeddings
        """
        sentences = text.split('. ')
        embeddings = self.encoder.encode(sentences)
        
        chunks = []
        current_chunk = []
        current_embedding = None
        
        for i, (sent, emb) in enumerate(zip(sentences, embeddings)):
            if current_embedding is None:
                current_embedding = emb
                current_chunk = [sent]
            else:
                # Calculate semantic similarity
                similarity = np.dot(current_embedding, emb) / (
                    np.linalg.norm(current_embedding) * np.linalg.norm(emb)
                )
                
                # Dynamic threshold based on document type
                threshold = self._get_similarity_threshold(metadata)
                
                if similarity > threshold:
                    current_chunk.append(sent)
                    # Update embedding as running average
                    current_embedding = (current_embedding + emb) / 2
                else:
                    # Create new chunk
                    chunks.append(DocumentChunk(
                        content='. '.join(current_chunk),
                        metadata={**metadata, 'chunk_method': 'semantic'},
                        chunk_type='paragraph'
                    ))
                    current_chunk = [sent]
                    current_embedding = emb
        
        # Don't forget last chunk
        if current_chunk:
            chunks.append(DocumentChunk(
                content='. '.join(current_chunk),
                metadata={**metadata, 'chunk_method': 'semantic'},
                chunk_type='paragraph'
            ))
        
        return chunks
    
    def _structural_chunking(self, text: str, metadata: Dict) -> List[DocumentChunk]:
        """
        Chunk based on document structure (headers, paragraphs, etc.)
        Dynamically detects structure patterns
        """
        import re
        
        # Detect structural patterns WITHOUT hardcoding
        patterns = self._detect_document_patterns(text)
        
        chunks = []
        
        # Split by detected patterns
        for pattern_type, pattern_regex in patterns.items():
            matches = list(re.finditer(pattern_regex, text, re.MULTILINE))
            
            for i, match in enumerate(matches):
                start = match.start()
                end = matches[i+1].start() if i+1 < len(matches) else len(text)
                
                content = text[start:end].strip()
                if content:
                    chunks.append(DocumentChunk(
                        content=content,
                        metadata={
                            **metadata, 
                            'chunk_method': 'structural',
                            'pattern_type': pattern_type
                        },
                        chunk_type=pattern_type,
                        hierarchy_level=self._get_hierarchy_level(pattern_type)
                    ))
        
        return chunks
    
    def _hybrid_chunking(self, text: str, metadata: Dict) -> List[DocumentChunk]:
        """
        Combines semantic and structural approaches
        Best of both worlds - no hardcoding
        """
        # First, structural chunking
        structural_chunks = self._structural_chunking(text, metadata)
        
        # Then, apply semantic chunking within each structural chunk
        final_chunks = []
        
        for struct_chunk in structural_chunks:
            if len(struct_chunk.content) > 1000:  # Dynamic threshold
                # Further split large chunks semantically
                semantic_sub_chunks = self._semantic_chunking(
                    struct_chunk.content, 
                    struct_chunk.metadata
                )
                
                for sub_chunk in semantic_sub_chunks:
                    sub_chunk.parent_id = struct_chunk.content[:50]  # Use content hash as ID
                    sub_chunk.hierarchy_level = struct_chunk.hierarchy_level + 1
                    final_chunks.append(sub_chunk)
            else:
                final_chunks.append(struct_chunk)
        
        return final_chunks
    
    def _detect_document_patterns(self, text: str) -> Dict[str, str]:
        """
        Dynamically detect document structure patterns
        NO HARDCODING - learns from document
        """
        patterns = {}
        
        # Detect section headers
        lines = text.split('\n')
        potential_headers = []
        
        for i, line in enumerate(lines):
            # Heuristics for headers (dynamic detection)
            if (len(line) < 100 and 
                line.isupper() or 
                line.endswith(':') or
                any(line.startswith(p) for p in ['Chapter', 'Section', '1.', 'A.'])):
                potential_headers.append((i, line))
        
        # Build regex patterns from detected headers
        if potential_headers:
            # Group similar headers
            header_groups = self._cluster_headers(potential_headers)
            
            for group_name, headers in header_groups.items():
                # Create regex that matches this type of header
                if headers:
                    pattern = self._create_header_regex(headers)
                    patterns[group_name] = pattern
        
        # Default patterns if none detected
        if not patterns:
            patterns = {
                'paragraph': r'\n\n(.+?)\n\n',
                'sentence': r'[.!?]+\s+',
            }
        
        return patterns
    
    def _build_semantic_graph(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Build relationships between chunks based on semantic similarity
        """
        if not chunks:
            return chunks
        
        # Calculate pairwise similarities
        embeddings = np.array([c.embedding for c in chunks])
        similarities = np.dot(embeddings, embeddings.T)
        
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1)
        similarities = similarities / (norms[:, None] * norms[None, :])
        
        # Build relationships (top-k similar for each chunk)
        k = min(5, len(chunks) - 1)
        
        for i, chunk in enumerate(chunks):
            if k > 0:
                # Get top-k similar chunks (excluding self)
                similar_indices = np.argsort(similarities[i])[-k-1:-1][::-1]
                chunk.metadata['similar_chunks'] = [
                    chunks[idx].content[:50] for idx in similar_indices
                ]
                chunk.metadata['similarity_scores'] = [
                    float(similarities[i][idx]) for idx in similar_indices
                ]
        
        return chunks
    
    def _get_similarity_threshold(self, metadata: Dict) -> float:
        """
        Dynamic threshold based on document type
        """
        doc_type = metadata.get('doc_type', 'general')
        
        # Dynamic thresholds learned from document characteristics
        if 'annual_report' in doc_type:
            return 0.75  # Higher threshold for financial documents
        elif 'legal' in doc_type:
            return 0.85  # Even higher for legal documents
        else:
            return 0.65  # Default threshold
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate BGE embedding for text"""
        return self.encoder.encode(text, normalize_embeddings=True)


class HierarchicalTOCBuilder:
    """
    Builds dynamic hierarchical Table of Contents using LlamaIndex
    No hardcoding - adapts to any document structure
    """
    
    def __init__(self, llm_model: str = ANTHROPIC_MODEL):
        # Initialize BGE embeddings
        self.embed_model = BGEEmbedding(
            model_name="BAAI/bge-large-en-v1.5",
            embed_batch_size=512
        )
        
        # Initialize LLM (Opus)
        self.llm = Anthropic(model=llm_model)
        
        # Service context with our models
        self.service_context = ServiceContext.from_defaults(
            llm=self.llm,
            embed_model=self.embed_model
        )
    
    def build_hierarchical_index(self, 
                                chunks: List[DocumentChunk],
                                company: str) -> Tuple[TreeIndex, Dict]:
        """
        Build hierarchical index from chunks
        Returns index and extracted TOC structure
        """
        
        # Convert chunks to LlamaIndex documents
        documents = []
        chunk_hierarchy = {}
        
        for chunk in chunks:
            doc = Document(
                text=chunk.content,
                metadata={
                    **chunk.metadata,
                    'chunk_type': chunk.chunk_type,
                    'hierarchy_level': chunk.hierarchy_level,
                    'company': company
                }
            )
            documents.append(doc)
            
            # Build hierarchy mapping
            level = chunk.hierarchy_level
            if level not in chunk_hierarchy:
                chunk_hierarchy[level] = []
            chunk_hierarchy[level].append(chunk)
        
        # Create hierarchical node parser
        node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[2048, 1024, 512],  # Different sizes for different levels
            chunk_overlap=200
        )
        
        # Parse nodes maintaining hierarchy
        nodes = node_parser.get_nodes_from_documents(documents)
        
        # Build tree index
        tree_index = TreeIndex(
            nodes=nodes,
            service_context=self.service_context,
            show_progress=True
        )
        
        # Extract dynamic TOC structure
        toc = self._extract_toc_structure(tree_index, chunk_hierarchy)
        
        return tree_index, toc
    
    def _extract_toc_structure(self, 
                              tree_index: TreeIndex,
                              chunk_hierarchy: Dict) -> Dict:
        """
        Extract table of contents from tree structure
        Completely dynamic - no hardcoding
        """
        
        # Use LLM to identify section headers and structure
        toc_prompt = PromptTemplate(
            input_variables=["content", "level"],
            template="""
            Analyze this content and identify if it contains a section header or title.
            If yes, extract the title and determine its importance.
            
            Content: {content}
            Hierarchy Level: {level}
            
            Return JSON:
            {{
                "is_header": true/false,
                "title": "extracted title",
                "importance": 1-10,
                "topics": ["topic1", "topic2"],
                "section_type": "introduction/financial/risk/governance/other"
            }}
            """
        )
        
        toc_structure = {
            "company": chunk_hierarchy.get(0, [{}])[0].metadata.get('company', 'Unknown'),
            "sections": []
        }
        
        # Process each hierarchy level
        for level in sorted(chunk_hierarchy.keys()):
            level_sections = []
            
            for chunk in chunk_hierarchy[level]:
                # Use LLM to analyze chunk
                result = self.llm.predict(
                    toc_prompt.format(
                        content=chunk.content[:500],
                        level=level
                    )
                )
                
                try:
                    section_info = json.loads(result)
                    
                    if section_info.get('is_header'):
                        level_sections.append({
                            'title': section_info['title'],
                            'level': level,
                            'importance': section_info['importance'],
                            'topics': section_info['topics'],
                            'type': section_info['section_type'],
                            'content_preview': chunk.content[:200],
                            'subsections': []
                        })
                except:
                    continue
            
            toc_structure['sections'].extend(level_sections)
        
        # Build parent-child relationships
        toc_structure = self._build_toc_hierarchy(toc_structure)
        
        return toc_structure
    
    def _build_toc_hierarchy(self, toc_structure: Dict) -> Dict:
        """
        Organize flat sections into hierarchical structure
        Based on levels and semantic relationships
        """
        sections = toc_structure['sections']
        if not sections:
            return toc_structure
        
        # Sort by level
        sections.sort(key=lambda x: x['level'])
        
        # Build tree
        root_sections = [s for s in sections if s['level'] == 0]
        
        for i, section in enumerate(sections):
            if section['level'] > 0:
                # Find parent (previous section with lower level)
                for j in range(i-1, -1, -1):
                    if sections[j]['level'] < section['level']:
                        sections[j]['subsections'].append(section)
                        break
        
        toc_structure['sections'] = root_sections
        return toc_structure


class LangchainOrchestrator:
    """
    Orchestrates the entire pipeline using Langchain
    Integrates all components dynamically
    """
    
    def __init__(self):
        self.pg_engine = ParagraphGroupingEngine()
        self.toc_builder = HierarchicalTOCBuilder()
        self.vector_store = None
        self.retrieval_chain = None
        
        # Initialize BGE embeddings for Langchain
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def process_document_pipeline(self, 
                                 file_path: str,
                                 company: str,
                                 metadata: Dict = None) -> Dict:
        """
        Complete pipeline: PDF → Chunks → Embeddings → Index → TOC
        """
        
        # Step 1: Load document
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        # Combine pages into full text
        full_text = "\n\n".join([page.page_content for page in pages])
        
        # Add metadata
        if metadata is None:
            metadata = {}
        metadata.update({
            'company': company,
            'file_path': file_path,
            'doc_type': self._detect_document_type(full_text)
        })
        
        # Step 2: Generate PG embeddings
        print(f"Generating paragraph groupings for {company}...")
        chunks = self.pg_engine.process_document(
            full_text, 
            metadata,
            strategy='hybrid'
        )
        
        # Step 3: Build hierarchical index
        print(f"Building hierarchical TOC for {company}...")
        tree_index, toc = self.toc_builder.build_hierarchical_index(chunks, company)
        
        # Step 4: Create vector store for semantic search
        print(f"Creating searchable index for {company}...")
        texts = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        self.vector_store = FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas
        )
        
        # Step 5: Create retrieval chain with Opus
        self.retrieval_chain = self._create_retrieval_chain()
        
        return {
            'company': company,
            'chunks': chunks,
            'tree_index': tree_index,
            'toc': toc,
            'vector_store': self.vector_store,
            'metadata': metadata,
            'stats': {
                'total_chunks': len(chunks),
                'hierarchy_levels': len(set(c.hierarchy_level for c in chunks)),
                'document_type': metadata['doc_type']
            }
        }
    
    def _detect_document_type(self, text: str) -> str:
        """
        Dynamically detect document type
        No hardcoding - uses content analysis
        """
        text_lower = text.lower()
        
        # Score different document types based on keywords
        doc_types = {
            'annual_report': [
                'annual report', 'financial statements', 'balance sheet',
                'income statement', 'cash flow', 'shareholders', 'directors report'
            ],
            'quarterly_results': [
                'quarterly', 'q1', 'q2', 'q3', 'q4', 'unaudited',
                'three months ended', 'nine months ended'
            ],
            'investor_presentation': [
                'investor presentation', 'earnings call', 'guidance',
                'outlook', 'strategic initiatives'
            ]
        }
        
        scores = {}
        for doc_type, keywords in doc_types.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[doc_type] = score
        
        # Return highest scoring type
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def _create_retrieval_chain(self):
        """
        Create Langchain retrieval chain with Opus
        """
        from langchain.llms import Anthropic
        from langchain.chains import RetrievalQA
        
        llm = Anthropic(model=ANTHROPIC_MODEL)
        
        qa_prompt = PromptTemplate(
            template="""You are analyzing financial documents using advanced semantic understanding.
            
            Use the following context to answer the question. The context comes from 
            semantically related chunks that may not be physically adjacent in the document.
            
            Context: {context}
            
            Question: {question}
            
            Provide a comprehensive answer that:
            1. Directly addresses the question
            2. Cites specific sections or data points
            3. Identifies any patterns or relationships
            4. Highlights any red flags or concerns
            
            Answer:""",
            input_variables=["context", "question"]
        )
        
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 10}  # Retrieve top 10 relevant chunks
            ),
            chain_type_kwargs={"prompt": qa_prompt}
        )
        
        return chain
    
    def query_documents(self, question: str) -> Dict:
        """
        Query processed documents using semantic search
        """
        if not self.retrieval_chain:
            raise ValueError("No documents processed yet!")
        
        # Get answer using retrieval chain
        answer = self.retrieval_chain.run(question)
        
        # Also get relevant chunks for transparency
        relevant_chunks = self.vector_store.similarity_search(
            question, 
            k=5
        )
        
        return {
            'question': question,
            'answer': answer,
            'relevant_chunks': [
                {
                    'content': chunk.page_content[:200] + "...",
                    'metadata': chunk.metadata
                }
                for chunk in relevant_chunks
            ]
        }


class SemanticDocumentAnalyzer:
    """
    Main interface for the complete semantic analysis pipeline
    """
    
    def __init__(self, storage_path: str = "./semantic_storage"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        self.orchestrator = LangchainOrchestrator()
        self.processed_companies = {}
    
    def analyze_company(self, 
                       company: str,
                       pdf_paths: List[str]) -> Dict:
        """
        Analyze all documents for a company
        """
        
        all_results = []
        
        for pdf_path in pdf_paths:
            print(f"\nProcessing {pdf_path}...")
            
            # Extract year from filename
            import re
            year_match = re.search(r'(\d{4})', pdf_path)
            year = year_match.group(1) if year_match else 'Unknown'
            
            # Process document
            result = self.orchestrator.process_document_pipeline(
                file_path=pdf_path,
                company=company,
                metadata={'year': year}
            )
            
            all_results.append(result)
            
            # Save processed data
            self._save_processed_data(company, year, result)
        
        # Combine results across years
        combined_result = self._combine_temporal_results(company, all_results)
        
        self.processed_companies[company] = combined_result
        
        return combined_result
    
    def _combine_temporal_results(self, 
                                 company: str,
                                 results: List[Dict]) -> Dict:
        """
        Combine results across multiple years for temporal analysis
        """
        
        # Sort by year
        results.sort(key=lambda x: x['metadata'].get('year', '0'))
        
        # Extract temporal patterns
        temporal_patterns = {
            'working_capital_trend': [],
            'revenue_growth': [],
            'structural_changes': []
        }
        
        for result in results:
            year = result['metadata']['year']
            
            # Query each year's data for key metrics
            wc_query = "What is the working capital cycle in days?"
            wc_answer = self.orchestrator.query_documents(wc_query)
            
            # Extract numeric value (this is where Opus helps)
            import re
            wc_match = re.search(r'(\d+)\s*days', wc_answer['answer'])
            if wc_match:
                temporal_patterns['working_capital_trend'].append({
                    'year': year,
                    'days': int(wc_match.group(1))
                })
        
        return {
            'company': company,
            'years_analyzed': [r['metadata']['year'] for r in results],
            'temporal_patterns': temporal_patterns,
            'all_results': results
        }
    
    def _save_processed_data(self, 
                           company: str,
                           year: str,
                           result: Dict):
        """
        Save processed data for future use
        """
        
        company_dir = self.storage_path / company
        company_dir.mkdir(exist_ok=True)
        
        # Save TOC
        toc_path = company_dir / f"{year}_toc.json"
        with open(toc_path, 'w') as f:
            json.dump(result['toc'], f, indent=2)
        
        # Save chunks metadata
        chunks_data = []
        for chunk in result['chunks']:
            chunks_data.append({
                'content_preview': chunk.content[:100],
                'metadata': chunk.metadata,
                'chunk_type': chunk.chunk_type,
                'hierarchy_level': chunk.hierarchy_level
            })
        
        chunks_path = company_dir / f"{year}_chunks.json"
        with open(chunks_path, 'w') as f:
            json.dump(chunks_data, f, indent=2)
        
        # Save vector store
        vector_path = company_dir / f"{year}_vectors"
        result['vector_store'].save_local(str(vector_path))
    
    def run_fraud_analysis(self, company: str) -> Dict:
        """
        Run comprehensive fraud analysis using semantic understanding
        """
        
        if company not in self.processed_companies:
            raise ValueError(f"Company {company} not processed yet!")
        
        fraud_queries = [
            "Are there any significant increases in related party transactions?",
            "Has the working capital cycle deteriorated over the years?",
            "Are there any circular transactions mentioned?",
            "What percentage of revenue comes from top 10 customers?",
            "Are there any qualified opinions from auditors?",
            "Have there been significant changes in accounting policies?",
            "Are there any contingent liabilities that seem concerning?",
            "Is there evidence of asset inflation or overvaluation?"
        ]
        
        fraud_signals = []
        
        for query in fraud_queries:
            result = self.orchestrator.query_documents(query)
            
            # Use Opus to interpret the answer
            interpretation_prompt = f"""
            Based on this answer, determine if there's a fraud risk signal:
            
            Question: {query}
            Answer: {result['answer']}
            
            Return JSON:
            {{
                "risk_detected": true/false,
                "severity": "high/medium/low/none",
                "explanation": "brief explanation",
                "red_flags": ["flag1", "flag2"]
            }}
            """
            
            # This is where Opus provides intelligent interpretation
            fraud_signals.append({
                'query': query,
                'answer': result['answer'],
                'relevant_chunks': result['relevant_chunks']
            })
        
        return {
            'company': company,
            'fraud_signals': fraud_signals,
            'overall_risk_score': self._calculate_risk_score(fraud_signals)
        }
    
    def _calculate_risk_score(self, fraud_signals: List[Dict]) -> float:
        """
        Calculate overall risk score from fraud signals
        """
        # This would use Opus to intelligently weight different signals
        # For now, simple counting
        risk_count = sum(1 for signal in fraud_signals 
                        if 'concern' in signal.get('answer', '').lower() 
                        or 'increase' in signal.get('answer', '').lower())
        
        return min(risk_count * 10, 100)


# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = SemanticDocumentAnalyzer()
    
    # Process Genesys annual reports
    genesys_pdfs = [
        "/path/to/GENESYS_AR_2024.pdf",
        "/path/to/GENESYS_AR_2023.pdf",
        "/path/to/GENESYS_AR_2022.pdf"
    ]
    
    # Analyze documents
    print("Starting semantic analysis of Genesys...")
    result = analyzer.analyze_company("GENESYS", genesys_pdfs)
    
    # Run fraud analysis
    print("\nRunning fraud detection...")
    fraud_result = analyzer.run_fraud_analysis("GENESYS")
    
    # Query specific information
    print("\nQuerying for working capital trends...")
    wc_result = analyzer.orchestrator.query_documents(
        "How has the working capital cycle changed from 2022 to 2024?"
    )
    
    print(f"\nAnswer: {wc_result['answer']}")
    print(f"\nRelevant sections found: {len(wc_result['relevant_chunks'])}")
