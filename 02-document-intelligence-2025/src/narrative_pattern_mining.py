# extraction/narrative_extractor.py
import re
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime
import spacy
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import torch
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import text

from config.setup import config, logger
from ingestion.google_drive_monitor import audit

# Load spaCy for NER
nlp = spacy.load("en_core_web_sm")

class NarrativeExtractor:
    """Extract and understand narratives with rich metadata"""
    
    def __init__(self):
        self.embed_model = config.embed_model
        self.conn = config.conn
        self.patterns = self._load_narrative_patterns()
        
        # Sentiment analyzer for narrative tone
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finBERT"  # Financial domain sentiment
        )
        
    def _load_narrative_patterns(self) -> Dict:
        """Load patterns for different narrative types"""
        
        return {
            'forward_guidance': {
                'patterns': [
                    r'(?:expect|anticipate|project|forecast|guide).*?(\d+)\s*(?:%|percent|bps|basis points)',
                    r'(?:target|aim|goal).*?(\d+)\s*(?:%|percent|bps)',
                    r'(?:likely|probable).*?(?:increase|decrease|grow|decline).*?(\d+)',
                    r'in (?:the )?(?:coming|next|following) (?:quarter|year|period)',
                    r'(?:FY|fiscal year) (\d{2,4}).*?(?:expect|target|guide)'
                ],
                'confidence_modifiers': {
                    'high': ['definitely', 'certainly', 'committed', 'confident'],
                    'medium': ['expect', 'anticipate', 'likely', 'probable'],
                    'low': ['may', 'might', 'could', 'possible', 'potential']
                }
            },
            'risk_narrative': {
                'patterns': [
                    r'(?:risk|concern|challenge|threat).*?(?:from|due to|because of|arising from)(.*?)(?:\.|;)',
                    r'(?:exposed|vulnerable|susceptible) to(.*?)(?:\.|;)',
                    r'(?:mitigate|manage|hedge|protect).*?(?:risk|exposure)',
                    r'(?:provision|provided|set aside).*?(?:for|against)(.*?)(?:\.|;)'
                ],
                'risk_categories': {
                    'credit': ['npa', 'asset quality', 'default', 'provision', 'slippage'],
                    'market': ['interest rate', 'forex', 'commodity', 'equity'],
                    'operational': ['cyber', 'fraud', 'system', 'compliance', 'regulatory'],
                    'strategic': ['competition', 'technology', 'disruption', 'market share']
                }
            },
            'performance_claim': {
                'patterns': [
                    r'(?:achieved|recorded|posted|reported).*?(\d+)\s*(?:%|percent|bps)',
                    r'(?:growth|increase|rise|improvement) of.*?(\d+)\s*(?:%|percent)',
                    r'(?:decline|decrease|fall|drop) of.*?(\d+)\s*(?:%|percent)',
                    r'(?:compared to|versus|vs|y-o-y|yoy).*?(\d+)\s*(?:%|percent)'
                ],
                'metric_indicators': {
                    'financial': ['revenue', 'income', 'profit', 'margin', 'nim', 'roi', 'roe'],
                    'operational': ['branches', 'customers', 'employees', 'transactions'],
                    'digital': ['users', 'adoption', 'digital', 'mobile', 'online'],
                    'quality': ['npa', 'provision', 'coverage', 'slippage', 'recovery']
                }
            },
            'strategic_initiative': {
                'patterns': [
                    r'(?:launch|introduce|roll out|unveil).*?(?:product|service|platform|initiative)',
                    r'(?:invest|investing|investment).*?in(.*?)(?:to|for)',
                    r'(?:partner|partnership|collaborate|tie-up).*?with(.*?)(?:to|for)',
                    r'(?:acquire|acquisition|merger|takeover)',
                    r'(?:expand|expansion).*?(?:into|in)(.*?)(?:market|region|segment)'
                ],
                'initiative_types': {
                    'digital': ['digital', 'mobile', 'online', 'platform', 'api', 'fintech'],
                    'geographic': ['expand', 'enter', 'presence', 'footprint', 'region'],
                    'product': ['launch', 'introduce', 'new product', 'service'],
                    'partnership': ['partner', 'collaborate', 'joint', 'alliance']
                }
            }
        }
    
    def extract_narratives(self, doc_id: str, structure: Dict) -> List[Dict]:
        """Extract narratives from document structure"""
        
        operation_id = f"narrative_extract_{doc_id}"
        narratives = []
        
        try:
            # Process each section
            for section_name, section_data in structure['sections'].items():
                section_narratives = self._process_section(
                    doc_id, section_name, section_data
                )
                narratives.extend(section_narratives)
            
            # Generate embeddings for all narratives
            narrative_texts = [n['text'] for n in narratives]
            if narrative_texts:
                embeddings = self._generate_embeddings(narrative_texts)
                for i, narrative in enumerate(narratives):
                    narrative['embedding'] = embeddings[i]
            
            # Store in database
            stored_count = self._store_narratives(narratives)
            
            audit.log_operation(
                'narrative_extraction', operation_id,
                {'doc_id': doc_id, 'section_count': len(structure['sections'])},
                {'narratives_extracted': len(narratives), 'stored': stored_count},
                success=True
            )
            
            return narratives
            
        except Exception as e:
            audit.log_operation(
                'narrative_extraction', operation_id,
                {'doc_id': doc_id},
                error=str(e),
                success=False
            )
            raise
    
    def _process_section(self, doc_id: str, section_name: str, 
                        section_data: Dict) -> List[Dict]:
        """Process a single section for narratives"""
        
        narratives = []
        
        for element in section_data.get('elements', []):
            if element['type'] == 'paragraph':
                # Extract narratives from paragraph
                para_narratives = self._extract_from_paragraph(
                    element['content'],
                    doc_id,
                    section_name,
                    element['page']
                )
                narratives.extend(para_narratives)
        
        return narratives
    
    def _extract_from_paragraph(self, text: str, doc_id: str, 
                              section: str, page: int) -> List[Dict]:
        """Extract narratives from a single paragraph"""
        
        narratives = []
        
        # Clean text
        text = self._clean_text(text)
        if len(text) < 50:  # Skip very short paragraphs
            return narratives
        
        # Check each narrative type
        for narrative_type, type_patterns in self.patterns.items():
            for pattern in type_patterns['patterns']:
                matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
                
                for match in matches:
                    # Extract the full sentence containing the match
                    sentence = self._extract_sentence(text, match.span())
                    
                    if sentence:
                        narrative = {
                            'doc_id': doc_id,
                            'text': sentence,
                            'narrative_type': narrative_type,
                            'section_context': section,
                            'page_number': page,
                            'pattern_matched': pattern,
                            'specific_extract': match.group(0),
                            'confidence_score': self._calculate_confidence(
                                sentence, narrative_type, match
                            ),
                            'entities': self._extract_entities(sentence),
                            'sentiment': self._analyze_sentiment(sentence),
                            'specificity_score': self._calculate_specificity(sentence),
                            'temporal_reference': self._extract_temporal_reference(sentence),
                            'has_numbers': bool(re.search(r'\d+', sentence)),
                            'number_count': len(re.findall(r'\d+', sentence))
                        }
                        
                        # Add type-specific metadata
                        if narrative_type == 'forward_guidance':
                            narrative['confidence_modifier'] = self._get_confidence_modifier(
                                sentence, type_patterns
                            )
                        elif narrative_type == 'risk_narrative':
                            narrative['risk_category'] = self._classify_risk(
                                sentence, type_patterns
                            )
                        elif narrative_type == 'performance_claim':
                            narrative['metric_type'] = self._identify_metric_type(
                                sentence, type_patterns
                            )
                        elif narrative_type == 'strategic_initiative':
                            narrative['initiative_type'] = self._classify_initiative(
                                sentence, type_patterns
                            )
                        
                        narratives.append(narrative)
        
        return narratives
    
    def _clean_text(self, text: str) -> str:
        """Clean text for processing"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove special characters but keep numbers and punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\-\%\/\(\)]', ' ', text)
        return text.strip()
    
    def _extract_sentence(self, text: str, match_span: Tuple[int, int]) -> str:
        """Extract complete sentence containing the match"""
        
        # Find sentence boundaries
        sentences = text.split('.')
        current_pos = 0
        
        for sentence in sentences:
            sentence_end = current_pos + len(sentence) + 1
            if current_pos <= match_span[0] < sentence_end:
                return sentence.strip() + '.'
            current_pos = sentence_end
        
        # Fallback: return text around match
        start = max(0, match_span[0] - 100)
        end = min(len(text), match_span[1] + 100)
        return text[start:end].strip()
    
    def _calculate_confidence(self, text: str, narrative_type: str, 
                            match: re.Match) -> float:
        """Calculate confidence score for narrative"""
        
        confidence = 0.5  # Base confidence
        
        # Boost for specific numbers
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        if numbers:
            confidence += 0.2
            if len(numbers) > 1:
                confidence += 0.1
        
        # Boost for longer, more complete sentences
        if len(text.split()) > 15:
            confidence += 0.1
        
        # Boost for financial/banking terms
        financial_terms = ['revenue', 'margin', 'growth', 'basis points', 'bps', 
                          'quarter', 'fiscal', 'yoy', 'qoq']
        term_count = sum(1 for term in financial_terms if term in text.lower())
        confidence += min(0.2, term_count * 0.05)
        
        # Type-specific boosts
        if narrative_type == 'forward_guidance' and any(
            word in text.lower() for word in ['expect', 'target', 'guide']
        ):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _extract_entities(self, text: str) -> Dict:
        """Extract named entities from text"""
        
        doc = nlp(text)
        entities = {
            'organizations': [],
            'money': [],
            'dates': [],
            'percentages': [],
            'quantities': []
        }
        
        for ent in doc.ents:
            if ent.label_ == "ORG":
                entities['organizations'].append(ent.text)
            elif ent.label_ == "MONEY":
                entities['money'].append(ent.text)
            elif ent.label_ in ["DATE", "TIME"]:
                entities['dates'].append(ent.text)
            elif ent.label_ == "PERCENT":
                entities['percentages'].append(ent.text)
            elif ent.label_ == "QUANTITY":
                entities['quantities'].append(ent.text)
        
        # Also extract percentages not caught by NER
        percent_pattern = r'(\d+(?:\.\d+)?)\s*(?:%|percent|bps|basis points)'
        for match in re.finditer(percent_pattern, text, re.IGNORECASE):
            entities['percentages'].append(match.group(0))
        
        return entities
    
    def _analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment using FinBERT"""
        
        try:
            result = self.sentiment_analyzer(text[:512])[0]  # Limit length
            
            return {
                'label': result['label'],
                'score': result['score'],
                'is_positive': result['label'] == 'POSITIVE',
                'is_negative': result['label'] == 'NEGATIVE',
                'is_neutral': result['label'] == 'NEUTRAL'
            }
        except:
            return {
                'label': 'UNKNOWN',
                'score': 0.0,
                'is_positive': False,
                'is_negative': False,
                'is_neutral': True
            }
    
    def _calculate_specificity(self, text: str) -> float:
        """Calculate how specific vs generic the narrative is"""
        
        specificity = 0.0
        
        # Specific numbers increase specificity
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        specificity += min(0.4, len(numbers) * 0.1)
        
        # Specific time references
        time_refs = ['quarter', 'month', 'year', 'q1', 'q2', 'q3', 'q4', 'fy']
        time_count = sum(1 for ref in time_refs if ref in text.lower())
        specificity += min(0.3, time_count * 0.1)
        
        # Named entities
        doc = nlp(text)
        entity_count = len(doc.ents)
        specificity += min(0.3, entity_count * 0.05)
        
        # Vague terms decrease specificity
        vague_terms = ['various', 'several', 'multiple', 'some', 'many', 'few']
        vague_count = sum(1 for term in vague_terms if term in text.lower())
        specificity -= min(0.2, vague_count * 0.05)
        
        return max(0.0, min(1.0, specificity))
    
    def _extract_temporal_reference(self, text: str) -> Optional[str]:
        """Extract time reference from narrative"""
        
        # Immediate
        if any(word in text.lower() for word in ['immediate', 'current', 'now', 'today']):
            return 'immediate'
        
        # Quarterly
        if any(word in text.lower() for word in ['quarter', 'quarterly', 'q1', 'q2', 'q3', 'q4']):
            return '1Q'
        
        # Annual
        if any(word in text.lower() for word in ['year', 'annual', 'yearly', 'fy']):
            return '1Y'
        
        # Long term
        if any(word in text.lower() for word in ['long term', 'long-term', 'years', 'future']):
            return 'long-term'
        
        return None
    
    def _get_confidence_modifier(self, text: str, patterns: Dict) -> str:
        """Get confidence level from forward guidance"""
        
        text_lower = text.lower()
        
        for level, modifiers in patterns['confidence_modifiers'].items():
            if any(modifier in text_lower for modifier in modifiers):
                return level
        
        return 'medium'  # Default
    
    def _classify_risk(self, text: str, patterns: Dict) -> str:
        """Classify risk type"""
        
        text_lower = text.lower()
        
        for category, keywords in patterns['risk_categories'].items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        
        return 'general'
    
    def _identify_metric_type(self, text: str, patterns: Dict) -> str:
        """Identify what metric is being discussed"""
        
        text_lower = text.lower()
        
        for metric_type, indicators in patterns['metric_indicators'].items():
            if any(indicator in text_lower for indicator in indicators):
                return metric_type
        
        return 'general'
    
    def _classify_initiative(self, text: str, patterns: Dict) -> str:
        """Classify strategic initiative type"""
        
        text_lower = text.lower()
        
        for init_type, keywords in patterns['initiative_types'].items():
            if any(keyword in text_lower for keyword in keywords):
                return init_type
        
        return 'general'
    
    def _generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate BGE embeddings for texts"""
        
        # BGE model expects specific format
        embeddings = self.embed_model.get_text_embedding_batch(
            texts,
            show_progress=True
        )
        
        return embeddings
    
    def _store_narratives(self, narratives: List[Dict]) -> int:
        """Store narratives in database"""
        
        cur = self.conn.cursor()
        stored = 0
        
        for narrative in narratives:
            try:
                # Convert embedding to pgvector format
                embedding = narrative.get('embedding', [])
                if isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()
                
                cur.execute("""
                    INSERT INTO narrative_elements
                    (doc_id, narrative_text, narrative_type, section_context,
                     page_number, embedding, key_entities, sentiment_score,
                     specificity_score, time_reference, confidence_score,
                     has_numbers, number_count, extraction_method)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING element_id
                """, (
                    narrative['doc_id'],
                    narrative['text'],
                    narrative['narrative_type'],
                    [narrative['section_context']],  # Array field
                    narrative['page_number'],
                    embedding,
                    json.dumps(narrative['entities']),
                    narrative['sentiment']['score'],
                    narrative['specificity_score'],
                    narrative['temporal_reference'],
                    narrative['confidence_score'],
                    narrative['has_numbers'],
                    narrative['number_count'],
                    'pattern_matching'
                ))
                
                element_id = cur.fetchone()[0]
                narrative['element_id'] = element_id
                stored += 1
                
            except Exception as e:
                logger.error(f"Failed to store narrative: {e}")
                self.conn.rollback()
        
        self.conn.commit()
        return stored

class PatternDiscoveryEngine:
    """Discover patterns from narratives - not predefined!"""
    
    def __init__(self):
        self.conn = config.conn
        self.embed_model = config.embed_model
        
    def discover_patterns(self, company: str, min_occurrences: int = 3) -> List[Dict]:
        """Discover recurring patterns in company narratives"""
        
        operation_id = f"pattern_discovery_{company}_{datetime.now().strftime('%Y%m%d')}"
        
        try:
            # Get all narratives for company
            narratives = self._load_company_narratives(company)
            
            # Cluster similar narratives
            clusters = self._cluster_narratives(narratives)
            
            # Identify temporal patterns
            temporal_patterns = self._find_temporal_patterns(clusters)
            
            # Extract pattern definitions
            discovered_patterns = []
            for pattern in temporal_patterns:
                if pattern['occurrence_count'] >= min_occurrences:
                    pattern_def = self._define_pattern(pattern)
                    discovered_patterns.append(pattern_def)
            
            # Store discovered patterns
            stored = self._store_patterns(discovered_patterns)
            
            audit.log_operation(
                'pattern_discovery', operation_id,
                {'company': company, 'narrative_count': len(narratives)},
                {'patterns_discovered': len(discovered_patterns), 'stored': stored},
                success=True
            )
            
            # Learn from discovery
            if discovered_patterns:
                audit.log_learning(
                    'pattern_discovery',
                    f"Discovered {len(discovered_patterns)} patterns for {company}",
                    {'action': 'validate_on_peers'},
                    {'company': company, 'pattern_names': [p['name'] for p in discovered_patterns]}
                )
            
            return discovered_patterns
            
        except Exception as e:
            audit.log_operation(
                'pattern_discovery', operation_id,
                {'company': company},
                error=str(e),
                success=False
            )
            raise
    
    def _load_company_narratives(self, company: str) -> List[Dict]:
        """Load all narratives for a company"""
        
        cur = self.conn.cursor()
        cur.execute("""
            SELECT 
                ne.element_id,
                ne.narrative_text,
                ne.narrative_type,
                ne.embedding,
                ne.sentiment_score,
                ne.specificity_score,
                ne.time_reference,
                ne.confidence_score,
                dr.fiscal_year,
                dr.fiscal_quarter
            FROM narrative_elements ne
            JOIN document_registry dr ON ne.doc_id = dr.doc_id
            WHERE dr.company = %s
            ORDER BY dr.fiscal_year, dr.fiscal_quarter
        """, (company,))
        
        narratives = []
        for row in cur.fetchall():
            narratives.append({
                'element_id': row[0],
                'text': row[1],
                'type': row[2],
                'embedding': np.array(row[3]) if row[3] else None,
                'sentiment': row[4],
                'specificity': row[5],
                'time_ref': row[6],
                'confidence': row[7],
                'fiscal_year': row[8],
                'fiscal_quarter': row[9]
            })
        
        return narratives
    
    def _cluster_narratives(self, narratives: List[Dict]) -> List[Dict]:
        """Cluster similar narratives using embeddings"""
        
        from sklearn.cluster import DBSCAN
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Get embeddings
        embeddings = []
        valid_narratives = []
        
        for n in narratives:
            if n['embedding'] is not None:
                embeddings.append(n['embedding'])
                valid_narratives.append(n)
        
        if not embeddings:
            return []
        
        embeddings = np.array(embeddings)
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # DBSCAN clustering with cosine distance
        distance_matrix = 1 - similarity_matrix
        clustering = DBSCAN(eps=0.3, min_samples=2, metric='precomputed')
        labels = clustering.fit_predict(distance_matrix)
        
        # Group narratives by cluster
        clusters = {}
        for i, label in enumerate(labels):
            if label != -1:  # Not noise
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(valid_narratives[i])
        
        # Convert to list format
        cluster_list = []
        for label, members in clusters.items():
            cluster_list.append({
                'cluster_id': label,
                'members': members,
                'size': len(members),
                'theme': self._extract_cluster_theme(members)
            })
        
        return cluster_list
    
    def _extract_cluster_theme(self, narratives: List[Dict]) -> str:
        """Extract common theme from cluster"""
        
        # Use most common narrative type
        types = [n['type'] for n in narratives]
        most_common_type = max(set(types), key=types.count)
        
        # Extract common keywords
        all_text = ' '.join([n['text'] for n in narratives])
        doc = nlp(all_text)
        
        # Get most frequent noun phrases
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]
        if noun_phrases:
            theme = max(set(noun_phrases), key=noun_phrases.count)
        else:
            theme = most_common_type
        
        return theme
    
    def _find_temporal_patterns(self, clusters: List[Dict]) -> List[Dict]:
        """Find patterns that evolve over time"""
        
        temporal_patterns = []
        
        for cluster in clusters:
            # Sort narratives by time
            sorted_narratives = sorted(
                cluster['members'],
                key=lambda x: (x['fiscal_year'], x['fiscal_quarter'] or 'Q0')
            )
            
            # Track evolution
            evolution = self._track_evolution(sorted_narratives)
            
            if evolution['is_evolving']:
                temporal_patterns.append({
                    'theme': cluster['theme'],
                    'occurrence_count': len(sorted_narratives),
                    'evolution': evolution,
                    'narratives': sorted_narratives
                })
        
        return temporal_patterns
    
    def _track_evolution(self, narratives: List[Dict]) -> Dict:
        """Track how narrative evolves over time"""
        
        if len(narratives) < 2:
            return {'is_evolving': False}
        
        # Track sentiment progression
        sentiments = [n['sentiment'] for n in narratives if n['sentiment']]
        
        # Track specificity progression
        specificities = [n['specificity'] for n in narratives if n['specificity']]
        
        # Detect stages based on content changes
        stages = []
        current_stage = {'start_idx': 0, 'characteristics': {}}
        
        for i in range(1, len(narratives)):
            # Check if narrative significantly changed
            if self._has_significant_change(narratives[i-1], narratives[i]):
                # End current stage
                current_stage['end_idx'] = i - 1
                stages.append(current_stage)
                
                # Start new stage
                current_stage = {'start_idx': i, 'characteristics': {}}
        
        # Add final stage
        current_stage['end_idx'] = len(narratives) - 1
        stages.append(current_stage)
        
        # Analyze each stage
        for stage in stages:
            stage_narratives = narratives[stage['start_idx']:stage['end_idx']+1]
            stage['characteristics'] = self._analyze_stage(stage_narratives)
        
        return {
            'is_evolving': len(stages) > 1,
            'stages': stages,
            'sentiment_trend': self._calculate_trend(sentiments),
            'specificity_trend': self._calculate_trend(specificities)
        }
    
    def _has_significant_change(self, n1: Dict, n2: Dict) -> bool:
        """Detect if narrative significantly changed"""
        
        # Sentiment change
        if n1['sentiment'] and n2['sentiment']:
            if abs(n1['sentiment'] - n2['sentiment']) > 0.3:
                return True
        
        # Specificity change
        if n1['specificity'] and n2['specificity']:
            if abs(n1['specificity'] - n2['specificity']) > 0.3:
                return True
        
        # Embedding similarity
        if n1['embedding'] is not None and n2['embedding'] is not None:
            similarity = cosine_similarity(
                n1['embedding'].reshape(1, -1),
                n2['embedding'].reshape(1, -1)
            )[0][0]
            if similarity < 0.7:  # Significant change
                return True
        
        return False
    
    def _analyze_stage(self, narratives: List[Dict]) -> Dict:
        """Analyze characteristics of a stage"""
        
        # Extract common keywords
        all_text = ' '.join([n['text'] for n in narratives])
        doc = nlp(all_text)
        
        keywords = []
        for token in doc:
            if token.pos_ in ['NOUN', 'VERB'] and not token.is_stop:
                keywords.append(token.lemma_)
        
        # Most common keywords
        from collections import Counter
        keyword_counts = Counter(keywords)
        top_keywords = [k for k, _ in keyword_counts.most_common(5)]
        
        return {
            'keywords': top_keywords,
            'avg_sentiment': np.mean([n['sentiment'] for n in narratives if n['sentiment']]),
            'avg_specificity': np.mean([n['specificity'] for n in narratives if n['specificity']]),
            'narrative_count': len(narratives)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        
        if not values or len(values) < 2:
            return 'stable'
        
        # Simple linear regression
        x = np.arange(len(values))
        y = np.array(values)
        
        # Remove None values
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return 'stable'
        
        x = x[mask]
        y = y[mask]
        
        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]
        
        if slope > 0.1:
            return 'increasing'
        elif slope < -0.1:
            return 'decreasing'
        else:
            return 'stable'
    
    def _define_pattern(self, temporal_pattern: Dict) -> Dict:
        """Define a discovered pattern"""
        
        evolution = temporal_pattern['evolution']
        
        # Generate pattern name
        theme = temporal_pattern['theme']
        trend = evolution['sentiment_trend']
        pattern_name = f"{theme}_{trend}_pattern"
        
        # Extract stage markers
        stage_markers = {}
        for i, stage in enumerate(evolution['stages']):
            stage_name = f"stage_{i+1}"
            stage_markers[stage_name] = stage['characteristics']['keywords']
        
        return {
            'name': pattern_name,
            'category': temporal_pattern['narratives'][0]['type'],
            'theme': theme,
            'stages': evolution['stages'],
            'stage_markers': stage_markers,
            'occurrence_count': temporal_pattern['occurrence_count'],
            'confidence': self._calculate_pattern_confidence(temporal_pattern)
        }
    
    def _calculate_pattern_confidence(self, pattern: Dict) -> float:
        """Calculate confidence in discovered pattern"""
        
        confidence = 0.5
        
        # More occurrences = higher confidence
        if pattern['occurrence_count'] > 10:
            confidence += 0.2
        elif pattern['occurrence_count'] > 5:
            confidence += 0.1
        
        # Clear evolution = higher confidence
        if pattern['evolution']['is_evolving']:
            confidence += 0.1
        
        # Consistent trend = higher confidence
        sentiment_trend = pattern['evolution']['sentiment_trend']
        if sentiment_trend in ['increasing', 'decreasing']:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _store_patterns(self, patterns: List[Dict]) -> int:
        """Store discovered patterns"""
        
        cur = self.conn.cursor()
        stored = 0
        
        for pattern in patterns:
            try:
                cur.execute("""
                    INSERT INTO discovered_patterns
                    (pattern_name, pattern_category, stages, stage_markers,
                     discovered_from_company, confidence_score)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING pattern_id
                """, (
                    pattern['name'],
                    pattern['category'],
                    json.dumps(pattern['stages']),
                    json.dumps(pattern['stage_markers']),
                    'HDFC',  # Hardcoded for now
                    pattern['confidence']
                ))
                
                pattern_id = cur.fetchone()[0]
                pattern['pattern_id'] = pattern_id
                stored += 1
                
            except Exception as e:
                logger.error(f"Failed to store pattern: {e}")
                self.conn.rollback()
        
        self.conn.commit()
        return stored

# Testing function
def test_narrative_extraction():
    """Test narrative extraction and pattern discovery"""
    
    print("🧪 Testing Narrative Extraction & Pattern Discovery\n")
    
    # Initialize
    extractor = NarrativeExtractor()
    pattern_engine = PatternDiscoveryEngine()
    
    # Load a processed document
    cur = config.conn.cursor()
    cur.execute("""
        SELECT doc_id, company 
        FROM document_registry 
        WHERE processing_status = 'completed'
        LIMIT 1
    """)
    
    result = cur.fetchone()
    if not result:
        print("❌ No processed documents found")
        return
        
    doc_id, company = result
    
    # Mock structure for testing
    structure = {
        'sections': {
            'Management Discussion': {
                'elements': [{
                    'type': 'paragraph',
                    'content': """We expect our Net Interest Margin to improve by 
                    15-20 basis points in the coming quarters as we benefit from 
                    the rising rate environment. Our CASA ratio has improved to 
                    48.5% which provides us with a cost advantage.""",
                    'page': 15
                }]
            }
        }
    }
    
    print(f"1️⃣ Extracting narratives from {company} document...")
    narratives = extractor.extract_narratives(doc_id, structure)
    print(f"   Extracted {len(narratives)} narratives")
    
    if narratives:
        print("\n   Sample narrative:")
        n = narratives[0]
        print(f"   Type: {n['narrative_type']}")
        print(f"   Confidence: {n['confidence_score']:.2f}")
        print(f"   Sentiment: {n['sentiment']['label']}")
        print(f"   Specificity: {n['specificity_score']:.2f}")
    
    print(f"\n2️⃣ Discovering patterns for {company}...")
    patterns = pattern_engine.discover_patterns(company)
    print(f"   Discovered {len(patterns)} patterns")
    
    if patterns:
        print("\n   Sample pattern:")
        p = patterns[0]
        print(f"   Name: {p['name']}")
        print(f"   Category: {p['category']}")
        print(f"   Confidence: {p['confidence']:.2f}")
        print(f"   Stages: {len(p['stages'])}")

if __name__ == "__main__":
    test_narrative_extraction()
