# Originally ingestion/google_drive_monitor.py (Drive scan + Document AI + PyMuPDF)
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import io
from googleapiclient.discovery import Resource
from googleapiclient.http import MediaIoBaseDownload
from google.cloud import documentai_v1 as documentai
import fitz  # PyMuPDF
import pandas as pd
from pathlib import Path

from config.setup import config, logger

class AuditTrail:
    """Centralized audit trail for all operations"""
    
    def __init__(self):
        self.conn = config.conn
        self.audit_logger = config.audit_logger
        
    def log_operation(self, operation_type: str, operation_id: str, 
                     input_data: Dict, output_data: Dict = None,
                     success: bool = True, error: str = None,
                     learnings: Dict = None, company: str = None) -> str:
        """Log every operation with full context"""
        
        audit_entry = {
            'operation_type': operation_type,
            'operation_id': operation_id,
            'timestamp': datetime.utcnow().isoformat(),
            'company': company,
            'input_data': input_data,
            'output_data': output_data,
            'success': success,
            'error_details': error,
            'learnings': learnings or {}
        }
        
        # Log to file (JSONL format for easy parsing)
        self.audit_logger.info(json.dumps(audit_entry))
        
        # Store in database
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO audit_trail 
            (operation_type, operation_id, company, input_data, output_data, 
             success, error_details, learnings)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING audit_id
        """, (
            operation_type, operation_id, company,
            json.dumps(input_data), json.dumps(output_data) if output_data else None,
            success, error, json.dumps(learnings) if learnings else None
        ))
        
        audit_id = cur.fetchone()[0]
        self.conn.commit()
        
        return audit_id
    
    def log_learning(self, learning_type: str, observation: str, 
                    improvement: Dict, context: Dict) -> None:
        """Log what we learned for future improvement"""
        
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO system_learning
            (learning_type, observation, improvement_action, learning_context)
            VALUES (%s, %s, %s, %s)
        """, (
            learning_type, observation, 
            json.dumps(improvement), json.dumps(context)
        ))
        self.conn.commit()

# Global audit trail instance
audit = AuditTrail()

class GoogleDriveMonitor:
    """Monitor Google Drive for banking documents"""
    
    def __init__(self, folder_id: str):
        self.drive_service = config.drive_service
        self.folder_id = folder_id
        self.processed_files = set()
        self._load_processed_files()
        
    def _load_processed_files(self):
        """Load already processed files from database"""
        cur = config.conn.cursor()
        cur.execute("SELECT google_drive_id FROM document_registry")
        self.processed_files = {row[0] for row in cur.fetchall()}
        
    def scan_folder(self, company_filter: str = None) -> List[Dict]:
        """Scan Drive folder for new documents"""
        
        operation_id = f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        audit.log_operation(
            'drive_scan', operation_id,
            {'folder_id': self.folder_id, 'company_filter': company_filter}
        )
        
        try:
            # Query for files
            query = f"'{self.folder_id}' in parents and trashed = false"
            if company_filter:
                query += f" and name contains '{company_filter}'"
                
            results = self.drive_service.files().list(
                q=query,
                fields="files(id, name, mimeType, size, createdTime, modifiedTime)",
                pageSize=100
            ).execute()
            
            files = results.get('files', [])
            new_files = []
            
            for file in files:
                if file['id'] not in self.processed_files:
                    # Classify document type
                    doc_type = self._classify_document(file['name'])
                    if doc_type:
                        file['doc_type'] = doc_type
                        file['company'] = self._extract_company(file['name'])
                        file['period'] = self._extract_period(file['name'])
                        new_files.append(file)
            
            audit.log_operation(
                'drive_scan', operation_id,
                {'folder_id': self.folder_id},
                {'new_files_found': len(new_files)},
                success=True
            )
            
            return new_files
            
        except Exception as e:
            audit.log_operation(
                'drive_scan', operation_id,
                {'folder_id': self.folder_id},
                error=str(e),
                success=False
            )
            raise
    
    def _classify_document(self, filename: str) -> Optional[str]:
        """Classify document type from filename"""
        
        filename_lower = filename.lower()
        
        # Document type patterns
        patterns = {
            'annual_report': ['annual report', 'ar_20', 'annual_20'],
            'quarterly_result': ['quarterly', 'q1', 'q2', 'q3', 'q4', 'quarter'],
            'earnings_call': ['earnings', 'transcript', 'call'],
            'investor_presentation': ['investor', 'presentation', 'ppt'],
            'regulatory_filing': ['filing', 'regulation', 'shareholding']
        }
        
        for doc_type, keywords in patterns.items():
            if any(keyword in filename_lower for keyword in keywords):
                return doc_type
                
        return None
    
    def _extract_company(self, filename: str) -> str:
        """Extract company name from filename"""
        
        # Known bank patterns
        banks = {
            'HDFC': ['hdfc', 'hdfcbank'],
            'ICICI': ['icici', 'icicibank'],
            'KOTAK': ['kotak', 'kotakbank'],
            'AXIS': ['axis', 'axisbank'],
            'FEDERAL': ['federal', 'federalbank'],
            'CITYUNION': ['city union', 'cub', 'cityunion']
        }
        
        filename_lower = filename.lower()
        for bank, patterns in banks.items():
            if any(pattern in filename_lower for pattern in patterns):
                return bank
                
        return 'UNKNOWN'
    
    def _extract_period(self, filename: str) -> Dict:
        """Extract fiscal period from filename"""
        
        import re
        
        # Year patterns
        year_match = re.search(r'20(1[5-9]|2[0-9])', filename)
        year = int(year_match.group()) if year_match else None
        
        # Quarter patterns
        quarter_match = re.search(r'Q([1-4])', filename, re.IGNORECASE)
        quarter = f"Q{quarter_match.group(1)}" if quarter_match else None
        
        # Fiscal year pattern (FY20, FY2024)
        fy_match = re.search(r'FY\s?(\d{2,4})', filename, re.IGNORECASE)
        if fy_match:
            fy = fy_match.group(1)
            if len(fy) == 2:
                year = 2000 + int(fy)
            else:
                year = int(fy)
        
        return {'year': year, 'quarter': quarter}
    
    def download_file(self, file_id: str, file_name: str) -> bytes:
        """Download file from Google Drive"""
        
        operation_id = f"download_{file_id}"
        
        try:
            request = self.drive_service.files().get_media(fileId=file_id)
            file_data = io.BytesIO()
            downloader = MediaIoBaseDownload(file_data, request)
            
            done = False
            while not done:
                status, done = downloader.next_chunk()
                
            file_data.seek(0)
            content = file_data.read()
            
            audit.log_operation(
                'file_download', operation_id,
                {'file_id': file_id, 'file_name': file_name},
                {'size_bytes': len(content)},
                success=True
            )
            
            return content
            
        except Exception as e:
            audit.log_operation(
                'file_download', operation_id,
                {'file_id': file_id, 'file_name': file_name},
                error=str(e),
                success=False
            )
            raise

class DocumentAIExtractor:
    """Extract structured content using Document AI"""
    
    def __init__(self):
        self.client = config.docai_client
        self.processor_name = config.processor_name
        
    def process_document(self, content: bytes, mime_type: str, 
                        doc_id: str) -> documentai.Document:
        """Process document with Document AI"""
        
        operation_id = f"docai_{doc_id}"
        
        try:
            # Create request
            raw_document = documentai.RawDocument(content=content, mime_type=mime_type)
            request = documentai.ProcessRequest(
                name=self.processor_name,
                raw_document=raw_document
            )
            
            # Process document
            start_time = datetime.now()
            result = self.client.process_document(request=request)
            process_time = (datetime.now() - start_time).total_seconds()
            
            document = result.document
            
            # Extract metrics
            metrics = {
                'page_count': len(document.pages),
                'entity_count': len(document.entities),
                'table_count': sum(len(page.tables) for page in document.pages),
                'confidence': document.pages[0].layout.confidence if document.pages else 0,
                'process_time_seconds': process_time
            }
            
            audit.log_operation(
                'document_ai_extraction', operation_id,
                {'doc_id': doc_id, 'mime_type': mime_type},
                metrics,
                success=True
            )
            
            # Learn from extraction
            if metrics['confidence'] < 0.8:
                audit.log_learning(
                    'extraction',
                    f"Low confidence extraction: {metrics['confidence']}",
                    {'action': 'review_document_quality', 'threshold': 0.8},
                    {'doc_id': doc_id, 'metrics': metrics}
                )
            
            return document
            
        except Exception as e:
            audit.log_operation(
                'document_ai_extraction', operation_id,
                {'doc_id': doc_id},
                error=str(e),
                success=False
            )
            raise

class HierarchicalExtractor:
    """Extract hierarchical structure with special attention to banking documents"""
    
    def __init__(self):
        self.section_patterns = {
            'annual_report': {
                'primary_sections': [
                    'Management Discussion and Analysis',
                    'Directors Report',
                    'Corporate Governance',
                    'Financial Statements',
                    'Independent Auditors Report',
                    'Risk Management'
                ],
                'key_subsections': {
                    'MD&A': [
                        'Financial Performance',
                        'Business Overview',
                        'Digital Initiatives',
                        'Asset Quality',
                        'Capital Adequacy'
                    ],
                    'Risk': [
                        'Credit Risk',
                        'Market Risk', 
                        'Operational Risk',
                        'Liquidity Risk'
                    ]
                }
            }
        }
        
    def extract_hierarchy(self, doc_ai_document: documentai.Document, 
                         pdf_content: bytes, doc_type: str) -> Dict:
        """Extract hierarchical structure from document"""
        
        operation_id = f"hierarchy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # First try Document AI structure
            dai_structure = self._extract_dai_structure(doc_ai_document)
            
            # Then enhance with PyMuPDF for better hierarchy
            pdf_structure = self._extract_pdf_structure(pdf_content)
            
            # Merge structures
            merged_structure = self._merge_structures(dai_structure, pdf_structure)
            
            # Apply banking-specific patterns
            enhanced_structure = self._apply_domain_patterns(
                merged_structure, doc_type
            )
            
            audit.log_operation(
                'hierarchy_extraction', operation_id,
                {'doc_type': doc_type},
                {
                    'sections_found': len(enhanced_structure['sections']),
                    'total_elements': enhanced_structure['element_count']
                },
                success=True
            )
            
            return enhanced_structure
            
        except Exception as e:
            audit.log_operation(
                'hierarchy_extraction', operation_id,
                {'doc_type': doc_type},
                error=str(e),
                success=False
            )
            raise
    
    def _extract_dai_structure(self, document: documentai.Document) -> Dict:
        """Extract structure from Document AI results"""
        
        structure = {
            'sections': {},
            'headings': [],
            'paragraphs': [],
            'tables': [],
            'visuals': []
        }
        
        for page_num, page in enumerate(document.pages):
            # Extract paragraphs with layout info
            for paragraph in page.paragraphs:
                para_text = self._get_text(paragraph.layout, document.text)
                if para_text:
                    structure['paragraphs'].append({
                        'text': para_text,
                        'page': page_num,
                        'confidence': paragraph.layout.confidence,
                        'bbox': self._get_bbox(paragraph.layout.bounding_poly)
                    })
            
            # Extract tables
            for table in page.tables:
                structure['tables'].append({
                    'page': page_num,
                    'rows': len(table.body_rows),
                    'cols': len(table.header_rows[0].cells) if table.header_rows else 0,
                    'data': self._extract_table_data(table, document.text)
                })
            
            # Extract form fields (useful for structured sections)
            for field in page.form_fields:
                field_name = self._get_text(field.field_name, document.text)
                field_value = self._get_text(field.field_value, document.text)
                
                # Use form fields to identify sections
                if field_name and 'section' in field_name.lower():
                    structure['sections'][field_name] = {
                        'page_start': page_num,
                        'content': field_value
                    }
        
        return structure
    
    def _extract_pdf_structure(self, pdf_content: bytes) -> Dict:
        """Extract structure using PyMuPDF for better hierarchy detection"""
        
        pdf = fitz.open(stream=pdf_content, filetype="pdf")
        structure = {
            'sections': {},
            'headings': [],
            'toc': []
        }
        
        # Try to get table of contents
        toc = pdf.get_toc()
        if toc:
            structure['toc'] = [
                {'level': level, 'title': title, 'page': page}
                for level, title, page in toc
            ]
            
            # Build section hierarchy from TOC
            for level, title, page in toc:
                if level == 1:  # Main sections
                    structure['sections'][title] = {
                        'page_start': page - 1,  # 0-indexed
                        'subsections': {}
                    }
        
        # Extract headings by font size
        for page_num, page in enumerate(pdf):
            blocks = page.get_text("dict")
            
            for block in blocks["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            # Identify headings by font size
                            if span["size"] > 14:  # Likely heading
                                structure['headings'].append({
                                    'text': span["text"].strip(),
                                    'page': page_num,
                                    'size': span["size"],
                                    'font': span["font"],
                                    'bbox': span["bbox"]
                                })
        
        pdf.close()
        return structure
    
    def _merge_structures(self, dai_structure: Dict, pdf_structure: Dict) -> Dict:
        """Intelligently merge Document AI and PDF structures"""
        
        merged = {
            'sections': {},
            'elements': [],
            'hierarchy': {},
            'element_count': 0
        }
        
        # Start with PDF TOC as base hierarchy
        if pdf_structure.get('toc'):
            for item in pdf_structure['toc']:
                section_key = item['title']
                merged['sections'][section_key] = {
                    'level': item['level'],
                    'page_start': item['page'],
                    'elements': []
                }
        
        # Map Document AI elements to sections
        current_section = None
        for para in dai_structure['paragraphs']:
            # Find which section this paragraph belongs to
            for section, info in merged['sections'].items():
                if para['page'] >= info['page_start']:
                    current_section = section
            
            if current_section:
                merged['sections'][current_section]['elements'].append({
                    'type': 'paragraph',
                    'content': para['text'],
                    'page': para['page'],
                    'confidence': para['confidence']
                })
                merged['element_count'] += 1
        
        # Add tables and visuals
        for table in dai_structure['tables']:
            merged['elements'].append({
                'type': 'table',
                'page': table['page'],
                'data': table['data']
            })
            merged['element_count'] += 1
        
        return merged
    
    def _apply_domain_patterns(self, structure: Dict, doc_type: str) -> Dict:
        """Apply banking-specific patterns to enhance structure"""
        
        if doc_type not in self.section_patterns:
            return structure
            
        patterns = self.section_patterns[doc_type]
        
        # Look for expected sections
        for expected_section in patterns['primary_sections']:
            found = False
            for section in structure['sections']:
                if expected_section.lower() in section.lower():
                    found = True
                    # Enhance with expected subsections
                    if expected_section in patterns.get('key_subsections', {}):
                        structure['sections'][section]['expected_subsections'] = \
                            patterns['key_subsections'][expected_section]
                    break
            
            if not found:
                # Log missing expected section
                audit.log_learning(
                    'structure',
                    f"Expected section '{expected_section}' not found",
                    {'action': 'refine_section_patterns'},
                    {'doc_type': doc_type, 'sections_found': list(structure['sections'].keys())}
                )
        
        return structure
    
    def _get_text(self, layout, document_text: str) -> str:
        """Extract text from Document AI layout"""
        if not layout.text_anchor.text_segments:
            return ""
        
        text = ""
        for segment in layout.text_anchor.text_segments:
            start = segment.start_index
            end = segment.end_index
            text += document_text[start:end]
        
        return text.strip()
    
    def _get_bbox(self, bounding_poly) -> Dict:
        """Convert bounding poly to bbox dict"""
        if not bounding_poly.vertices:
            return {}
            
        return {
            'x1': bounding_poly.vertices[0].x,
            'y1': bounding_poly.vertices[0].y,
            'x2': bounding_poly.vertices[2].x,
            'y2': bounding_poly.vertices[2].y
        }
    
    def _extract_table_data(self, table, document_text: str) -> List[List[str]]:
        """Extract table data into 2D array"""
        data = []
        
        # Headers
        if table.header_rows:
            for row in table.header_rows:
                row_data = []
                for cell in row.cells:
                    cell_text = self._get_text(cell.layout, document_text)
                    row_data.append(cell_text)
                data.append(row_data)
        
        # Body
        for row in table.body_rows:
            row_data = []
            for cell in row.cells:
                cell_text = self._get_text(cell.layout, document_text)
                row_data.append(cell_text)
            data.append(row_data)
        
        return data

class VisualElementExtractor:
    """Special handling for charts, graphs, and complex tables"""
    
    def __init__(self):
        self.visual_patterns = {
            'chart_keywords': ['chart', 'graph', 'figure', 'exhibit'],
            'table_keywords': ['table', 'schedule', 'annexure']
        }
        
    def extract_visuals(self, doc_ai_document: documentai.Document, 
                       pdf_content: bytes) -> List[Dict]:
        """Extract and understand visual elements"""
        
        visuals = []
        
        # Extract from Document AI
        for page_num, page in enumerate(doc_ai_document.pages):
            # Get visual elements (non-text)
            for visual in page.visual_elements:
                visual_type = visual.type_
                
                visual_data = {
                    'page': page_num,
                    'type': visual_type,
                    'confidence': visual.confidence,
                    'bbox': self._get_bbox(visual.layout.bounding_poly)
                }
                
                # Try to find associated caption/title
                caption = self._find_visual_caption(visual, page, doc_ai_document.text)
                if caption:
                    visual_data['caption'] = caption
                    visual_data['extracted_meaning'] = self._interpret_visual(
                        visual_type, caption
                    )
                
                visuals.append(visual_data)
        
        # Enhance with PyMuPDF image extraction
        pdf_visuals = self._extract_pdf_images(pdf_content)
        
        # Merge and deduplicate
        merged_visuals = self._merge_visuals(visuals, pdf_visuals)
        
        logger.info(f"Extracted {len(merged_visuals)} visual elements")
        
        return merged_visuals
    
    def _find_visual_caption(self, visual, page, document_text: str) -> Optional[str]:
        """Find caption or title near visual element"""
        
        # Look for text elements near the visual
        visual_bbox = visual.layout.bounding_poly
        
        for paragraph in page.paragraphs:
            para_bbox = paragraph.layout.bounding_poly
            
            # Check if paragraph is near visual (above or below)
            if self._is_near(visual_bbox, para_bbox):
                text = self._get_text(paragraph.layout, document_text)
                
                # Check if it looks like a caption
                if any(keyword in text.lower() for keyword in self.visual_patterns['chart_keywords']):
                    return text
                    
        return None
    
    def _is_near(self, bbox1, bbox2, threshold: int = 50) -> bool:
        """Check if two bounding boxes are near each other"""
        
        # Simple proximity check
        center1_y = (bbox1.vertices[0].y + bbox1.vertices[2].y) / 2
        center2_y = (bbox2.vertices[0].y + bbox2.vertices[2].y) / 2
        
        return abs(center1_y - center2_y) < threshold
    
    def _interpret_visual(self, visual_type: str, caption: str) -> str:
        """Interpret what the visual is communicating"""
        
        # Banking-specific interpretations
        interpretations = {
            'revenue': 'Shows revenue trends and growth patterns',
            'nim': 'Illustrates Net Interest Margin evolution',
            'asset quality': 'Depicts NPAs and provisioning trends',
            'digital': 'Demonstrates digital adoption metrics',
            'branch': 'Shows branch network expansion'
        }
        
        caption_lower = caption.lower()
        for key, interpretation in interpretations.items():
            if key in caption_lower:
                return interpretation
                
        return f"Visual element of type {visual_type}"
    
    def _extract_pdf_images(self, pdf_content: bytes) -> List[Dict]:
        """Extract actual images from PDF"""
        
        pdf = fitz.open(stream=pdf_content, filetype="pdf")
        images = []
        
        for page_num, page in enumerate(pdf):
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                
                # Get image metadata
                images.append({
                    'page': page_num,
                    'type': 'embedded_image',
                    'xref': xref,
                    'index': img_index
                })
        
        pdf.close()
        return images
    
    def _merge_visuals(self, dai_visuals: List[Dict], 
                      pdf_visuals: List[Dict]) -> List[Dict]:
        """Merge visual elements from different sources"""
        
        # Simple merge for now - can be enhanced with deduplication
        all_visuals = dai_visuals + pdf_visuals
        
        # Sort by page number
        all_visuals.sort(key=lambda x: x['page'])
        
        return all_visuals
    
    def _get_bbox(self, bounding_poly) -> Dict:
        """Convert bounding poly to bbox dict"""
        if not bounding_poly.vertices:
            return {}
            
        return {
            'x1': bounding_poly.vertices[0].x,
            'y1': bounding_poly.vertices[0].y,
            'x2': bounding_poly.vertices[2].x,
            'y2': bounding_poly.vertices[2].y
        }
    
    def _get_text(self, layout, document_text: str) -> str:
        """Extract text from Document AI layout"""
        if not layout.text_anchor.text_segments:
            return ""
        
        text = ""
        for segment in layout.text_anchor.text_segments:
            start = segment.start_index
            end = segment.end_index
            text += document_text[start:end]
        
        return text.strip()

# Testing function
def test_document_pipeline():
    """Test the complete document ingestion pipeline"""
    
    print("🧪 Testing Document Ingestion Pipeline\n")
    
    # Initialize components
    drive_monitor = GoogleDriveMonitor(folder_id=os.getenv('GDRIVE_FOLDER_ID', ''))
    doc_ai = DocumentAIExtractor()
    hierarchy_extractor = HierarchicalExtractor()
    visual_extractor = VisualElementExtractor()
    
    # Scan for HDFC documents
    print("1️⃣ Scanning Google Drive...")
    new_files = drive_monitor.scan_folder(company_filter='HDFC')
    print(f"   Found {len(new_files)} new files")
    
    if new_files:
        # Process first file
        file = new_files[0]
        print(f"\n2️⃣ Processing: {file['name']}")
        
        # Download
        content = drive_monitor.download_file(file['id'], file['name'])
        print(f"   Downloaded: {len(content)/1024/1024:.1f} MB")
        
        # Extract with Document AI
        print("\n3️⃣ Running Document AI extraction...")
        doc_ai_result = doc_ai.process_document(
            content, 'application/pdf', file['id']
        )
        print(f"   Pages: {len(doc_ai_result.pages)}")
        
        # Extract hierarchy
        print("\n4️⃣ Extracting hierarchical structure...")
        hierarchy = hierarchy_extractor.extract_hierarchy(
            doc_ai_result, content, file['doc_type']
        )
        print(f"   Sections found: {len(hierarchy['sections'])}")
        
        # Extract visuals
        print("\n5️⃣ Extracting visual elements...")
        visuals = visual_extractor.extract_visuals(doc_ai_result, content)
        print(f"   Visuals found: {len(visuals)}")
        
        print("\n✅ Pipeline test complete!")
        
        # Show sample structure
        print("\n📊 Sample Document Structure:")
        for section, info in list(hierarchy['sections'].items())[:3]:
            print(f"   - {section} (Page {info.get('page_start', 'N/A')})")

if __name__ == "__main__":
    test_document_pipeline()
