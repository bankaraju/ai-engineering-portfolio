# document_structure.py
"""Recover chapter / section / figure / table structure from a PDF using
PyMuPDF font-size and bold-weight cues. Preserved from a 2025 project;
lightly cleaned (see README)."""
import fitz  # PyMuPDF
import os
import json
import re
import logging
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('doc-structure')

class DocumentStructureAnalyzer:
    """Analyzes PDF document structure including chapters, sections, tables, and figures"""
    
    def __init__(self, pdf_path, output_dir=None):
        self.pdf_path = pdf_path
        self.output_dir = output_dir or os.path.join(os.path.dirname(pdf_path), "extracted")
        self.doc = None
        self.text_blocks = []
        self.chapters = []
        self.sections = []
        self.figures = []
        self.tables = []
        self.structure = {}
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "chapters"), exist_ok=True)
    
    def analyze(self):
        """Run the complete analysis pipeline"""
        try:
            self.load_document()
            self.extract_text_blocks()
            self.identify_chapters()
            self.identify_sections()
            self.extract_figures_and_tables()
            self.build_structure()
            self.save_structure()
            self.extract_chapters_text()
            return self.structure
        except Exception as e:
            logger.error(f"Error analyzing document: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def load_document(self):
        """Load the PDF document"""
        logger.info(f"Loading document: {self.pdf_path}")
        self.doc = fitz.open(self.pdf_path)
        logger.info(f"Document loaded: {len(self.doc)} pages")
    
    def extract_text_blocks(self):
        """Extract text blocks with formatting information"""
        logger.info("Extracting text blocks...")
        
        for page_idx, page in enumerate(tqdm(self.doc, desc="Processing pages")):
            # Get text blocks with formatting information
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if block.get("type") == 0:  # Text block
                    for line in block.get("lines", []):
                        text = ""
                        # Get text size and font information from spans
                        max_size = 0
                        is_bold = False
                        
                        for span in line.get("spans", []):
                            text += span.get("text", "")
                            max_size = max(max_size, span.get("size", 0))
                            # Check if bold by examining font name
                            if "bold" in span.get("font", "").lower():
                                is_bold = True
                        
                        if text.strip():
                            self.text_blocks.append({
                                "page": page_idx + 1,
                                "text": text.strip(),
                                "size": max_size,
                                "is_bold": is_bold,
                                "bbox": line.get("bbox"),  # Bounding box [x0, y0, x1, y1]
                                "block_type": "text"
                            })
        
        logger.info(f"Extracted {len(self.text_blocks)} text blocks")
    
    def identify_chapters(self):
        """Identify chapter headings in the document"""
        logger.info("Identifying chapters...")
        
        # Chapter patterns to look for
        chapter_patterns = [
            re.compile(r'^CHAPTER\s+(\d+)\s*[:.-]\s*(.+)$', re.IGNORECASE),
            re.compile(r'^(\d+)\.\s+(.+)$'),
            # Add more patterns as needed
        ]
        
        # Calculate text size statistics for heading detection
        sizes = [block["size"] for block in self.text_blocks if block["size"] > 0]
        if sizes:
            median_size = np.median(sizes)
            large_size_threshold = median_size * 1.3  # 30% larger than median
            
            # Find potential chapter headings
            for i, block in enumerate(self.text_blocks):
                text = block["text"]
                is_potential_heading = (
                    block["size"] > large_size_threshold or 
                    block["is_bold"]
                )
                
                if is_potential_heading:
                    for pattern in chapter_patterns:
                        match = pattern.match(text)
                        if match:
                            chapter_num = match.group(1)
                            chapter_title = match.group(2).strip()
                            
                            self.chapters.append({
                                "index": i,
                                "page": block["page"],
                                "number": chapter_num,
                                "title": chapter_title,
                                "full_text": text,
                                "size": block["size"]
                            })
                            break
        
        # If no chapters found, try alternative detection methods
        if not self.chapters:
            logger.info("No standard chapter patterns found, trying alternative methods...")
            self._identify_chapters_by_formatting()
        
        logger.info(f"Identified {len(self.chapters)} chapters")
    
    def _identify_chapters_by_formatting(self):
        """Identify chapters by text formatting and positioning when standard patterns fail"""
        # Group blocks by size and formatting
        size_groups = defaultdict(list)
        
        for i, block in enumerate(self.text_blocks):
            if block["is_bold"] and block["size"] > 0:
                size_groups[block["size"]].append((i, block))
        
        # Find the largest size that appears regularly and contains numeric indicators
        candidate_sizes = sorted(size_groups.keys(), reverse=True)
        
        for size in candidate_sizes[:3]:  # Check the top 3 sizes
            blocks = size_groups[size]
            
            # Check if these might be chapter headings
            chapter_count = 0
            for i, block in blocks:
                text = block["text"]
                
                # Check for numeric indicators
                if re.search(r'\d+', text) and len(text.split()) < 10:
                    chapter_count += 1
 # If we have a reasonable number of potential chapters
            if 3 <= chapter_count <= 100:
                chapter_num = 1
                
                for i, block in blocks:
                    text = block["text"]
                    
                    # Try to extract chapter number
                    num_match = re.search(r'(\d+)', text)
                    if num_match:
                        chapter_num = num_match.group(1)
                    
                    # Try to extract title
                    title = text
                    title = re.sub(r'^CHAPTER\s+\d+\s*[:.-]\s*', '', title, flags=re.IGNORECASE)
                    title = re.sub(r'^\d+\.\s*', '', title)
                    
                    self.chapters.append({
                        "index": i,
                        "page": block["page"],
                        "number": chapter_num,
                        "title": title.strip(),
                        "full_text": text,
                        "size": block["size"]
                    })
                    
                    chapter_num = str(int(chapter_num) + 1) if str(chapter_num).isdigit() else chapter_num
                
                break
    
    def identify_sections(self):
        """Identify sections and subsections within chapters"""
        logger.info("Identifying sections...")
        
        if not self.chapters:
            logger.warning("No chapters identified, skipping section identification")
            return
        
        # Calculate text size statistics for section detection
        sizes = [block["size"] for block in self.text_blocks if block["size"] > 0]
        if not sizes:
            return
            
        median_size = np.median(sizes)
        section_size_threshold = median_size * 1.15  # 15% larger than median
        
        # Iterate through chapters
        for i, chapter in enumerate(self.chapters):
            chapter_start_idx = chapter["index"]
            chapter_end_idx = self.chapters[i+1]["index"] if i < len(self.chapters)-1 else len(self.text_blocks)
            
            # Find sections within chapter range
            chapter_sections = []
            
            for j in range(chapter_start_idx + 1, chapter_end_idx):
                block = self.text_blocks[j]
                
                # Potential section heading
                is_potential_section = (
                    block["size"] >= section_size_threshold or 
                    block["is_bold"]
                ) and len(block["text"].split()) < 15  # Not too long
                
                if is_potential_section:
                    # Check if it's not part of a paragraph
                    if j < len(self.text_blocks) - 1:
                        next_block = self.text_blocks[j+1]
                        if next_block["size"] < block["size"] or next_block["is_bold"] == False:
                            section_text = block["text"].strip()
                            
                            # Skip if it looks like regular text
                            if len(section_text.split()) > 10 or section_text.endswith('.'):
                                continue
                                
                            chapter_sections.append({
                                "index": j,
                                "page": block["page"],
                                "title": section_text,
                                "level": 2 if block["size"] < chapter["size"] else 1
                            })
            
            # Add sections to the chapter
            self.chapters[i]["sections"] = chapter_sections
            # Add to global sections list
            self.sections.extend(chapter_sections)
        
        logger.info(f"Identified {len(self.sections)} sections across all chapters")
    
    def extract_figures_and_tables(self):
        """Extract figures and tables from the document"""
        logger.info("Extracting figures and tables...")
        
        # Process each page for images
        for page_idx, page in enumerate(tqdm(self.doc, desc="Extracting figures")):
            # Get images
            image_list = page.get_images(full=True)
            
            for img_idx, img in enumerate(image_list):
                xref = img[0]
                
                try:
                    base_image = self.doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    extension = base_image["ext"]
                    
                    # Save the image
                    image_filename = f"page_{page_idx+1}_img_{img_idx+1}.{extension}"
                    image_path = os.path.join(self.output_dir, "images", image_filename)
                    
                    with open(image_path, "wb") as img_file:
                        img_file.write(image_bytes)
                    
                    # Try to find caption
                    caption = self._find_figure_caption(page_idx, img)
                    
                    self.figures.append({
                        "page": page_idx + 1,
                        "filename": image_filename,
                        "path": image_path,
                        "caption": caption,
                        "width": base_image.get("width", 0),
                        "height": base_image.get("height", 0)
                    })
                except Exception as e:
                    logger.warning(f"Error extracting image on page {page_idx+1}: {e}")
 # Process each page for tables
        for page_idx, page in enumerate(tqdm(self.doc, desc="Detecting tables")):
            # Use heuristics to detect tables
            # Tables often have consistent spacing and alignment
            blocks = page.get_text("dict")["blocks"]
            
            for block_idx, block in enumerate(blocks):
                if block.get("type") == 0:  # Text block
                    lines = block.get("lines", [])
                    
                    # Check for table-like structure
                    if len(lines) >= 3:  # Need at least a few rows
                        # Check if lines have similar structure (potential table)
                        col_positions = self._analyze_potential_table_structure(lines)
                        
                        if col_positions and len(col_positions) >= 2:  # At least 2 columns
                            # Extract table content
                            table_data = self._extract_table_data(lines, col_positions)
                            
                            if table_data and len(table_data) >= 2:  # At least 2 rows
                                # Try to find caption
                                caption = self._find_table_caption(page_idx, block)
                                
                                self.tables.append({
                                    "page": page_idx + 1,
                                    "data": table_data,
                                    "caption": caption,
                                    "columns": len(col_positions),
                                    "rows": len(table_data)
                                })
        
        logger.info(f"Extracted {len(self.figures)} figures and {len(self.tables)} tables")
    
    def _find_figure_caption(self, page_idx, img_info):
        """Attempt to find a caption for a figure"""
        # Look for text blocks near the image that start with "Figure", "Fig.", etc.
        caption_patterns = [
            re.compile(r'^fig(ure)?\.?\s*\d+', re.IGNORECASE),
            re.compile(r'^image\s*\d+', re.IGNORECASE),
            re.compile(r'^chart\s*\d+', re.IGNORECASE),
            re.compile(r'^illustration\s*\d+', re.IGNORECASE)
        ]
        
        for block in self.text_blocks:
            if block["page"] == page_idx + 1:  # Same page
                for pattern in caption_patterns:
                    if pattern.match(block["text"]):
                        return block["text"]
        
        return ""
    
    def _find_table_caption(self, page_idx, block):
        """Attempt to find a caption for a table"""
        # Look for text blocks near the table that start with "Table", etc.
        caption_patterns = [
            re.compile(r'^table\s*\d+', re.IGNORECASE),
            re.compile(r'^tbl\.?\s*\d+', re.IGNORECASE)
        ]
        
        for text_block in self.text_blocks:
            if text_block["page"] == page_idx + 1:  # Same page
                for pattern in caption_patterns:
                    if pattern.match(text_block["text"]):
                        return text_block["text"]
        
        return ""
    
    def _analyze_potential_table_structure(self, lines):
        """Analyze if a block of lines has table-like structure"""
        if not lines:
            return []
            
        # Check if lines have consistent horizontal spacing
        x_positions = []
        
        for line in lines:
            line_positions = []
            for span in line.get("spans", []):
                line_positions.append(span["bbox"][0])
            x_positions.append(line_positions)
        
        # Check if there are consistent column positions
        if not x_positions:
            return []
            
        # Flatten and find unique positions with some tolerance
        all_positions = []
        tolerance = 5  # pixels
        
        for positions in x_positions:
            for pos in positions:
                # Check if this position is close to an existing one
                found = False
                for i, existing_pos in enumerate(all_positions):
                    if abs(pos - existing_pos) <= tolerance:
                        # Update to average position
                        all_positions[i] = (existing_pos + pos) / 2
                        found = True
                        break
                
                if not found:
                    all_positions.append(pos)
        
        # Sort positions
        all_positions.sort()
        
        # Filter out positions that aren't used consistently
        position_counts = defaultdict(int)
        for positions in x_positions:
            for pos in positions:
                for i, column_pos in enumerate(all_positions):
                    if abs(pos - column_pos) <= tolerance:
                        position_counts[i] += 1
                        break
        
        # Keep positions that appear in at least 1/3 of the lines
        threshold = len(lines) / 3
        consistent_positions = [all_positions[i] for i in position_counts if position_counts[i] >= threshold]
        
        return consistent_positions
    
    def _extract_table_data(self, lines, col_positions):
        """Extract table data from lines using column positions"""
        table_data = []
        tolerance = 5  # pixels
        
        for line in lines:
            row_data = [""] * len(col_positions)
            
            for span in line.get("spans", []):
                # Find which column this span belongs to
                for i, pos in enumerate(col_positions):
                    if abs(span["bbox"][0] - pos) <= tolerance:
                        row_data[i] += span["text"] + " "
                        break
            
            # Clean up the row data
            row_data = [cell.strip() for cell in row_data]
            
            # Add non-empty rows
            if any(cell for cell in row_data):
                table_data.append(row_data)
        
        return table_data
    
    def build_structure(self):
        """Build a complete document structure"""
        logger.info("Building document structure...")
        
        # Basic document info
        self.structure = {
            "filename": os.path.basename(self.pdf_path),
            "pages": len(self.doc),
            "chapters": [],
            "figures": self.figures,
            "tables": self.tables
        }
 # Process chapters with their content
        for i, chapter in enumerate(self.chapters):
            chapter_start_idx = chapter["index"]
            chapter_end_idx = self.chapters[i+1]["index"] if i < len(self.chapters)-1 else len(self.text_blocks)
            
            # Extract chapter text blocks
            chapter_blocks = self.text_blocks[chapter_start_idx:chapter_end_idx]
            
            # Build sections list
            sections = []
            for section in chapter.get("sections", []):
                sections.append({
                    "title": section["title"],
                    "page": section["page"],
                    "level": section["level"]
                })
            
            # Add to structure
            self.structure["chapters"].append({
                "number": chapter["number"],
                "title": chapter["title"],
                "start_page": chapter["page"],
                "sections": sections,
                "block_count": len(chapter_blocks)
            })
    
    def save_structure(self):
        """Save the document structure to a JSON file"""
        output_file = os.path.join(self.output_dir, "document_structure.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.structure, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Document structure saved to {output_file}")
    
    def extract_chapters_text(self):
        """Extract and save text content for each chapter"""
        logger.info("Extracting chapter content...")
        
        for i, chapter in enumerate(self.chapters):
            chapter_start_idx = chapter["index"]
            chapter_end_idx = self.chapters[i+1]["index"] if i < len(self.chapters)-1 else len(self.text_blocks)
            
            # Extract chapter text blocks
            chapter_blocks = self.text_blocks[chapter_start_idx:chapter_end_idx]
            
            # Convert to plain text
            chapter_text = []
            current_page = None
            
            for block in chapter_blocks:
                # Add page markers
                if current_page != block["page"]:
                    current_page = block["page"]
                    chapter_text.append(f"\n--- Page {current_page} ---\n")
                
                chapter_text.append(block["text"])
            
            # Save chapter text
            chapter_filename = f"chapter_{chapter['number']}_{chapter['title'].lower().replace(' ', '_')[:50]}.txt"
            chapter_path = os.path.join(self.output_dir, "chapters", chapter_filename)
            
            with open(chapter_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(chapter_text))
            
            # Create JSON version with structure
            chapter_json_filename = f"chapter_{chapter['number']}_{chapter['title'].lower().replace(' ', '_')[:50]}.json"
            chapter_json_path = os.path.join(self.output_dir, "chapters", chapter_json_filename)
            
            with open(chapter_json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "number": chapter["number"],
                    "title": chapter["title"],
                    "start_page": chapter["page"],
                    "sections": chapter.get("sections", []),
                    "text_blocks": chapter_blocks,
                    "text": "\n".join(chapter_text)
                }, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved chapter {chapter['number']} to {chapter_path}")

def process_pdf(pdf_path, output_dir=None):
    """Process a single PDF file"""
    analyzer = DocumentStructureAnalyzer(pdf_path, output_dir)
    return analyzer.analyze()

def process_directory(input_dir, output_dir=None):
    """Process all PDF files in a directory"""
    if not output_dir:
        output_dir = os.path.join(input_dir, "extracted")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all PDF files
    pdf_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    # Process each file
    results = []
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        # Create output subdirectory based on filename
        file_output_dir = os.path.join(output_dir, os.path.splitext(os.path.basename(pdf_file))[0])
        
        try:
            result = process_pdf(pdf_file, file_output_dir)
            if result:
                results.append({
                    "file": pdf_file,
                    "success": True,
                    "chapters": len(result.get("chapters", [])),
                    "figures": len(result.get("figures", [])),
                    "tables": len(result.get("tables", [])),
                    "output_dir": file_output_dir
                })
            else:
                results.append({
                    "file": pdf_file,
                    "success": False,
                    "error": "Analysis failed"
                })
        except Exception as e:
            logger.error(f"Error processing {pdf_file}: {e}")
            results.append({
                "file": pdf_file,
                "success": False,
                "error": str(e)
            })
    
    # Save summary report
    summary_file = os.path.join(output_dir, "processing_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Create CSV report
    df = pd.DataFrame(results)
    csv_file = os.path.join(output_dir, "processing_summary.csv")
    df.to_csv(csv_file, index=False)
    
    logger.info(f"Processed {len(results)} PDF files")
    logger.info(f"Summary saved to {summary_file} and {csv_file}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract structure from PDF documents")
    parser.add_argument("--input", required=True, help="Input PDF file or directory")
    parser.add_argument("--output", help="Output directory (defaults to 'extracted' subdirectory)")
    parser.add_argument("--batch", action="store_true", help="Process all PDFs in the input directory")
    
    args = parser.parse_args()
    
    if args.batch:
        process_directory(args.input, args.output)
    else:
        process_pdf(args.input, args.output)
