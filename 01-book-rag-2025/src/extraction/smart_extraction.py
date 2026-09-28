#!/usr/bin/env python3
"""
Smart PDF Element Detection and Extraction
-----------------------------------------
First scans pages to identify which ones likely contain tables, figures or infographics,
then performs targeted extraction only on those pages.
"""

import os
import sys
import json
import time
import re
import argparse
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Ensure PyMuPDF is available
try:
    import fitz  # PyMuPDF
except ImportError:
    logger.error("PyMuPDF is required. Please install it: pip install PyMuPDF")
    sys.exit(1)

def is_table_likely(text, page_obj):
    """
    Check if a page likely contains a table based on various patterns
    
    Args:
        text: Extracted text from the page
        page_obj: PyMuPDF page object
        
    Returns:
        float: Likelihood score (0-1) that page contains a table
    """
    score = 0.0
    
    # Check for table indicators in text
    indicators = [
        # Column headers with separating characters
        r"\n[A-Za-z]+\s+[A-Za-z]+\s+[A-Za-z]+\s+[A-Za-z]+\s*\n",
        # Multiple numbers in sequence
        r"\n\s*[\d,\.%]+\s+[\d,\.%]+\s+[\d,\.%]+\s*\n",
        # Presence of percentage symbols
        r"\d+\s*%\s+\d+\s*%",
        # Common table header patterns
        r"(Table|TABLE)\s+\d+[\.:]",
        # Years as column headers
        r"(19|20)\d{2}\s+(19|20)\d{2}\s+(19|20)\d{2}",
        # Financial metrics on separate lines
        r"(Revenue|Sales|Profit|EBITDA|ROI|ROE|EPS|P/E|Ratio)",
        # Tabular data with units
        r"(Rs\.|USD|\$|₹|€)\s*[\d,\.]+"
    ]
    
    for pattern in indicators:
        if re.search(pattern, text):
            score += 0.2
            logger.debug(f"Found table indicator pattern: {pattern}")
    
    # Check if page has many horizontal or vertical lines
    # (often indicative of tables)
    paths = page_obj.get_drawings()
    h_lines = 0
    v_lines = 0
    
    for path in paths:
        for item in path["items"]:
            if item[0] == "l":  # Line segment: ("l", Point, Point)
                p1, p2 = item[1], item[2]
                x0, y0, x1, y1 = p1.x, p1.y, p2.x, p2.y
                if abs(y1 - y0) < 2:  # Horizontal line
                    h_lines += 1
                if abs(x1 - x0) < 2:  # Vertical line
                    v_lines += 1
    
    if h_lines > 3 and v_lines > 3:
        score += 0.3
        logger.debug(f"Found grid structure: {h_lines} horizontal, {v_lines} vertical lines")
    
    # Check text blocks for grid-like alignment
    blocks = page_obj.get_text("dict")["blocks"]
    y_positions = {}
    
    for block in blocks:
        if block["type"] == 0:  # Text block
            for line in block["lines"]:
                y = round(line["bbox"][1])
                if y in y_positions:
                    y_positions[y] += 1
                else:
                    y_positions[y] = 1
    
    # If multiple lines share the same Y positions, likely a table
    aligned_rows = sum(1 for count in y_positions.values() if count > 1)
    if aligned_rows > 3:
        score += 0.2
        logger.debug(f"Found aligned text rows: {aligned_rows}")
    
    # Cap the score at 1.0
    return min(score, 1.0)

def is_figure_likely(page_obj):
    """
    Check if a page likely contains a figure/infographic
    
    Args:
        page_obj: PyMuPDF page object
        
    Returns:
        float: Likelihood score (0-1) that page contains a figure
    """
    score = 0.0
    
    # Check for images
    image_list = page_obj.get_images(full=True)
    if len(image_list) > 0:
        score += min(0.5, len(image_list) * 0.1)
        logger.debug(f"Found {len(image_list)} images")
    
    # Check for shapes and drawings
    paths = page_obj.get_drawings()
    shapes_count = len(paths)
    
    if shapes_count > 5:
        score += min(0.3, shapes_count * 0.01)
        logger.debug(f"Found {shapes_count} drawing paths")
    
    # Check for color
    pixmap = page_obj.get_pixmap(matrix=fitz.Matrix(72/72, 72/72))
    if pixmap.n >= 3:  # Color image
        # Sample pixels to check for color diversity
        samples = []
        width, height = pixmap.width, pixmap.height
        for x in range(0, width, width//10):
            for y in range(0, height, height//10):
                idx = y * pixmap.stride + x * pixmap.n
                if pixmap.n >= 3:
                    samples.append((pixmap.samples[idx], pixmap.samples[idx+1], pixmap.samples[idx+2]))
        
        # Count unique colors
        unique_colors = len(set(samples))
        if unique_colors > 50:
            score += min(0.2, unique_colors * 0.001)
            logger.debug(f"Found diverse colors: {unique_colors} unique samples")
    
    # Cap the score at 1.0
    return min(score, 1.0)

def detect_tables_from_text(page, min_rows=3, min_cols=2):
    """
    Detect tables by analyzing text layout patterns
    
    Args:
        page: PyMuPDF page object
        min_rows: Minimum number of rows to consider as a table
        min_cols: Minimum number of columns to consider as a table
        
    Returns:
        List of detected tables with their content
    """
    # Get text blocks with their positions
    blocks = page.get_text("dict")["blocks"]
    
    # Extract lines of text with their y-positions
    lines = []
    for block in blocks:
        if block["type"] == 0:  # Text block
            for line in block["lines"]:
                line_text = ""
                for span in line["spans"]:
                    line_text += span["text"]
                if line_text.strip():
                    lines.append({
                        "text": line_text.strip(),
                        "y": line["bbox"][1],  # Top y-coordinate
                        "x": line["bbox"][0],  # Left x-coordinate
                        "bbox": line["bbox"]   # Complete bounding box
                    })
    
    # Sort lines by y-position
    lines = sorted(lines, key=lambda l: l["y"])
    
    # Find groups of lines with similar y-positions (potential table rows)
    rows = []
    current_row = []
    current_y = None
    y_threshold = 5  # Maximum pixel difference to consider lines in same row
    
    for line in lines:
        if current_y is None or abs(line["y"] - current_y) <= y_threshold:
            current_row.append(line)
            current_y = line["y"]
        else:
            if current_row:
                rows.append(sorted(current_row, key=lambda l: l["x"]))
            current_row = [line]
            current_y = line["y"]
    
    # Add the last row
    if current_row:
        rows.append(sorted(current_row, key=lambda l: l["x"]))
    
    # Identify potential tables by looking for consistent row patterns
    tables = []
    i = 0
    while i < len(rows) - min_rows:
        # Check if we have a consistent number of cells per row
        row_lens = [len(rows[i+j]) for j in range(min_rows)]
        
        # Only consider as table if all rows have at least min_cols
        if min(row_lens) >= min_cols:
            # We found a potential table
            table_start = i
            table_end = i + min_rows
            
            # Extend table if next rows have similar structure
            while table_end < len(rows) and len(rows[table_end]) >= min_cols:
                table_end += 1
            
            # Extract table content
            table_rows = []
            for row_idx in range(table_start, table_end):
                table_row = [cell["text"] for cell in rows[row_idx]]
                table_rows.append(table_row)
            
            # Get table bounds
            if rows[table_start] and rows[table_end-1]:
                first_row = rows[table_start]
                last_row = rows[table_end-1]
                if first_row and last_row:
                    x0 = min(cell["bbox"][0] for cell in first_row)
                    y0 = min(cell["bbox"][1] for cell in first_row)
                    x1 = max(cell["bbox"][2] for cell in last_row)
                    y1 = max(cell["bbox"][3] for cell in last_row)
                    
                    table_bounds = [x0, y0, x1, y1]
                else:
                    table_bounds = None
            else:
                table_bounds = None
            
            # Add table to results
            tables.append({
                "rows": table_rows,
                "row_count": len(table_rows),
                "col_count": max(len(row) for row in table_rows),
                "bounds": table_bounds
            })
            
            # Continue search after this table
            i = table_end
        else:
            i += 1
    
    return tables

def is_numeric_column(column):
    """Check if a column is mostly numeric values"""
    numeric_count = 0
    for cell in column:
        # Check if cell is a number or percentage
        if re.match(r'^-?\d+\.?\d*%?$', cell.strip()):
            numeric_count += 1
    
    return numeric_count > len(column) * 0.6  # If more than 60% are numeric

def refine_detected_tables(tables):
    """
    Further refine tables by checking for numeric columns and headers
    
    Args:
        tables: List of raw detected tables
        
    Returns:
        List of refined tables with better structure
    """
    refined_tables = []
    
    for table in tables:
        rows = table["rows"]
        if not rows:
            continue
            
        # Check if first row might be a header
        has_header = False
        if len(rows) > 1:
            first_row = rows[0]
            other_rows = rows[1:]
            
            # Check if any column is mostly numeric except in header row
            for col_idx in range(min(len(row) for row in rows)):
                column = [row[col_idx] for row in other_rows if col_idx < len(row)]
                if column and is_numeric_column(column) and not re.match(r'^-?\d+\.?\d*%?$', first_row[col_idx].strip()):
                    has_header = True
                    break
        
        # Add refined table
        refined_tables.append({
            "content": rows,
            "has_header": has_header,
            "rows": len(rows),
            "cols": max(len(row) for row in rows) if rows else 0,
            "bounds": table["bounds"]
        })
    
    return refined_tables

def detect_and_extract_elements(pdf_path, output_dir, start_page=None, end_page=None, min_score=0.5):
    """
    Detect and extract tables, figures, and infographics from PDF
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save extraction results
        start_page: Page to start extraction (1-based, optional)
        end_page: Page to end extraction (1-based, optional)
        min_score: Minimum likelihood score to trigger extraction (0-1)
    
    Returns:
        Dictionary with extraction summary
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Images directory
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    logger.info(f"Processing PDF: {pdf_path}")
    start_time = time.time()
    
    # Open the PDF
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"Error opening PDF file: {e}")
        return None
    
    total_pages = len(doc)
    logger.info(f"Total pages in document: {total_pages}")
    
    # Validate and adjust page range
    if start_page is None:
        start_page = 1
    if end_page is None:
        end_page = total_pages
    
    # Convert to 0-based indexing for PyMuPDF
    start_idx = max(0, start_page - 1)
    end_idx = min(total_pages - 1, end_page - 1)
    
    logger.info(f"Analyzing pages {start_page} to {end_page}")
    
    # First phase: Detect pages with tables and figures
    candidate_pages = []
    
    for page_idx in range(start_idx, end_idx + 1):
        page_num = page_idx + 1  # 1-based page number for reporting
        
        try:
            page = doc[page_idx]
            text = page.get_text()
            
            # Check for tables and figures
            table_score = is_table_likely(text, page)
            figure_score = is_figure_likely(page)
            
            logger.info(f"Page {page_num}: Table likelihood: {table_score:.2f}, Figure likelihood: {figure_score:.2f}")
            
            # If either score is above threshold, mark as candidate
            if table_score >= min_score or figure_score >= min_score:
                candidate_pages.append({
                    "page_idx": page_idx,
                    "page_num": page_num,
                    "table_score": table_score,
                    "figure_score": figure_score
                })
        except Exception as e:
            logger.error(f"Error analyzing page {page_num}: {e}")
    
    logger.info(f"Found {len(candidate_pages)} candidate pages for extraction")
    # Second phase: Extract tables and figures from candidate pages
    tables_data = []
    figures_data = []
    text_data = []
    
    for candidate in candidate_pages:
        page_idx = candidate["page_idx"]
        page_num = candidate["page_num"]
        
        logger.info(f"Extracting from page {page_num}")
        
        try:
            page = doc[page_idx]
            text = page.get_text()
            
            # Store text for this page
            text_data.append({
                "page": page_num,
                "text": text
            })
            
            # If likely a table, extract table structures
            if candidate["table_score"] >= min_score:
                logger.info(f"Extracting tables from page {page_num}")
                
                try:
                    # Get tables using our custom detection
                    detected_tables = detect_tables_from_text(page)
                    refined_tables = refine_detected_tables(detected_tables)
                    
                    for i, table in enumerate(refined_tables):
                        table_data = {
                            "page": page_num,
                            "index": i,
                            "rows": table["rows"],
                            "cols": table["cols"],
                            "has_header": table["has_header"],
                            "content": table["content"],
                            "bounds": table["bounds"]
                        }
                        tables_data.append(table_data)
                        logger.info(f"  Found table {i+1} with {table_data['rows']} rows × {table_data['cols']} columns")
                except Exception as e:
                    logger.error(f"Error extracting tables from page {page_num}: {e}")
                    
                    # If the imported function failed, try a simpler backup approach
                    logger.info("Trying backup table detection...")
                    
                    # Simple table detection (backup)
                    blocks = page.get_text("dict")["blocks"]
                    lines_by_y = {}
                    
                    for block in blocks:
                        if block["type"] == 0:  # Text block
                            for line in block["lines"]:
                                y = round(line["bbox"][1])
                                line_text = "".join([span["text"] for span in line["spans"]])
                                
                                if y not in lines_by_y:
                                    lines_by_y[y] = []
                                lines_by_y[y].append({
                                    "text": line_text,
                                    "x": line["bbox"][0]
                                })
                    
                    # Sort Y positions
                    y_positions = sorted(lines_by_y.keys())
                    
                    # Only consider it a table if we have multiple aligned rows
                    if len(y_positions) >= 3:
                        # Sort each row by X position
                        rows = []
                        for y in y_positions:
                            sorted_line = sorted(lines_by_y[y], key=lambda x: x["x"])
                            rows.append([line["text"] for line in sorted_line])
                        
                        # Only save if we found reasonable data
                        if len(rows) >= 3 and max(len(row) for row in rows) >= 2:
                            table_data = {
                                "page": page_num,
                                "index": 0,
                                "rows": len(rows),
                                "cols": max(len(row) for row in rows),
                                "content": rows,
                                "bounds": None,
                                "has_header": True if len(rows) > 0 else False
                            }
                            tables_data.append(table_data)
                            logger.info(f"  Found backup table with {table_data['rows']} rows × {table_data['cols']} columns")
            # If likely a figure, extract images
            if candidate["figure_score"] >= min_score:
                logger.info(f"Extracting figures from page {page_num}")
                
                # Extract images
                image_list = page.get_images(full=True)
                for img_idx, img in enumerate(image_list):
                    xref = img[0]
                    try:
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        
                        # Save image to file
                        image_filename = f"page_{page_num}_image_{img_idx+1}.png"
                        image_path = os.path.join(images_dir, image_filename)
                        with open(image_path, "wb") as img_file:
                            img_file.write(image_bytes)
                        
                        # Get image dimensions and position
                        width = base_image.get("width", 0)
                        height = base_image.get("height", 0)
                        
                        figure_data = {
                            "page": page_num,
                            "index": img_idx,
                            "filename": f"images/{image_filename}",
                            "width": width,
                            "height": height
                        }
                        figures_data.append(figure_data)
                        logger.info(f"  Found image {img_idx+1} - saved to {image_filename}")
                    except Exception as img_err:
                        logger.error(f"  Error extracting image: {img_err}")
                
                # If no images found but score was high, capture the page as a screenshot
                if len(image_list) == 0:
                    logger.info(f"No embedded images found on page {page_num}, capturing as screenshot")
                    try:
                        # Higher resolution for better quality
                        zoom = 2.0
                        matrix = fitz.Matrix(zoom, zoom)
                        pixmap = page.get_pixmap(matrix=matrix)
                        
                        # Save the pixmap as an image
                        image_filename = f"page_{page_num}_screenshot.png"
                        image_path = os.path.join(images_dir, image_filename)
                        pixmap.save(image_path)
                        
                        figure_data = {
                            "page": page_num,
                            "index": 0,
                            "filename": f"images/{image_filename}",
                            "width": pixmap.width,
                            "height": pixmap.height,
                            "type": "screenshot"
                        }
                        figures_data.append(figure_data)
                        logger.info(f"  Captured page screenshot - saved to {image_filename}")
                    except Exception as ss_err:
                        logger.error(f"  Error capturing screenshot: {ss_err}")
        
        except Exception as page_err:
            logger.error(f"Error processing page {page_num}: {page_err}")
    
    # Close the document
    doc.close()
    # Save extraction results
    tables_file = os.path.join(output_dir, "tables.json")
    with open(tables_file, "w") as f:
        json.dump(tables_data, f, indent=2)
    
    figures_file = os.path.join(output_dir, "figures.json")
    with open(figures_file, "w") as f:
        json.dump(figures_data, f, indent=2)
    
    text_file = os.path.join(output_dir, "text.json")
    with open(text_file, "w") as f:
        json.dump(text_data, f, indent=2)
    
    # Save candidate pages data
    candidates_file = os.path.join(output_dir, "candidate_pages.json")
    with open(candidates_file, "w") as f:
        json.dump(candidate_pages, f, indent=2)
    
    # Create summary file
    summary = {
        "pdf_file": os.path.basename(pdf_path),
        "total_pages": total_pages,
        "analyzed_pages": end_idx - start_idx + 1,
        "candidate_pages": len(candidate_pages),
        "tables_count": len(tables_data),
        "figures_count": len(figures_data),
        "processing_time": time.time() - start_time
    }
    
    summary_file = os.path.join(output_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nExtraction complete in {time.time() - start_time:.2f} seconds:")
    logger.info(f"  - Analyzed {end_idx - start_idx + 1} page(s)")
    logger.info(f"  - Found {len(candidate_pages)} candidate pages with potential content")
    logger.info(f"  - Extracted {len(tables_data)} tables")
    logger.info(f"  - Extracted {len(figures_data)} figures/images")
    logger.info(f"  - Results saved to {output_dir}")
    
    return summary

def main():
    """Main function for smart PDF extraction"""
    parser = argparse.ArgumentParser(description='Smart detection and extraction of tables and figures from PDF')
    parser.add_argument('input_pdf', help='Path to the input PDF file')
    parser.add_argument('--output', '-o', default="smart_extraction_output",
                      help='Directory to save extraction results')
    parser.add_argument('--start-page', '-s', type=int,
                      help='Start page for extraction (1-based, optional)')
    parser.add_argument('--end-page', '-e', type=int,
                      help='End page for extraction (1-based, optional)')
    parser.add_argument('--min-score', '-m', type=float, default=0.5,
                      help='Minimum likelihood score to trigger extraction (0-1, default: 0.5)')
    
    args = parser.parse_args()
    
    logger.info("=== Smart PDF Element Detection and Extraction ===")
    detect_and_extract_elements(
        args.input_pdf, 
        args.output, 
        args.start_page, 
        args.end_page, 
        args.min_score
    )

if __name__ == "__main__":
    main()
