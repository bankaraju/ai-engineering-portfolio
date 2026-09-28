"""Turn a section heading into a natural-language question (rule-based)."""
import re

def generate_question_from_heading(heading_text):
    """Convert a heading into a natural question"""
    # Clean the heading text
    clean_heading = heading_text.strip().rstrip(':').strip()
    
    # Convert to title case if it's all caps
    if clean_heading.isupper():
        clean_heading = clean_heading.title()
    
    # Different patterns for different heading types
    if clean_heading.startswith("All "):
        # "All Items Need Review" -> "What items need review?"
        return f"What {clean_heading[4:].lower()}?"
    
    elif "vs." in clean_heading.lower() or "versus" in clean_heading.lower():
        # "Cost vs. Benefit" -> "What is the relationship between cost and benefit?"
        parts = re.split(r' vs\.| versus ', clean_heading, flags=re.IGNORECASE)
        return f"What is the relationship between {parts[0].lower()} and {parts[1].lower()}?"
    
    elif re.match(r'^[A-Z][a-z]+ing', clean_heading):
        # "Reading Tables" -> "How do you read tables?"
        verb = re.match(r'^([A-Z][a-z]+)ing', clean_heading).group(1)
        rest = clean_heading[len(verb) + 3:].lower()
        return f"How do you {verb.lower()} {rest}?"
    
    elif "control" in clean_heading.lower():
        # "Quality Control in Practice" -> "How was quality controlled in practice?"
        return f"How was {clean_heading.lower().replace('control', 'controlled')}?"
    
    else:
        # Generic fallback
        return f"What should I know about {clean_heading.lower()}?"
