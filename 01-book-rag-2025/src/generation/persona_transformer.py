"""Rule-based persona rewrite: swap investment terms for a metaphor vocabulary
(sweet-maker or chess) and add framing text. No LLM involved."""
import re

def transform_content(content, persona='standard'):
    """Transform content based on selected persona"""
    if not content:
        return content
        
    if persona == 'halwai':
        return transform_to_halwai(content)
    elif persona == 'chess':
        return transform_to_chess(content)
    else:
        return transform_to_standard(content)

def transform_to_halwai(content):
    """Transform content using Sweet-Maker (Halwai) metaphors"""
    # Add Halwai introduction
    prefix = "🍯 Let me explain this like a traditional Indian Halwai (sweet-maker) would... \n\n"
    
    # Investment terms to sweet-making metaphors
    metaphors = {
        'risk': 'heat of the kitchen',
        'return': 'sweetness of the mithai',
        'portfolio': 'thali of sweets',
        'diversification': 'variety of sweets',
        'market': 'bazaar',
        'stocks': 'ingredients',
        'bonds': 'staple ingredients',
        'investment': 'recipe',
        'profit': 'sweetness',
        'loss': 'bitterness',
        'volatility': 'temperature fluctuations',
        'compound interest': 'fermenting process',
        'long-term': 'slow cooking',
        'short-term': 'quick frying',
        'value investing': 'traditional recipe'
    }
    
    transformed = content
    for term, metaphor in metaphors.items():
        transformed = re.sub(r'\b' + re.escape(term) + r'\b', metaphor, transformed, flags=re.IGNORECASE)
    
    # Add sweets-related emojis
    emojis = ['🍯', '🍪', '🍬', '🧁', '🍮', '🥮', '🍚', '⏲️', '🔥', '👨‍🍳']
    paragraphs = transformed.split('\n\n')
    
    for i in range(len(paragraphs)):
        if len(paragraphs[i].strip()) > 50:  # Only add to substantial paragraphs
            emoji = emojis[i % len(emojis)]
            paragraphs[i] = f"{emoji} {paragraphs[i]}"
    
    transformed = prefix + '\n\n'.join(paragraphs)
    
    # Add conclusion
    transformed += "\n\n🥄 Just as we need patience to make the perfect mithai, we need patience in investing too. Remember that rushing the process only leads to half-cooked results!"
    
    return transformed

def transform_to_chess(content):
    """Transform content using Chess metaphors"""
    # Add Chess introduction
    prefix = "♟️ Looking at this from a chess master's perspective... \n\n"
    
    # Investment terms to chess metaphors
    metaphors = {
        'risk': 'gambit',
        'return': 'advantage',
        'portfolio': 'position',
        'diversification': 'piece development',
        'market': 'board',
        'stocks': 'pieces',
        'bonds': 'pawns',
        'volatility': 'tactical complexity',
        'compound interest': 'positional advantage',
        'long-term': 'endgame',
        'short-term': 'opening moves',
        'strategy': 'opening theory',
        'analysis': 'calculation',
        'opportunity': 'tactic',
        'value investing': 'positional play'
    }
    
    transformed = content
    for term, metaphor in metaphors.items():
        transformed = re.sub(r'\b' + re.escape(term) + r'\b', metaphor, transformed, flags=re.IGNORECASE)
    
    # Add chess-related emojis
    emojis = ['♟️', '♔', '♕', '♖', '♗', '♘', '⚔️', '🛡️', '🏆', '🧠']
    paragraphs = transformed.split('\n\n')
    
    for i in range(len(paragraphs)):
        if len(paragraphs[i].strip()) > 50:
            emoji = emojis[i % len(emojis)]
            paragraphs[i] = f"{emoji} {paragraphs[i]}"
    
    transformed = prefix + '\n\n'.join(paragraphs)
    
    # Add conclusion
    transformed += "\n\n👑 As in chess, the best investors think several moves ahead, maintain strategic patience, and know when to convert a small advantage into a winning position."
    
    return transformed

def transform_to_standard(content):
    """Transform content to standard format with engaging emojis"""
    paragraphs = content.split('\n\n')
    emojis = ['📈', '💰', '🔍', '📊', '⚖️', '🧩', '🔄', '💡', '🚀', '💼']
    
    for i in range(len(paragraphs)):
        if len(paragraphs[i].strip()) > 50:
            emoji = emojis[i % len(emojis)]
            paragraphs[i] = f"{emoji} {paragraphs[i]}"
    
    # Add a closing line
    transformed = '\n\n'.join(paragraphs)
    transformed += "\n\n💭 Remember: Successful investing requires both analytical rigor and emotional discipline. Focus on long-term value creation rather than short-term price movements."
    
    return transformed
