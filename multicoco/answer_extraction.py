import logging
import re
from typing import List, Pattern, Tuple, Optional, Dict, Any
from .constants import CHOICE_MAPPINGS, VALID_CHOICE_NUMBERS
from .exceptions import AnswerExtractionError
logger = logging.getLogger(__name__)

# Standard text-based extraction patterns
EXTRACTION_PATTERNS: List[Tuple[Pattern[str], str]] = [
    (re.compile('(\\d+)\\s*:\\s*[a-zA-Z]'), 'number_colon'), 
    (re.compile('^(\\d+)(?:\\s|$)'), 'leading_number'), 
    (re.compile('(?:answer is|choice is|option is)\\s*(\\d+)', re.IGNORECASE), 'answer_format'), 
    (re.compile('(\\d+)'), 'any_digit')
]

# IMPROVEMENT: Multimodal-specific extraction patterns
MULTIMODAL_PATTERNS: List[Tuple[Pattern[str], str]] = [
    (re.compile(r'(?:image shows?|picture shows?|can see|visible|depicted)\s*(?:is\s*)?([^.!?]+)', re.IGNORECASE), 'visual_description'),
    (re.compile(r'(?:color|colour)\s*(?:is\s*)?(\w+)', re.IGNORECASE), 'color_extraction'),
    (re.compile(r'(?:object|item|thing)\s*(?:is\s*)?(?:a\s*|an\s*)?([^.!?,]+)', re.IGNORECASE), 'object_identification'),
    (re.compile(r'(?:count|number|total)\s*(?:is\s*|of\s*)?(\d+)', re.IGNORECASE), 'count_extraction'),
    (re.compile(r'(?:located|positioned|placed)\s*(?:at\s*|in\s*|on\s*)?([^.!?,]+)', re.IGNORECASE), 'location_extraction'),
]

def extract_answer_choice(generated_text: str, is_cot: bool=False, is_multimodal: bool=False, expected_type: str='choice') -> str:
    """
    IMPROVEMENT: Enhanced answer extraction with multimodal support.
    
    Args:
        generated_text: The generated response text
        is_cot: Whether this is from chain-of-thought generation
        is_multimodal: Whether this is from a multimodal model
        expected_type: Type of expected answer ('choice', 'description', 'count', 'color', etc.)
    """
    text = generated_text.strip()
    if not text:
        return ''
    
    # IMPROVEMENT: Apply multimodal extraction if needed
    if is_multimodal and expected_type != 'choice':
        result = _extract_multimodal_answer(text, expected_type)
        if result:
            return result
    
    # Standard choice extraction
    for pattern, pattern_name in EXTRACTION_PATTERNS:
        result = _extract_with_pattern(pattern, text, pattern_name)
        if result in VALID_CHOICE_NUMBERS:
            return result
    
    # Word mapping fallback
    result = _extract_word_mappings(text)
    if result in VALID_CHOICE_NUMBERS:
        return result
    
    # IMPROVEMENT: Multimodal fallback extraction
    if is_multimodal:
        multimodal_result = _extract_multimodal_fallback(text, expected_type)
        if multimodal_result:
            return multimodal_result
    
    logger.warning(f'Could not extract valid answer from: {text[:100]}')
    return text.strip()

def _extract_multimodal_answer(text: str, expected_type: str) -> str:
    """
    IMPROVEMENT: Extract answers from multimodal responses based on expected type.
    """
    if expected_type == 'description':
        return _extract_visual_description(text)
    elif expected_type == 'color':
        return _extract_color(text)
    elif expected_type == 'count':
        return _extract_count(text)
    elif expected_type == 'object':
        return _extract_object(text)
    elif expected_type == 'location':
        return _extract_location(text)
    else:
        return ''

def _extract_visual_description(text: str) -> str:
    """Extract visual descriptions from generated text."""
    for pattern, _ in MULTIMODAL_PATTERNS:
        if 'visual_description' in _:
            match = pattern.search(text)
            if match:
                return match.group(1).strip()
    
    # Fallback: extract text after "shows", "depicts", etc.
    fallback_patterns = [
        r'shows?\s+([^.!?]+)',
        r'depicts?\s+([^.!?]+)',
        r'image\s+(?:contains?|has)\s+([^.!?]+)',
    ]
    
    for pattern in fallback_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return ''

def _extract_color(text: str) -> str:
    """Extract color information from text."""
    # Common colors
    colors = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'brown', 'gray', 'grey', 
              'orange', 'purple', 'pink', 'gold', 'silver', 'dark', 'light']
    
    text_lower = text.lower()
    for color in colors:
        if color in text_lower:
            # Look for color with context
            color_pattern = rf'\b{color}\b'
            if re.search(color_pattern, text_lower):
                return color
    
    return ''

def _extract_count(text: str) -> str:
    """Extract numerical counts from text."""
    # Look for explicit numbers
    count_patterns = [
        r'(?:count|number|total|amount)\s*(?:is\s*|of\s*)?(\d+)',
        r'(\d+)\s*(?:items?|objects?|things?|pieces?)',
        r'(?:there\s+(?:are|is)\s*)?(\d+)',
    ]
    
    for pattern in count_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return ''

def _extract_object(text: str) -> str:
    """Extract object identification from text."""
    # Look for object descriptions
    object_patterns = [
        r'(?:object|item|thing)\s+(?:is\s+)?(?:a\s+|an\s+)?([^.!?,]+)',
        r'(?:see|shows?|depicts?)\s+(?:a\s+|an\s+)?([^.!?,]+)',
        r'(?:main|primary|central)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+){0,2})',
        r'(?:is\s+(?:a\s+|an\s+)?)?([a-zA-Z]+(?:\s+[a-zA-Z]+){0,2})\s*(?:$|[.!?])',
    ]
    
    for pattern in object_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            obj = match.group(1).strip()
            # Filter out common non-objects and clean up
            if obj.lower() not in ['the', 'this', 'that', 'it', 'one', 'some', 'any', 'main', 'primary', 'central']:
                # Clean up common prefixes
                obj = re.sub(r'^(main|primary|central)\s+', '', obj, flags=re.IGNORECASE)
                return obj
    
    return ''

def _extract_location(text: str) -> str:
    """Extract location information from text."""
    location_patterns = [
        r'(?:located|positioned|placed|found)\s+(?:at\s+|in\s+|on\s+|near\s+)?([^.!?,]+)',
        r'(?:at\s+the\s+|in\s+the\s+|on\s+the\s+)([^.!?,]+)',
        r'(?:left|right|top|bottom|center|middle|corner)\s*(?:of\s+)?([^.!?,]*)',
    ]
    
    for pattern in location_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return ''

def _extract_multimodal_fallback(text: str, expected_type: str) -> str:
    """
    IMPROVEMENT: Fallback extraction for multimodal content when primary methods fail.
    """
    # Clean up the text and extract the most relevant part
    sentences = re.split(r'[.!?]+', text)
    
    if expected_type == 'description':
        # Return the longest sentence as description
        return max(sentences, key=len).strip() if sentences else ''
    
    elif expected_type in ['color', 'object', 'count', 'location']:
        # Look for relevant keywords in each sentence
        relevant_keywords = {
            'color': ['color', 'colour', 'red', 'blue', 'green', 'yellow', 'black', 'white'],
            'object': ['object', 'item', 'thing', 'is', 'shows', 'depicts'],
            'count': ['count', 'number', 'total', 'many', 'few', 'several'],
            'location': ['located', 'position', 'place', 'at', 'in', 'on', 'left', 'right']
        }
        
        keywords = relevant_keywords.get(expected_type, [])
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in keywords):
                return sentence.strip()
    
    # Final fallback: return first non-empty sentence
    for sentence in sentences:
        if sentence.strip():
            return sentence.strip()
    
    return ''

def _extract_with_pattern(pattern: Pattern[str], text: str, pattern_name: str) -> str:
    if pattern_name == 'any_digit':
        return next((match for match in pattern.findall(text) if match in VALID_CHOICE_NUMBERS), '')
    match = pattern.search(text)
    return match.group(1) if match else ''

def _extract_word_mappings(text: str) -> str:
    text_lower = text.lower()
    for word, choice in CHOICE_MAPPINGS.items():
        if word in text_lower:
            return choice
    return ''