"""
Copyright Notice Identifier

This module detects copyright notices in images using OCR.
"""

import re
from typing import Optional, List, Dict
from PIL import Image

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    print("Warning: PaddleOCR not installed. Install with: pip install paddleocr")


# Common copyright notice patterns
COPYRIGHT_PATTERNS = [
    r'\bcopyright\b',
    r'©',
    r'\(c\)',
    r'\ball\s+rights\s+reserved\b',
    r'\ball\s+rights\b',
    r'\bcopyrighted\b',
    r'\bcopyright\s+notice\b',
    r'\bproprietary\b',
]


def detect_copyright_notice_in_text(text: str) -> bool:
    """
    Detect copyright notice in text using pattern matching.
    
    Args:
        text: Text to analyze
        
    Returns:
        True if copyright notice is detected, False otherwise
    """
    text_lower = text.lower()
    
    # Check for copyright symbol or explicit copyright text
    # Must have at least one copyright indicator
    has_copyright_indicator = False
    has_rights_statement = False
    
    for pattern in COPYRIGHT_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            if 'rights' in pattern or 'reserved' in pattern:
                has_rights_statement = True
            else:
                has_copyright_indicator = True
    
    # Consider it a copyright notice if we find copyright indicator
    # or rights statement (either is sufficient)
    return has_copyright_indicator or has_rights_statement


def identify_copyright_notice_from_image(image_path: str, 
                                        ocr_engine: Optional[object] = None) -> Dict:
    """
    Identify copyright notices in an image using OCR.
    
    Args:
        image_path: Path to the image file
        ocr_engine: Optional OCR engine (PaddleOCR instance)
        
    Returns:
        Dictionary with detection results:
        {
            'has_notice': bool,
            'detected_text': str,
            'notice_found': bool
        }
    """
    result = {
        'has_notice': False,
        'detected_text': '',
        'notice_found': False
    }
    
    if not PADDLEOCR_AVAILABLE and ocr_engine is None:
        print("Warning: OCR not available. Install PaddleOCR for image text detection.")
        return result
    
    try:
        # Initialize OCR if not provided
        if ocr_engine is None:
            ocr_engine = PaddleOCR(use_angle_cls=True, lang='en')
        
        # Perform OCR
        ocr_result = ocr_engine.ocr(image_path, cls=True)
        
        # Extract text from OCR results
        detected_text = ""
        if ocr_result and ocr_result[0]:
            for line in ocr_result[0]:
                if line and len(line) >= 2:
                    detected_text += line[1][0] + " "
        
        result['detected_text'] = detected_text.strip()
        
        # Check for copyright notice patterns
        result['notice_found'] = detect_copyright_notice_in_text(detected_text)
        result['has_notice'] = result['notice_found']
        
    except Exception as e:
        print(f"Error during OCR: {e}")
        result['has_notice'] = False
    
    return result


def identify_copyright_notice_from_text(text: str) -> Dict:
    """
    Identify copyright notices in text content.
    
    Args:
        text: Text content to analyze
        
    Returns:
        Dictionary with detection results:
        {
            'has_notice': bool,
            'notice_found': bool
        }
    """
    notice_found = detect_copyright_notice_in_text(text)
    
    return {
        'has_notice': notice_found,
        'notice_found': notice_found
    }


def identify_copyright_notice(image_path: Optional[str] = None,
                               text: Optional[str] = None,
                               ocr_engine: Optional[object] = None) -> Dict:
    """
    Identify copyright notices from image or text.
    
    Args:
        image_path: Optional path to image file
        text: Optional text content
        ocr_engine: Optional OCR engine
        
    Returns:
        Dictionary with detection results
    """
    if image_path:
        return identify_copyright_notice_from_image(image_path, ocr_engine)
    elif text:
        return identify_copyright_notice_from_text(text)
    else:
        return {
            'has_notice': False,
            'notice_found': False,
            'error': 'No image or text provided'
        }

