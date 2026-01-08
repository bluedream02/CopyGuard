"""
Copyright Status Verifier (Complete Implementation)

This module implements the complete copyright status verification as described in the paper:
- Uses Serper API to search for text sources
- Uses DeepSeek-R1-all to verify copyright status
"""

import os
import requests
from typing import Dict, Optional, List
import json


def search_with_serper(query: str, api_key: Optional[str] = None) -> Dict:
    """
    Search for text sources using Serper Google Search API.
    
    This is used to identify the source of text content (e.g., book title)
    from OCR-extracted text.
    
    Args:
        query: Search query (e.g., OCR-extracted text snippet)
        api_key: Serper API key (can use SERPER_API_KEY env var)
        
    Returns:
        Dictionary containing search results
    """
    if api_key is None:
        api_key = os.getenv("SERPER_API_KEY")
    
    if not api_key:
        print("Warning: SERPER_API_KEY not provided. Serper search disabled.")
        return {}
    
    url = "https://google.serper.dev/search"
    
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }
    
    payload = {
        "q": query,
        "num": 10  # Number of results
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Serper API request failed with status {response.status_code}")
            return {}
    except Exception as e:
        print(f"Error calling Serper API: {e}")
        return {}


def extract_source_from_search_results(search_results: Dict) -> Optional[str]:
    """
    Extract potential source (e.g., book title) from search results.
    
    Args:
        search_results: Results from Serper API
        
    Returns:
        Extracted source title or None
    """
    if not search_results:
        return None
    
    # Try to extract from organic results
    organic = search_results.get("organic", [])
    if organic:
        # Get the first result's title
        first_result = organic[0]
        return first_result.get("title", None)
    
    return None


def verify_copyright_status_with_deepseek(text_snippet: str,
                                         source_info: Optional[str] = None,
                                         api_key: Optional[str] = None,
                                         api_base: Optional[str] = None) -> Dict:
    """
    Verify copyright status using DeepSeek-R1-all.
    
    This uses DeepSeek-R1-all (a search engine-enhanced model) to check
    whether content remains under copyright protection.
    
    Args:
        text_snippet: Text snippet to verify
        source_info: Optional source information (e.g., book title)
        api_key: DeepSeek API key
        api_base: DeepSeek API base URL
        
    Returns:
        Dictionary with copyright status:
        {
            'is_protected': bool,
            'copyright_status': str,
            'confidence': float,
            'details': str
        }
    """
    if api_key is None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
    if api_base is None:
        api_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
    
    if not api_key:
        print("Warning: DEEPSEEK_API_KEY not provided. Using fallback verification.")
        # Fallback: assume protected if we can't verify
        return {
            'is_protected': True,
            'copyright_status': 'unknown',
            'confidence': 0.5,
            'details': 'Cannot verify without API key'
        }
    
    # Construct prompt for DeepSeek-R1
    prompt = f"""Please analyze the following text and determine its copyright status.

Text snippet: "{text_snippet[:500]}"

{f"Source information: {source_info}" if source_info else ""}

Please determine:
1. Is this content likely protected by copyright?
2. What is the copyright status (protected/public domain/unknown)?
3. Provide a brief explanation.

Respond in JSON format:
{{
    "is_protected": true/false,
    "copyright_status": "protected" or "public_domain" or "unknown",
    "explanation": "brief explanation"
}}"""

    try:
        from litellm import completion
        
        # Set API key and base
        if api_key:
            os.environ["DEEPSEEK_API_KEY"] = api_key
        if api_base:
            os.environ["DEEPSEEK_API_BASE"] = api_base
        
        response = completion(
            model="deepseek-reasoner",  # DeepSeek-R1 model
            messages=[
                {"role": "system", "content": "You are a copyright status verification assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.3
        )
        
        result_text = response['choices'][0]['message']['content']
        
        # Try to parse JSON response
        try:
            import re
            json_match = re.search(r'\{[^}]+\}', result_text, re.DOTALL)
            if json_match:
                result_json = json.loads(json_match.group())
                return {
                    'is_protected': result_json.get('is_protected', True),
                    'copyright_status': result_json.get('copyright_status', 'unknown'),
                    'confidence': 0.8,
                    'details': result_json.get('explanation', '')
                }
        except:
            pass
        
        # Fallback: parse text response
        is_protected = 'protected' in result_text.lower() or 'copyright' in result_text.lower()
        return {
            'is_protected': is_protected,
            'copyright_status': 'protected' if is_protected else 'unknown',
            'confidence': 0.7,
            'details': result_text[:200]
        }
        
    except Exception as e:
        print(f"Error calling DeepSeek API: {e}")
        return {
            'is_protected': True,  # Default to protected if verification fails
            'copyright_status': 'unknown',
            'confidence': 0.5,
            'details': f'Verification error: {str(e)}'
        }


def verify_copyright_status_complete(ocr_text: Optional[str] = None,
                                    text: Optional[str] = None,
                                    serper_api_key: Optional[str] = None,
                                    deepseek_api_key: Optional[str] = None,
                                    deepseek_api_base: Optional[str] = None) -> Dict:
    """
    Complete copyright status verification as described in the paper.
    
    Process:
    1. If OCR text available, use Serper API to search for source
    2. Use DeepSeek-R1-all to verify copyright status
    
    Args:
        ocr_text: OCR-extracted text from image
        text: Text content (fallback if no OCR text)
        serper_api_key: Serper API key
        deepseek_api_key: DeepSeek API key
        deepseek_api_base: DeepSeek API base URL
        
    Returns:
        Dictionary with verification results
    """
    search_text = ocr_text or text
    
    if not search_text:
        return {
            'is_protected': True,  # Default to protected
            'copyright_status': 'unknown',
            'source': None,
            'verification_details': {}
        }
    
    # Step 1: Search for source using Serper API
    source_info = None
    if serper_api_key or os.getenv("SERPER_API_KEY"):
        search_results = search_with_serper(search_text[:200], serper_api_key)  # Use first 200 chars
        source_info = extract_source_from_search_results(search_results)
    
    # Step 2: Verify copyright status using DeepSeek-R1-all
    verification_result = verify_copyright_status_with_deepseek(
        text_snippet=search_text[:500],  # Use first 500 chars
        source_info=source_info,
        api_key=deepseek_api_key,
        api_base=deepseek_api_base
    )
    
    return {
        'is_protected': verification_result['is_protected'],
        'copyright_status': verification_result['copyright_status'],
        'source': source_info,
        'confidence': verification_result.get('confidence', 0.5),
        'verification_details': verification_result
    }

