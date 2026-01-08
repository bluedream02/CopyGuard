"""
Query Risk Analyzer

This module analyzes user queries for potential copyright infringement risks
and suggests alternative queries to avoid infringement.
"""

from typing import Dict, List, Optional
import re


# High-risk query patterns that may lead to copyright infringement
HIGH_RISK_PATTERNS = [
    r'repeat',
    r'reproduce',
    r'copy',
    r'duplicate',
    r'verbatim',
    r'exact',
    r'word for word',
    r'extract',
    r'pull out',
    r'quote',
    r'translate',
    r'render.*in.*language',
    r'paraphrase',
    r'rephrase',
    r'reword',
    r'summarize.*entire',
    r'full.*text',
    r'complete.*content',
]

# Medium-risk query patterns
MEDIUM_RISK_PATTERNS = [
    r'summarize',
    r'summary',
    r'brief',
    r'overview',
    r'explain',
    r'describe',
]

# Low-risk query patterns (generally safe)
LOW_RISK_PATTERNS = [
    r'author',
    r'who wrote',
    r'when.*published',
    r'genre',
    r'theme',
    r'comment',
    r'opinion',
    r'analysis',
    r'critique',
]


def analyze_query_risk(query: str) -> Dict:
    """
    Analyze a query for potential copyright infringement risk.
    
    Args:
        query: User query to analyze
        
    Returns:
        Dictionary with risk analysis:
        {
            'risk_level': 'high' | 'medium' | 'low' | 'none',
            'risk_score': float (0.0 to 1.0),
            'matched_patterns': List[str],
            'suggestions': List[str]
        }
    """
    query_lower = query.lower()
    matched_patterns = []
    risk_score = 0.0
    
    # Check high-risk patterns
    high_risk_matches = []
    for pattern in HIGH_RISK_PATTERNS:
        if re.search(pattern, query_lower):
            high_risk_matches.append(pattern)
            matched_patterns.append(f"high-risk: {pattern}")
            risk_score += 0.5
    
    # Check medium-risk patterns
    medium_risk_matches = []
    for pattern in MEDIUM_RISK_PATTERNS:
        if re.search(pattern, query_lower):
            medium_risk_matches.append(pattern)
            matched_patterns.append(f"medium-risk: {pattern}")
            risk_score += 0.2
    
    # Check low-risk patterns (reduce risk score)
    low_risk_matches = []
    for pattern in LOW_RISK_PATTERNS:
        if re.search(pattern, query_lower):
            low_risk_matches.append(pattern)
            risk_score = max(0.0, risk_score - 0.1)
    
    # Normalize risk score
    risk_score = min(1.0, risk_score)
    
    # Determine risk level
    if risk_score >= 0.5:
        risk_level = 'high'
    elif risk_score >= 0.2:
        risk_level = 'medium'
    elif risk_score > 0.0:
        risk_level = 'low'
    else:
        risk_level = 'none'
    
    # Generate suggestions
    suggestions = generate_alternative_queries(query, risk_level, high_risk_matches)
    
    return {
        'risk_level': risk_level,
        'risk_score': risk_score,
        'matched_patterns': matched_patterns,
        'suggestions': suggestions
    }


def generate_alternative_queries(original_query: str, 
                                  risk_level: str,
                                  matched_patterns: List[str]) -> List[str]:
    """
    Generate alternative queries that avoid copyright infringement.
    
    Args:
        original_query: Original user query
        risk_level: Risk level of the original query
        matched_patterns: List of matched high-risk patterns
        
    Returns:
        List of alternative query suggestions
    """
    suggestions = []
    
    if risk_level == 'high':
        # Suggest safer alternatives for high-risk queries
        if any('repeat' in p or 'copy' in p or 'duplicate' in p for p in matched_patterns):
            suggestions.append("Instead of repeating the content, could you provide a brief summary of the main themes?")
            suggestions.append("Could you discuss the key ideas without reproducing the exact text?")
        
        if any('extract' in p or 'quote' in p for p in matched_patterns):
            suggestions.append("Could you explain the main points instead of extracting specific text?")
            suggestions.append("Would you like a summary of the key concepts?")
        
        if any('translate' in p for p in matched_patterns):
            suggestions.append("Could you discuss the content in your own words instead of translating?")
            suggestions.append("Would you like an explanation of the main ideas?")
        
        if any('paraphrase' in p or 'rephrase' in p for p in matched_patterns):
            suggestions.append("Could you provide a summary or analysis instead of paraphrasing?")
            suggestions.append("Would you like to discuss the key themes and ideas?")
    
    elif risk_level == 'medium':
        suggestions.append("You could ask for a high-level summary or key points instead.")
        suggestions.append("Consider asking about themes, analysis, or commentary instead.")
    
    # Always provide general safe alternatives
    if not suggestions:
        suggestions.append("You could ask about the author, publication date, or genre.")
        suggestions.append("Consider asking for analysis, themes, or commentary.")
    
    return suggestions


def should_block_query(query: str, has_copyright: bool = False) -> Dict:
    """
    Determine if a query should be blocked based on risk analysis.
    
    Args:
        query: User query
        has_copyright: Whether the content is copyrighted
        
    Returns:
        Dictionary with blocking decision:
        {
            'should_block': bool,
            'reason': str,
            'risk_analysis': Dict
        }
    """
    risk_analysis = analyze_query_risk(query)
    
    # Block high-risk queries on copyrighted content
    if has_copyright and risk_analysis['risk_level'] == 'high':
        return {
            'should_block': True,
            'reason': 'High-risk query on copyrighted content',
            'risk_analysis': risk_analysis
        }
    
    # Allow other queries but provide warnings
    return {
        'should_block': False,
        'reason': 'Query allowed with risk assessment',
        'risk_analysis': risk_analysis
    }

