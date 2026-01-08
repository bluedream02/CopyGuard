"""
Copyright Status Reminder

This module provides copyright status notifications and reminders
to guide LVLMs and users toward compliant actions.
"""

from typing import Dict, Optional, List


def generate_copyright_reminder(has_notice: bool,
                                copyright_status: Optional[str] = None,
                                risk_level: Optional[str] = None,
                                content_type: Optional[str] = None) -> str:
    """
    Generate a copyright status reminder message.
    
    Args:
        has_notice: Whether copyright notice was detected
        copyright_status: Copyright status ('protected', 'public_domain', 'unknown')
        risk_level: Risk level of the query ('high', 'medium', 'low', 'none')
        content_type: Type of content ('book', 'code', 'lyrics', 'news')
        
    Returns:
        Copyright reminder message
    """
    reminder_parts = []
    
    # Base reminder
    if has_notice:
        reminder_parts.append("⚠️ Copyright Notice Detected: This content is protected by copyright.")
    elif copyright_status == 'protected':
        reminder_parts.append("⚠️ Copyright Status: This content appears to be protected by copyright.")
    elif copyright_status == 'public_domain':
        reminder_parts.append("ℹ️ Copyright Status: This content appears to be in the public domain.")
    
    # Add content type information
    if content_type:
        reminder_parts.append(f"Content Type: {content_type}")
    
    # Add risk-based guidance
    if risk_level == 'high':
        reminder_parts.append(
            "🚫 High Risk: The requested action may infringe copyright. "
            "Please avoid repeating, extracting, paraphrasing, or translating copyrighted content."
        )
    elif risk_level == 'medium':
        reminder_parts.append(
            "⚠️ Medium Risk: Be cautious when handling this content. "
            "Consider summarizing key points or providing analysis instead."
        )
    
    # General guidance
    reminder_parts.append(
        "💡 Suggestion: You can discuss themes, provide analysis, or ask about "
        "authors/publication information without reproducing the content."
    )
    
    return "\n".join(reminder_parts)


def create_system_prompt_with_reminder(base_prompt: str,
                                       reminder: str) -> str:
    """
    Create a system prompt that includes copyright reminder.
    
    Args:
        base_prompt: Base system prompt
        reminder: Copyright reminder message
        
    Returns:
        Combined system prompt with reminder
    """
    return f"{base_prompt}\n\n{reminder}"


def format_reminder_for_lvlm(reminder: str, 
                             query: Optional[str] = None) -> str:
    """
    Format reminder message for LVLM input.
    
    Args:
        reminder: Reminder message
        query: Optional user query
        
    Returns:
        Formatted message for LVLM
    """
    formatted = f"Copyright Compliance Notice:\n{reminder}\n"
    
    if query:
        formatted += f"\nUser Query: {query}\n"
        formatted += "Please respond in a way that respects copyright while being helpful."
    
    return formatted


def get_reminder_template(reminder_type: str = "standard") -> str:
    """
    Get a reminder template by type.
    
    Args:
        reminder_type: Type of reminder ('standard', 'strict', 'lenient')
        
    Returns:
        Reminder template string
    """
    templates = {
        "standard": (
            "This content may be protected by copyright. "
            "Please do not reproduce, extract, paraphrase, or translate the content. "
            "Instead, you can provide analysis, discuss themes, or answer questions about the content."
        ),
        "strict": (
            "⚠️ COPYRIGHT NOTICE: This content is protected by copyright. "
            "You must NOT repeat, extract, paraphrase, or translate any part of this content. "
            "You may only provide general information, analysis, or commentary."
        ),
        "lenient": (
            "Note: This content may be copyrighted. "
            "Please be mindful of copyright when responding. "
            "Consider providing summaries or analysis rather than reproducing content."
        )
    }
    
    return templates.get(reminder_type, templates["standard"])

