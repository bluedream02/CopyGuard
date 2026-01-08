"""
CopyGuard Defense Framework

This module provides the CopyGuard defense mechanism for enhancing
copyright compliance in LVLMs.

Components:
1. Copyright Notice Identifier - Detects copyright notices using OCR
2. Copyright Status Verifier - Verifies copyright status using public records
3. Query Risk Analyzer - Analyzes queries for infringement risks
4. Copyright Status Reminder - Provides copyright status notifications
"""

from .copyright_verifier import (
    search_titles,
    search_book_gutenberg,
    search_book,
    is_public_domain
)

from .copyright_status_verifier import (
    search_with_serper,
    verify_copyright_status_with_deepseek,
    verify_copyright_status_complete
)

from .notice_identifier import (
    identify_copyright_notice,
    detect_copyright_notice_in_text
)

from .query_risk_analyzer import (
    analyze_query_risk,
    generate_alternative_queries,
    should_block_query
)

from .status_reminder import (
    generate_copyright_reminder,
    format_reminder_for_lvlm,
    get_reminder_template
)

from .copyguard import (
    CopyGuard,
    create_copyguard
)

__all__ = [
    # Copyright Verifier (Basic)
    'search_titles',
    'search_book_gutenberg',
    'search_book',
    'is_public_domain',
    # Copyright Status Verifier (Complete - as per paper)
    'search_with_serper',
    'verify_copyright_status_with_deepseek',
    'verify_copyright_status_complete',
    # Notice Identifier
    'identify_copyright_notice',
    'detect_copyright_notice_in_text',
    # Query Risk Analyzer
    'analyze_query_risk',
    'generate_alternative_queries',
    'should_block_query',
    # Status Reminder
    'generate_copyright_reminder',
    'format_reminder_for_lvlm',
    'get_reminder_template',
    # Main Framework
    'CopyGuard',
    'create_copyguard'
]



