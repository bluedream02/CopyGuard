"""
CopyGuard: Complete Defense Framework

This module integrates all CopyGuard components following the paper's workflow:
1. Copyright Notice Identifier - Detects copyright notices using OCR
2. Copyright Status Verifier - Uses Serper API + DeepSeek-R1-all to verify status
3. Query Risk Analyzer - Analyzes query for infringement risks
4. Copyright Status Reminder - Provides notifications and guidance

Workflow (as per paper):
1. Use PaddleOCR to extract text from image
2. Check for copyright notices in extracted text
3. If no notice found, use Serper API to search for source
4. Use DeepSeek-R1-all to verify copyright status
5. Analyze query for infringement risks
6. Generate reminder if copyright detected and query is risky
"""

import os
from typing import Dict, Optional, List
from .notice_identifier import identify_copyright_notice
from .copyright_verifier import search_book, is_public_domain
from .copyright_status_verifier import verify_copyright_status_complete
from .query_risk_analyzer import analyze_query_risk, should_block_query
from .status_reminder import generate_copyright_reminder, format_reminder_for_lvlm


class CopyGuard:
    """
    Complete CopyGuard defense framework.
    """
    
    def __init__(self, 
                 enable_ocr: bool = True,
                 enable_verifier: bool = True,
                 enable_risk_analyzer: bool = True,
                 enable_reminder: bool = True):
        """
        Initialize CopyGuard framework.
        
        Args:
            enable_ocr: Enable OCR-based copyright notice detection
            enable_verifier: Enable copyright status verification
            enable_risk_analyzer: Enable query risk analysis
            enable_reminder: Enable copyright status reminders
        """
        self.enable_ocr = enable_ocr
        self.enable_verifier = enable_verifier
        self.enable_risk_analyzer = enable_risk_analyzer
        self.enable_reminder = enable_reminder
        
        self.ocr_engine = None
        if enable_ocr:
            try:
                from paddleocr import PaddleOCR
                self.ocr_engine = PaddleOCR(use_angle_cls=True, lang='en')
            except ImportError:
                print("Warning: PaddleOCR not available. OCR features disabled.")
                self.enable_ocr = False
    
    def analyze_content(self,
                       image_path: Optional[str] = None,
                       text: Optional[str] = None,
                       content_type: Optional[str] = None) -> Dict:
        """
        Analyze content for copyright status.
        
        Args:
            image_path: Optional path to image file
            text: Optional text content
            content_type: Type of content ('book', 'code', 'lyrics', 'news')
            
        Returns:
            Dictionary with analysis results:
            {
                'has_notice': bool,
                'copyright_status': str,
                'is_protected': bool,
                'details': Dict
            }
        """
        result = {
            'has_notice': False,
            'copyright_status': 'unknown',
            'is_protected': False,
            'details': {}
        }
        
        # Step 1: Copyright Notice Identifier
        # Always check text for copyright notices (even if OCR is disabled)
        if text:
            notice_result = identify_copyright_notice(
                text=text,
                ocr_engine=None  # Text-based detection doesn't need OCR
            )
            result['has_notice'] = notice_result.get('has_notice', False)
            result['details']['notice_detection'] = notice_result
        
        # Also check image if OCR is enabled
        if self.enable_ocr and image_path:
            image_notice_result = identify_copyright_notice(
                image_path=image_path,
                ocr_engine=self.ocr_engine
            )
            # Combine results (if either detects notice, mark as having notice)
            if image_notice_result.get('has_notice', False):
                result['has_notice'] = True
                result['details']['image_notice_detection'] = image_notice_result
        
        # Step 2: Copyright Status Verifier (if no explicit notice)
        # As described in the paper: use Serper API to identify source,
        # then use DeepSeek-R1-all to verify copyright status
        if self.enable_verifier and not result['has_notice']:
            # Get OCR text if available from image detection
            ocr_text = None
            image_notice_detection = result.get('details', {}).get('image_notice_detection', {})
            if image_notice_detection and image_notice_detection.get('detected_text'):
                ocr_text = image_notice_detection['detected_text']
            
            # Use complete verification process (Serper + DeepSeek-R1)
            try:
                verification_result = verify_copyright_status_complete(
                    ocr_text=ocr_text,
                    text=text,
                    serper_api_key=os.getenv("SERPER_API_KEY"),
                    deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
                    deepseek_api_base=os.getenv("DEEPSEEK_API_BASE")
                )
                
                result['copyright_status'] = verification_result['copyright_status']
                result['is_protected'] = verification_result['is_protected']
                result['details']['verification'] = verification_result
                
                # Fallback: if verification unavailable, assume protected
                if result['copyright_status'] == 'unknown' and not verification_result.get('source'):
                    result['is_protected'] = True
            except Exception as e:
                print(f"Warning: Copyright verification failed: {e}")
                # Default to protected if verification fails
                result['copyright_status'] = 'unknown'
                result['is_protected'] = True
                result['details']['verification_error'] = str(e)
        
        # If notice found, assume protected
        if result['has_notice']:
            result['copyright_status'] = 'protected'
            result['is_protected'] = True
        
        return result
    
    def analyze_query(self, query: str, has_copyright: bool = False) -> Dict:
        """
        Analyze query for copyright infringement risk.
        
        Args:
            query: User query
            has_copyright: Whether content is copyrighted
            
        Returns:
            Dictionary with risk analysis and blocking decision
        """
        if not self.enable_risk_analyzer:
            return {
                'should_block': False,
                'risk_level': 'unknown',
                'suggestions': []
            }
        
        return should_block_query(query, has_copyright)
    
    def generate_reminder(self,
                         has_notice: bool,
                         copyright_status: str,
                         risk_level: str,
                         content_type: Optional[str] = None) -> str:
        """
        Generate copyright status reminder.
        
        Args:
            has_notice: Whether copyright notice was detected
            copyright_status: Copyright status
            risk_level: Risk level of query
            content_type: Type of content
            
        Returns:
            Reminder message
        """
        if not self.enable_reminder:
            return ""
        
        return generate_copyright_reminder(
            has_notice=has_notice,
            copyright_status=copyright_status,
            risk_level=risk_level,
            content_type=content_type
        )
    
    def process_request(self,
                       query: str,
                       image_path: Optional[str] = None,
                       text: Optional[str] = None,
                       content_type: Optional[str] = None) -> Dict:
        """
        Complete processing of a request through CopyGuard.
        
        Args:
            query: User query
            image_path: Optional path to image
            text: Optional text content
            content_type: Type of content
            
        Returns:
            Complete analysis result:
            {
                'content_analysis': Dict,
                'query_analysis': Dict,
                'reminder': str,
                'should_block': bool,
                'suggested_query': Optional[str]
            }
        """
        # Analyze content
        content_analysis = self.analyze_content(
            image_path=image_path,
            text=text,
            content_type=content_type
        )
        
        # Analyze query
        query_analysis = self.analyze_query(
            query=query,
            has_copyright=content_analysis['is_protected']
        )
        
        # Generate reminder
        reminder = self.generate_reminder(
            has_notice=content_analysis['has_notice'],
            copyright_status=content_analysis['copyright_status'],
            risk_level=query_analysis['risk_analysis']['risk_level'],
            content_type=content_type
        )
        
        # Get suggested alternative query
        suggested_query = None
        if query_analysis['risk_analysis']['suggestions']:
            suggested_query = query_analysis['risk_analysis']['suggestions'][0]
        
        return {
            'content_analysis': content_analysis,
            'query_analysis': query_analysis,
            'reminder': reminder,
            'should_block': query_analysis['should_block'],
            'suggested_query': suggested_query,
            'formatted_reminder': format_reminder_for_lvlm(reminder, query)
        }


def create_copyguard(enable_ocr: bool = True,
                    enable_verifier: bool = True,
                    enable_risk_analyzer: bool = True,
                    enable_reminder: bool = True) -> CopyGuard:
    """
    Factory function to create a CopyGuard instance.
    
    Args:
        enable_ocr: Enable OCR features
        enable_verifier: Enable copyright verification
        enable_risk_analyzer: Enable risk analysis
        enable_reminder: Enable reminders
        
    Returns:
        CopyGuard instance
    """
    return CopyGuard(
        enable_ocr=enable_ocr,
        enable_verifier=enable_verifier,
        enable_risk_analyzer=enable_risk_analyzer,
        enable_reminder=enable_reminder
    )

