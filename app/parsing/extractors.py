"""
Field Extractors for Receipt Data.
Extracts merchant name, date, and total amount from OCR results.
"""
import re
from typing import List, Optional, Tuple
from datetime import datetime, date
import logging

from app.config import ExtractionConfig
from app.ocr.ocr_engine import OCRResult
from app.parsing.currency_parser import CurrencyParser, extract_all_amounts

logger = logging.getLogger(__name__)


class MerchantExtractor:
    """
    Extracts merchant/store name from receipt.
    
    Strategy:
    - Merchant name is typically in the first 2-3 lines
    - Filter out dates, phone numbers, addresses
    - Look for all-caps text or prominent text at top
    """
    
    # Patterns to exclude (not merchant names)
    EXCLUDE_PATTERNS = [
        r"^\d{2}[/\-]\d{2}[/\-]\d{2,4}$",  # Dates
        r"^\d{2}:\d{2}(:\d{2})?$",  # Times
        r"^\+?\d[\d\s\-]{8,}$",  # Phone numbers
        r"^(jl\.|jalan|alamat|address)\b",  # Address prefixes
        r"^(telp?|phone|hp|whatsapp)\b",  # Phone prefixes (removed 'wa' to avoid matching 'warung')
        r"^wa\s*:",  # WhatsApp with colon (wa:, wa :)
        r"^(npwp|nik)\b",  # ID numbers
        r"^no\s*:",  # Number label
        r"^[\-=_\*]{3,}$",  # Separator lines
        r"^(struk|receipt|invoice|nota)\b",  # Document type labels
    ]
    
    @classmethod
    def extract(cls, ocr_results: List[OCRResult], top_n_lines: int = 5) -> Optional[str]:
        """
        Extract merchant name from OCR results.
        
        Args:
            ocr_results: List of OCRResult sorted by vertical position
            top_n_lines: Number of top lines to consider
            
        Returns:
            Extracted merchant name or None
        """
        if not ocr_results:
            return None
        
        candidates = []
        
        # Consider only top N lines
        for result in ocr_results[:top_n_lines]:
            text = result.text.strip()
            
            if not text or len(text) < 2:
                continue
            
            # Skip if matches exclude patterns
            if cls._should_exclude(text):
                continue
            
            # Score the candidate
            score = cls._score_merchant_candidate(text, result)
            candidates.append((text, score, result.confidence))
        
        if not candidates:
            # Fallback to first non-empty line
            for result in ocr_results[:3]:
                if result.text.strip():
                    return result.text.strip()
            return None
        
        # Sort by score (descending) and take the best
        candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
        
        logger.info(f"Merchant candidates: {candidates[:3]}")
        return candidates[0][0]
    
    @classmethod
    def _should_exclude(cls, text: str) -> bool:
        """Check if text matches exclusion patterns."""
        text_lower = text.lower()
        for pattern in cls.EXCLUDE_PATTERNS:
            if re.match(pattern, text_lower):
                return True
        return False
    
    @classmethod
    def _score_merchant_candidate(cls, text: str, result: OCRResult) -> float:
        """Score a candidate text for being the merchant name."""
        score = 0.0
        
        # Bonus for being in ALL CAPS (common for store names)
        if text.isupper() and len(text) > 3:
            score += 2.0
        
        # Bonus for reasonable length (3-50 chars)
        if 3 <= len(text) <= 50:
            score += 1.0
        
        # Bonus for being near the top (lower line index)
        score += max(0, 3 - result.line_index)
        
        # Penalty for containing only numbers
        if text.isdigit():
            score -= 5.0
        
        # Bonus for containing letters
        if any(c.isalpha() for c in text):
            score += 1.0
        
        return score


class DateExtractor:
    """
    Extracts transaction date from receipt.
    
    Supports Indonesian and international date formats:
    - DD/MM/YYYY, DD-MM-YYYY
    - YYYY-MM-DD
    - DD MMM YYYY (11 Jan 2026)
    """
    
    @classmethod
    def extract(cls, ocr_results: List[OCRResult]) -> Tuple[Optional[str], Optional[date]]:
        """
        Extract transaction date from OCR results.
        
        Args:
            ocr_results: List of OCRResult objects
            
        Returns:
            Tuple of (raw_date_string, parsed_date) or (None, None)
        """
        # Combine all text for searching
        full_text = " ".join(r.text for r in ocr_results)
        
        for pattern in ExtractionConfig.DATE_PATTERNS:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                raw_string = match.group(0)
                parsed_date = cls._parse_date_match(match, pattern)
                
                if parsed_date and cls._is_valid_date(parsed_date):
                    logger.info(f"Extracted date: {raw_string} -> {parsed_date}")
                    return raw_string, parsed_date
        
        logger.warning("No valid date found in receipt")
        return None, None
    
    @classmethod
    def _parse_date_match(cls, match: re.Match, pattern: str) -> Optional[date]:
        """Parse a regex match into a date object."""
        groups = match.groups()
        
        try:
            # Filter out None values first
            valid_groups = [g for g in groups if g is not None]
            
            # Check if it's a text month pattern
            if any(month in pattern.lower() for month in ["jan", "feb", "mar"]):
                # DD MMM YYYY format - need at least 3 groups
                if len(valid_groups) >= 3:
                    day = int(valid_groups[0])
                    month_str = valid_groups[1].lower()
                    year = int(valid_groups[2])
                    
                    month = ExtractionConfig.MONTH_MAP.get(month_str)
                    if month:
                        return date(year, month, day)
            
            # Numeric patterns
            elif len(valid_groups) >= 3:
                # Get numeric values only
                nums = []
                for g in valid_groups:
                    if isinstance(g, str) and g.isdigit():
                        nums.append(int(g))
                
                if len(nums) >= 3:
                    # Determine format based on pattern
                    if "YYYY" in pattern and pattern.find("YYYY") == 0:
                        # YYYY/MM/DD
                        year, month, day = nums[0], nums[1], nums[2]
                    else:
                        # DD/MM/YYYY
                        day, month, year = nums[0], nums[1], nums[2]
                    
                    # Handle 2-digit years
                    if year < 100:
                        year += 2000 if year < 50 else 1900
                    
                    return date(year, month, day)
        
        except (ValueError, IndexError, TypeError) as e:
            logger.debug(f"Date parsing failed: {e}")
        
        return None
    
    @classmethod
    def _is_valid_date(cls, d: date) -> bool:
        """Check if date is reasonable (not in future, not too old)."""
        today = date.today()
        
        # Not in the future
        if d > today:
            return False
        
        # Not more than 5 years old (for reimbursement purposes)
        min_date = date(today.year - 5, 1, 1)
        if d < min_date:
            return False
        
        return True


class TotalAmountExtractor:
    """
    Extracts the total/grand total amount from receipt.
    
    Strategy (in order of priority):
    1. Find keywords like "Total", "Grand Total", "Bayar" and get associated amount
    2. Get amounts at the bottom of receipt (last 30% of text)
    3. Fall back to maximum amount if ambiguous
    """
    
    @classmethod
    def extract(
        cls,
        ocr_results: List[OCRResult],
        text_lines: List[List[OCRResult]]
    ) -> Tuple[Optional[str], Optional[float], float]:
        """
        Extract total amount from OCR results.
        
        Args:
            ocr_results: List of OCRResult sorted by vertical position
            text_lines: Text organized into lines
            
        Returns:
            Tuple of (raw_amount_string, parsed_value, confidence)
        """
        if not ocr_results:
            return None, None, 0.0
        
        # Strategy 1: Keyword-based extraction
        keyword_result = cls._extract_by_keyword(text_lines)
        if keyword_result[1] is not None:
            logger.info(f"Total found by keyword: {keyword_result}")
            return keyword_result
        
        # Strategy 2: Position-based (bottom of receipt)
        position_result = cls._extract_by_position(ocr_results, text_lines)
        if position_result[1] is not None:
            logger.info(f"Total found by position: {position_result}")
            return position_result
        
        # Strategy 3: Maximum value fallback
        max_result = cls._extract_max_value(ocr_results)
        if max_result[1] is not None:
            logger.info(f"Total found by max value: {max_result}")
            return max_result
        
        logger.warning("Could not extract total amount")
        return None, None, 0.0
    
    @classmethod
    def _extract_by_keyword(
        cls, 
        text_lines: List[List[OCRResult]]
    ) -> Tuple[Optional[str], Optional[float], float]:
        """Find total amount based on keywords."""
        parser = CurrencyParser()
        
        for line in reversed(text_lines):
            # Combine line text
            line_text = " ".join(r.text for r in line).lower()
            
            # Check if line contains exclude keywords
            if any(kw in line_text for kw in ExtractionConfig.EXCLUDE_KEYWORDS):
                continue
            
            # Check for total keywords
            for keyword in ExtractionConfig.TOTAL_KEYWORDS:
                if keyword in line_text:
                    # Found a keyword line, extract amount
                    
                    # Look for amount in this line (right of keyword)
                    for result in reversed(line):  # Right-to-left
                        parsed, raw = parser.parse(result.text)
                        if parsed is not None and parsed > 0:
                            # Validate it's not a small quantity
                            if parsed >= 100:  # Minimum reasonable total
                                return raw, parsed, 0.9
                    
                    # Also check the next line if amount not on same line
                    line_idx = text_lines.index(line)
                    if line_idx + 1 < len(text_lines):
                        next_line = text_lines[line_idx + 1]
                        for result in next_line:
                            parsed, raw = parser.parse(result.text)
                            if parsed is not None and parsed >= 100:
                                return raw, parsed, 0.85
        
        return None, None, 0.0
    
    @classmethod
    def _extract_by_position(
        cls,
        ocr_results: List[OCRResult],
        text_lines: List[List[OCRResult]]
    ) -> Tuple[Optional[str], Optional[float], float]:
        """Extract amount from bottom portion of receipt."""
        if not text_lines:
            return None, None, 0.0
        
        parser = CurrencyParser()
        
        # Look at bottom 30% of lines
        bottom_start = int(len(text_lines) * 0.7)
        bottom_lines = text_lines[bottom_start:]
        
        amounts = []
        
        for line in bottom_lines:
            line_text = " ".join(r.text for r in line).lower()
            
            # Skip if contains exclude keywords
            if any(kw in line_text for kw in ExtractionConfig.EXCLUDE_KEYWORDS):
                continue
            
            for result in line:
                parsed, raw = parser.parse(result.text)
                if parsed is not None and parsed >= 100:
                    amounts.append((raw, parsed, result.confidence))
        
        if amounts:
            # Get the largest amount from bottom section
            amounts.sort(key=lambda x: x[1], reverse=True)
            best = amounts[0]
            return best[0], best[1], 0.75
        
        return None, None, 0.0
    
    @classmethod
    def _extract_max_value(
        cls,
        ocr_results: List[OCRResult]
    ) -> Tuple[Optional[str], Optional[float], float]:
        """Fall back to maximum monetary value found."""
        parser = CurrencyParser()
        amounts = []
        
        for result in ocr_results:
            parsed, raw = parser.parse(result.text)
            # Filter out likely dates/IDs (often 6-8 digit numbers in specific formats)
            if parsed is not None and parsed >= 100:
                # Skip if looks like a date or ID
                if not cls._looks_like_date_or_id(result.text):
                    amounts.append((raw, parsed, result.confidence))
        
        if amounts:
            # Get maximum
            amounts.sort(key=lambda x: x[1], reverse=True)
            best = amounts[0]
            return best[0], best[1], 0.6  # Lower confidence for fallback
        
        return None, None, 0.0
    
    @classmethod
    def _looks_like_date_or_id(cls, text: str) -> bool:
        """Check if text looks like a date or transaction ID."""
        # Check for date patterns
        if re.match(r"^\d{2}[/\-]\d{2}[/\-]\d{2,4}$", text):
            return True
        
        # Check for time patterns
        if re.match(r"^\d{2}:\d{2}(:\d{2})?$", text):
            return True
        
        # Check for receipt/transaction ID patterns
        if re.match(r"^[A-Z]{2,}\d{6,}$", text):
            return True
        
        return False


class ReceiptExtractor:
    """
    Main orchestrator for extracting all fields from a receipt.
    """
    
    def __init__(self):
        self.merchant_extractor = MerchantExtractor()
        self.date_extractor = DateExtractor()
        self.total_extractor = TotalAmountExtractor()
    
    def extract_all(
        self,
        ocr_results: List[OCRResult],
        text_lines: List[List[OCRResult]]
    ) -> dict:
        """
        Extract all fields from OCR results.
        
        Args:
            ocr_results: List of OCRResult objects
            text_lines: Text organized into lines
            
        Returns:
            Dictionary with extracted fields
        """
        # Extract merchant
        merchant_name = self.merchant_extractor.extract(ocr_results)
        
        # Extract date
        date_raw, date_parsed = self.date_extractor.extract(ocr_results)
        
        # Extract total
        total_raw, total_value, total_confidence = self.total_extractor.extract(
            ocr_results, text_lines
        )
        
        # Calculate overall confidence
        ocr_avg_confidence = sum(r.confidence for r in ocr_results) / len(ocr_results) if ocr_results else 0
        
        # Weight: OCR confidence (30%), total extraction confidence (70%)
        overall_confidence = (ocr_avg_confidence * 0.3) + (total_confidence * 0.7)
        
        return {
            "merchant_name": merchant_name,
            "transaction_date_raw": date_raw,
            "transaction_date": date_parsed.isoformat() if date_parsed else None,
            "total_amount_raw": total_raw,
            "total_amount_value": total_value,
            "confidence_score": round(overall_confidence, 3)
        }
