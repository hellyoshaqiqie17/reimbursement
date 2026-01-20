"""
Currency Parser Module.
Handles Indonesian Rupiah and other currency format parsing.
Converts formatted currency strings to numeric values.
"""
import re
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class CurrencyParser:
    """
    Parser for currency strings commonly found on Indonesian receipts.
    
    Handles formats:
    - Rp 50.000,00
    - Rp. 50,000
    - Rp50000
    - IDR 50.000
    - 50.000,00
    - 50,000
    """
    
    # Currency symbol patterns to remove
    CURRENCY_PREFIXES = [
        r"[Rr][Pp]\.?\s*",  # Rp, Rp., rp
        r"[Ii][Dd][Rr]\s*",  # IDR, idr
        r"\$\s*",            # USD
        r"[Uu][Ss][Dd]\s*",  # USD
    ]
    
    # Pattern to match number with various separators
    NUMBER_PATTERN = re.compile(
        r"^[^\d]*"  # Skip non-digits at start
        r"([\d.,\s]+)"  # Capture digits with separators
        r"[^\d]*$"  # Skip non-digits at end
    )
    
    @classmethod
    def parse(cls, currency_string: str) -> Tuple[Optional[float], str]:
        """
        Parse a currency string into a numeric value.
        
        Args:
            currency_string: Raw currency string (e.g., "Rp 50.000,00")
            
        Returns:
            Tuple of (parsed_value, cleaned_string)
            Returns (None, original_string) if parsing fails
        """
        if not currency_string or not isinstance(currency_string, str):
            return None, str(currency_string) if currency_string else ""
        
        original = currency_string.strip()
        cleaned = original
        
        # Remove currency prefixes
        for prefix_pattern in cls.CURRENCY_PREFIXES:
            cleaned = re.sub(prefix_pattern, "", cleaned, flags=re.IGNORECASE)
        
        cleaned = cleaned.strip()
        
        if not cleaned:
            return None, original
        
        # Try to extract the numeric portion
        match = cls.NUMBER_PATTERN.match(cleaned)
        if not match:
            logger.debug(f"No numeric pattern found in: {cleaned}")
            return None, original
        
        number_str = match.group(1).strip()
        
        # Determine the format based on separator patterns
        value = cls._parse_number_string(number_str)
        
        if value is not None:
            logger.debug(f"Parsed '{original}' -> {value}")
            return value, original
        
        return None, original
    
    @classmethod
    def _parse_number_string(cls, number_str: str) -> Optional[float]:
        """
        Parse a number string with various separator formats.
        
        Indonesian format: . for thousands, , for decimals (50.000,00 = 50000.00)
        International format: , for thousands, . for decimals (50,000.00 = 50000.00)
        """
        # Remove any whitespace within the number
        number_str = re.sub(r"\s+", "", number_str)
        
        if not number_str:
            return None
        
        # Count separators
        dots = number_str.count(".")
        commas = number_str.count(",")
        
        try:
            if dots == 0 and commas == 0:
                # Plain number: 50000
                return float(number_str)
            
            elif dots == 0 and commas == 1:
                # Could be decimal (50,00) or thousands (50,000)
                # Check position of comma
                parts = number_str.split(",")
                if len(parts[1]) == 2:
                    # Likely decimal: 50,00 -> 50.00
                    return float(number_str.replace(",", "."))
                elif len(parts[1]) == 3:
                    # Likely thousands: 50,000 -> 50000
                    return float(number_str.replace(",", ""))
                else:
                    # Ambiguous, treat as decimal
                    return float(number_str.replace(",", "."))
            
            elif dots == 1 and commas == 0:
                # Could be decimal (50.00) or thousands (50.000)
                parts = number_str.split(".")
                if len(parts[1]) == 2:
                    # Likely decimal: 50.00
                    return float(number_str)
                elif len(parts[1]) == 3:
                    # Likely thousands (Indonesian): 50.000 -> 50000
                    return float(number_str.replace(".", ""))
                else:
                    # Treat as decimal
                    return float(number_str)
            
            elif dots >= 1 and commas >= 1:
                # Mixed separators: check which one comes last to determine decimal separator
                last_dot = number_str.rfind(".")
                last_comma = number_str.rfind(",")
                
                if last_comma > last_dot:
                    # Indonesian format: 50.000,00 (comma is decimal)
                    normalized = number_str.replace(".", "").replace(",", ".")
                    return float(normalized)
                else:
                    # International format: 50,000.00 (dot is decimal)
                    normalized = number_str.replace(",", "")
                    return float(normalized)
            
            elif dots > 1 and commas == 0:
                # Multiple dots, treat as thousands separators (Indonesian): 1.234.567
                normalized = number_str.replace(".", "")
                return float(normalized)
            
            elif dots == 0 and commas > 1:
                # Multiple commas, treat as thousands separators (International): 1,234,567
                normalized = number_str.replace(",", "")
                return float(normalized)
            
            else:
                # Complex case, try best effort
                logger.warning(f"Ambiguous number format: {number_str}")
                # Assume last separator is decimal
                last_dot = number_str.rfind(".")
                last_comma = number_str.rfind(",")
                
                if last_comma > last_dot:
                    # Indonesian: comma is decimal
                    normalized = number_str.replace(".", "").replace(",", ".")
                else:
                    # International: dot is decimal
                    normalized = number_str.replace(",", "")
                
                return float(normalized)
                
        except ValueError as e:
            logger.debug(f"Failed to parse number '{number_str}': {e}")
            return None
    
    @classmethod
    def format_as_integer(cls, value: Optional[float]) -> Optional[int]:
        """Convert parsed value to integer (for Rupiah, no cents)."""
        if value is None:
            return None
        return int(round(value))
    
    @classmethod
    def is_valid_amount(cls, value: Optional[float], min_value: float = 0, max_value: float = 1e12) -> bool:
        """
        Validate that a parsed amount is reasonable.
        
        Args:
            value: Parsed currency value
            min_value: Minimum acceptable value (default 0)
            max_value: Maximum acceptable value (default 1 trillion)
            
        Returns:
            True if value is valid and within range
        """
        if value is None:
            return False
        return min_value <= value <= max_value


def extract_all_amounts(text: str) -> list:
    """
    Extract all potential monetary amounts from a text string.
    
    Args:
        text: Raw text to scan for amounts
        
    Returns:
        List of tuples: (raw_string, parsed_value, start_position)
    """
    # Pattern to find currency-like strings
    # Matches: Rp 50.000, 50,000, 50.000,00, etc.
    pattern = re.compile(
        r"(?:[Rr][Pp]\.?\s*|[Ii][Dd][Rr]\s*)?"  # Optional currency prefix
        r"(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?)",  # Number with separators
        re.IGNORECASE
    )
    
    results = []
    parser = CurrencyParser()
    
    for match in pattern.finditer(text):
        raw_string = match.group(0)
        start_pos = match.start()
        
        parsed_value, _ = parser.parse(raw_string)
        
        if parsed_value is not None and parsed_value > 0:
            results.append((raw_string, parsed_value, start_pos))
    
    return results
