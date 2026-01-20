"""
Unit tests for Field Extractors module.
"""
import pytest
from datetime import date
from unittest.mock import MagicMock

from app.parsing.extractors import (
    MerchantExtractor,
    DateExtractor,
    TotalAmountExtractor,
    ReceiptExtractor
)
from app.ocr.ocr_engine import OCRResult


def create_ocr_result(text: str, y_position: float, confidence: float = 0.95) -> OCRResult:
    """Helper to create OCRResult for testing."""
    # Create a simple bounding box centered at y_position
    bbox = [
        [0, y_position - 10],
        [100, y_position - 10],
        [100, y_position + 10],
        [0, y_position + 10]
    ]
    result = OCRResult(text=text, confidence=confidence, bbox=bbox)
    return result


class TestMerchantExtractor:
    """Test cases for MerchantExtractor class."""
    
    def test_extract_from_header(self):
        """Test extracting merchant from receipt header."""
        results = [
            create_ocr_result("INDOMARET", 10),
            create_ocr_result("Jl. Sudirman No. 123", 30),
            create_ocr_result("11/01/2026 14:30", 50),
        ]
        for i, r in enumerate(results):
            r.line_index = i
        
        merchant = MerchantExtractor.extract(results)
        assert merchant == "INDOMARET"
    
    def test_skip_date_line(self):
        """Test that date lines are skipped."""
        results = [
            create_ocr_result("11/01/2026", 10),
            create_ocr_result("TOKO MAJU", 30),
        ]
        for i, r in enumerate(results):
            r.line_index = i
        
        merchant = MerchantExtractor.extract(results)
        assert merchant == "TOKO MAJU"
    
    def test_skip_phone_number(self):
        """Test that phone numbers are skipped."""
        results = [
            create_ocr_result("+62 812 3456 7890", 10),
            create_ocr_result("SUPERMARKET ABC", 30),
        ]
        for i, r in enumerate(results):
            r.line_index = i
        
        merchant = MerchantExtractor.extract(results)
        assert merchant == "SUPERMARKET ABC"
    
    def test_prefer_uppercase(self):
        """Test that uppercase names get higher score when on same line level."""
        results = [
            create_ocr_result("struk belanja", 10),  # Will be excluded by pattern
            create_ocr_result("WARUNG MAKAN SEDERHANA", 30),
        ]
        for i, r in enumerate(results):
            r.line_index = i
        
        merchant = MerchantExtractor.extract(results)
        # "struk" is excluded by pattern, so WARUNG should be extracted
        assert merchant == "WARUNG MAKAN SEDERHANA"
    
    def test_empty_results(self):
        """Test with empty results list."""
        merchant = MerchantExtractor.extract([])
        assert merchant is None


class TestDateExtractor:
    """Test cases for DateExtractor class."""
    
    def test_extract_dd_mm_yyyy_slash(self):
        """Test extracting DD/MM/YYYY format."""
        results = [
            create_ocr_result("Tanggal: 11/01/2026", 100),
        ]
        raw, parsed = DateExtractor.extract(results)
        assert parsed == date(2026, 1, 11)
        assert "11/01/2026" in raw
    
    def test_extract_dd_mm_yyyy_dash(self):
        """Test extracting DD-MM-YYYY format."""
        results = [
            create_ocr_result("Date: 10-01-2026", 100),
        ]
        raw, parsed = DateExtractor.extract(results)
        assert parsed == date(2026, 1, 10)
    
    def test_extract_text_month(self):
        """Test extracting 'DD MMM YYYY' format."""
        results = [
            create_ocr_result("11 Jan 2026", 100),
        ]
        raw, parsed = DateExtractor.extract(results)
        assert parsed == date(2026, 1, 11)
    
    def test_extract_indonesian_month(self):
        """Test extracting Indonesian month name."""
        results = [
            create_ocr_result("25 Desember 2025", 100),
        ]
        raw, parsed = DateExtractor.extract(results)
        assert parsed == date(2025, 12, 25)
    
    def test_no_date_found(self):
        """Test when no date is present."""
        results = [
            create_ocr_result("TOKO ABC", 10),
            create_ocr_result("Item 1: 50000", 50),
        ]
        raw, parsed = DateExtractor.extract(results)
        assert raw is None
        assert parsed is None


class TestTotalAmountExtractor:
    """Test cases for TotalAmountExtractor class."""
    
    def create_text_lines(self, ocr_results):
        """Helper to create text lines from results."""
        return [[r] for r in ocr_results]
    
    def test_extract_by_keyword_total(self):
        """Test extracting total by 'Total' keyword."""
        results = [
            create_ocr_result("Item 1", 10),
            create_ocr_result("10.000", 30),
            create_ocr_result("Total", 100),
            create_ocr_result("Rp 150.000", 100),
        ]
        text_lines = [
            [results[0]],
            [results[1]],
            [results[2], results[3]],  # Total and amount on same line
        ]
        
        raw, value, confidence = TotalAmountExtractor.extract(results, text_lines)
        assert value == 150000.0
        assert confidence >= 0.85
    
    def test_extract_by_keyword_grand_total(self):
        """Test extracting by 'Grand Total' keyword."""
        results = [
            create_ocr_result("Subtotal", 50),
            create_ocr_result("100.000", 50),
            create_ocr_result("Grand Total", 100),
            create_ocr_result("125.000", 100),
        ]
        text_lines = [
            [results[0], results[1]],
            [results[2], results[3]],
        ]
        
        raw, value, confidence = TotalAmountExtractor.extract(results, text_lines)
        assert value == 125000.0
    
    def test_extract_by_keyword_bayar(self):
        """Test extracting by Indonesian 'Bayar' keyword."""
        results = [
            create_ocr_result("Bayar", 100),
            create_ocr_result("200.000", 100),
        ]
        text_lines = [[results[0], results[1]]]
        
        raw, value, confidence = TotalAmountExtractor.extract(results, text_lines)
        assert value == 200000.0
    
    def test_skip_discount_line(self):
        """Test that discount lines are skipped."""
        results = [
            create_ocr_result("Diskon", 50),
            create_ocr_result("50.000", 50),
            create_ocr_result("Total", 100),
            create_ocr_result("150.000", 100),
        ]
        text_lines = [
            [results[0], results[1]],
            [results[2], results[3]],
        ]
        
        raw, value, confidence = TotalAmountExtractor.extract(results, text_lines)
        assert value == 150000.0  # Should get Total, not Diskon
    
    def test_fallback_to_max_value(self):
        """Test fallback to maximum value when no keywords found."""
        results = [
            create_ocr_result("10.000", 10),
            create_ocr_result("25.000", 30),
            create_ocr_result("100.000", 50),  # Largest
            create_ocr_result("Thanks", 100),
        ]
        text_lines = self.create_text_lines(results)
        
        raw, value, confidence = TotalAmountExtractor.extract(results, text_lines)
        assert value == 100000.0
        # This actually triggers position-based extraction (bottom 30%) which has 0.75 confidence
        # or fallback which has 0.6. Either way, we expect a reasonable confidence.
        assert confidence >= 0.6
    
    def test_empty_results(self):
        """Test with empty results."""
        raw, value, confidence = TotalAmountExtractor.extract([], [])
        assert raw is None
        assert value is None
        assert confidence == 0.0


class TestReceiptExtractor:
    """Test cases for ReceiptExtractor orchestrator class."""
    
    def test_extract_all_fields(self):
        """Test extracting all fields from a complete receipt."""
        results = [
            create_ocr_result("INDOMARET", 10),
            create_ocr_result("Jl. Sudirman 123", 30),
            create_ocr_result("11/01/2026 14:30", 50),
            create_ocr_result("Item 1", 70),
            create_ocr_result("25.000", 70),
            create_ocr_result("Item 2", 90),
            create_ocr_result("30.000", 90),
            create_ocr_result("Total", 130),
            create_ocr_result("Rp 55.000", 130),
        ]
        for i, r in enumerate(results):
            r.line_index = i
        
        text_lines = [
            [results[0]],
            [results[1]],
            [results[2]],
            [results[3], results[4]],
            [results[5], results[6]],
            [results[7], results[8]],
        ]
        
        extractor = ReceiptExtractor()
        extracted = extractor.extract_all(results, text_lines)
        
        assert extracted["merchant_name"] == "INDOMARET"
        assert extracted["transaction_date"] == "2026-01-11"
        assert extracted["total_amount_value"] == 55000.0
        assert extracted["confidence_score"] > 0
    
    def test_partial_extraction(self):
        """Test extraction when some fields are missing."""
        results = [
            create_ocr_result("Unknown Store", 10),
            create_ocr_result("Total: 100.000", 50),
        ]
        for i, r in enumerate(results):
            r.line_index = i
        
        text_lines = [[r] for r in results]
        
        extractor = ReceiptExtractor()
        extracted = extractor.extract_all(results, text_lines)
        
        assert extracted["merchant_name"] == "Unknown Store"
        assert extracted["transaction_date"] is None  # No date found
        assert extracted["total_amount_value"] == 100000.0
