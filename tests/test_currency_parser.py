"""
Unit tests for Currency Parser module.
"""
import pytest
from app.parsing.currency_parser import CurrencyParser, extract_all_amounts


class TestCurrencyParser:
    """Test cases for CurrencyParser class."""
    
    def test_parse_indonesian_format_with_rp(self):
        """Test parsing Indonesian Rupiah with Rp prefix."""
        value, raw = CurrencyParser.parse("Rp 50.000")
        assert value == 50000.0
        assert raw == "Rp 50.000"
    
    def test_parse_indonesian_format_with_decimals(self):
        """Test parsing with comma as decimal separator."""
        value, raw = CurrencyParser.parse("Rp 50.000,00")
        assert value == 50000.0
    
    def test_parse_large_amount(self):
        """Test parsing large amounts with multiple thousand separators."""
        value, raw = CurrencyParser.parse("Rp 1.234.567")
        assert value == 1234567.0
    
    def test_parse_with_rp_dot(self):
        """Test parsing with 'Rp.' prefix."""
        value, raw = CurrencyParser.parse("Rp. 75.000")
        assert value == 75000.0
    
    def test_parse_idr_prefix(self):
        """Test parsing with IDR prefix."""
        value, raw = CurrencyParser.parse("IDR 100.000")
        assert value == 100000.0
    
    def test_parse_no_prefix(self):
        """Test parsing amount without currency prefix."""
        value, raw = CurrencyParser.parse("50.000")
        assert value == 50000.0
    
    def test_parse_international_format(self):
        """Test parsing international format (comma for thousands)."""
        value, raw = CurrencyParser.parse("50,000")
        assert value == 50000.0
    
    def test_parse_international_with_decimals(self):
        """Test parsing international format with decimals."""
        value, raw = CurrencyParser.parse("50,000.00")
        assert value == 50000.0
    
    def test_parse_plain_number(self):
        """Test parsing plain number without separators."""
        value, raw = CurrencyParser.parse("50000")
        assert value == 50000.0
    
    def test_parse_with_whitespace(self):
        """Test parsing with extra whitespace."""
        value, raw = CurrencyParser.parse("  Rp  50.000  ")
        assert value == 50000.0
    
    def test_parse_lowercase_rp(self):
        """Test parsing with lowercase 'rp'."""
        value, raw = CurrencyParser.parse("rp 25.000")
        assert value == 25000.0
    
    def test_parse_empty_string(self):
        """Test parsing empty string returns None."""
        value, raw = CurrencyParser.parse("")
        assert value is None
    
    def test_parse_none_input(self):
        """Test parsing None input returns None."""
        value, raw = CurrencyParser.parse(None)
        assert value is None
    
    def test_parse_non_numeric(self):
        """Test parsing non-numeric string returns None."""
        value, raw = CurrencyParser.parse("Hello World")
        assert value is None
    
    def test_parse_small_decimal(self):
        """Test parsing small amount with decimals (ambiguous format)."""
        value, raw = CurrencyParser.parse("50,00")
        # Should interpret as 50.00 (comma as decimal)
        assert value == 50.0
    
    def test_format_as_integer(self):
        """Test conversion to integer."""
        value, _ = CurrencyParser.parse("Rp 50.500")
        integer = CurrencyParser.format_as_integer(value)
        assert integer == 50500
    
    def test_format_as_integer_with_rounding(self):
        """Test integer conversion rounds correctly."""
        value, _ = CurrencyParser.parse("50,60")
        integer = CurrencyParser.format_as_integer(value)
        assert integer == 51  # 50.60 rounds to 51
    
    def test_is_valid_amount_positive(self):
        """Test valid amount validation."""
        assert CurrencyParser.is_valid_amount(50000.0) is True
    
    def test_is_valid_amount_negative(self):
        """Test invalid negative amount."""
        assert CurrencyParser.is_valid_amount(-100.0) is False
    
    def test_is_valid_amount_none(self):
        """Test None is invalid."""
        assert CurrencyParser.is_valid_amount(None) is False
    
    def test_is_valid_amount_too_large(self):
        """Test amount exceeding max value."""
        assert CurrencyParser.is_valid_amount(1e15) is False


class TestExtractAllAmounts:
    """Test cases for extract_all_amounts function."""
    
    def test_extract_single_amount(self):
        """Test extracting single amount from text."""
        results = extract_all_amounts("Total: Rp 50.000")
        assert len(results) >= 1
        # Find the main amount
        amounts = [r[1] for r in results]
        assert 50000.0 in amounts
    
    def test_extract_multiple_amounts(self):
        """Test extracting multiple amounts from text."""
        text = "Item 1: Rp 10.000\nItem 2: Rp 20.000\nTotal: Rp 30.000"
        results = extract_all_amounts(text)
        amounts = [r[1] for r in results]
        assert 10000.0 in amounts
        assert 20000.0 in amounts
        assert 30000.0 in amounts
    
    def test_extract_no_amounts(self):
        """Test extracting from text with no amounts."""
        results = extract_all_amounts("Hello World")
        assert len(results) == 0
    
    def test_extract_with_positions(self):
        """Test that positions are returned correctly."""
        text = "Price: Rp 50.000"
        results = extract_all_amounts(text)
        assert len(results) >= 1
        # Each result should have (raw_string, value, position)
        for raw, value, pos in results:
            assert isinstance(raw, str)
            assert isinstance(value, float)
            assert isinstance(pos, int)
            assert pos >= 0


# Additional edge case tests
class TestEdgeCases:
    """Test edge cases and real-world receipt formats."""
    
    def test_indonesian_receipt_total(self):
        """Test common Indonesian receipt total format."""
        value, _ = CurrencyParser.parse("TOTAL Rp 125.500")
        assert value == 125500.0
    
    def test_minimarket_receipt(self):
        """Test Indomaret/Alfamart style receipt."""
        value, _ = CurrencyParser.parse("TUNAI    150.000")
        assert value == 150000.0
    
    def test_restaurant_receipt(self):
        """Test restaurant receipt with 'Jumlah'."""
        value, _ = CurrencyParser.parse("Jumlah : Rp. 250.000,00")
        assert value == 250000.0
    
    def test_mixed_text_and_number(self):
        """Test extracting number from mixed content."""
        value, _ = CurrencyParser.parse("Grand Total: 175.000")
        assert value == 175000.0
