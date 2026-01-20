"""
Configuration management for Receipt OCR Pipeline.
Contains OCR settings, keyword lists, and regex patterns.
"""
from typing import List
import re


class OCRConfig:
    """PaddleOCR configuration settings."""
    
    # Language settings
    LANG: str = "en"  # PaddleOCR language code
    USE_ANGLE_CLS: bool = True  # Enable angle classification
    USE_GPU: bool = False  # Set True if CUDA available
    
    # Detection settings
    DET_DB_THRESH: float = 0.3
    DET_DB_BOX_THRESH: float = 0.5
    
    # Recognition settings
    REC_BATCH_NUM: int = 6
    DROP_SCORE: float = 0.5  # Minimum confidence threshold


class ExtractionConfig:
    """Configuration for field extraction logic."""
    
    # Keywords for total amount detection (Indonesian + English)
    TOTAL_KEYWORDS: List[str] = [
        "grand total",
        "total",
        "jumlah",
        "bayar",
        "amount",
        "tunai",
        "cash",
        "subtotal",
        "sub total",
        "pembayaran",
        "tagihan",
        "total bayar",
        "total belanja",
    ]
    
    # Keywords that indicate NOT a total (to filter out)
    EXCLUDE_KEYWORDS: List[str] = [
        "diskon",
        "discount",
        "ppn",
        "tax",
        "pajak",
        "kembalian",
        "change",
        "item",
        "qty",
        "quantity",
    ]
    
    # Date format patterns (Indonesian common formats)
    DATE_PATTERNS: List[str] = [
        r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})\b",  # DD/MM/YYYY or DD-MM-YYYY
        r"\b(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})\b",  # YYYY/MM/DD or YYYY-MM-DD
        r"\b(\d{1,2})\s+(jan|feb|mar|apr|may|mei|jun|jul|aug|agu|sep|oct|okt|nov|dec|des)\w*\s+(\d{4})\b",  # DD MMM YYYY
        r"\b(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})\b",
    ]
    
    # Month mappings for Indonesian
    MONTH_MAP: dict = {
        "jan": 1, "january": 1, "januari": 1,
        "feb": 2, "february": 2, "februari": 2,
        "mar": 3, "march": 3, "maret": 3,
        "apr": 4, "april": 4,
        "may": 5, "mei": 5,
        "jun": 6, "june": 6, "juni": 6,
        "jul": 7, "july": 7, "juli": 7,
        "aug": 8, "august": 8, "agu": 8, "agustus": 8,
        "sep": 9, "september": 9,
        "oct": 10, "october": 10, "okt": 10, "oktober": 10,
        "nov": 11, "november": 11,
        "dec": 12, "december": 12, "des": 12, "desember": 12,
    }


class CurrencyConfig:
    """Configuration for currency parsing."""
    
    # Currency symbols to remove
    CURRENCY_SYMBOLS: List[str] = ["rp", "rp.", "idr", "$", "usd"]
    
    # Pattern to extract numeric value from Indonesian currency format
    # Handles: Rp 50.000,00 | 50,000 | 50.000 | Rp50000
    CURRENCY_PATTERN: str = r"[Rr][Pp]\.?\s*|[Ii][Dd][Rr]\s*|\$\s*"
    
    # Indonesian format uses . for thousands and , for decimals
    # International format uses , for thousands and . for decimals


class ImageConfig:
    """Configuration for image preprocessing."""
    
    # Supported image formats
    SUPPORTED_FORMATS: List[str] = [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"]
    
    # Maximum image dimensions (to prevent memory issues)
    MAX_WIDTH: int = 1280
    MAX_HEIGHT: int = 1280
    
    # Preprocessing settings
    DENOISE_STRENGTH: int = 10
    CLAHE_CLIP_LIMIT: float = 2.0
    CLAHE_TILE_SIZE: tuple = (8, 8)
