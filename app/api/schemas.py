"""
Pydantic schemas for API request/response validation.
"""
from pydantic import BaseModel, Field
from typing import Optional


class ExtractionResponse(BaseModel):
    """Response schema for receipt extraction endpoint."""
    
    merchant_name: Optional[str] = Field(
        None,
        description="Name of the merchant/store from receipt header"
    )
    
    transaction_date: Optional[str] = Field(
        None,
        description="Transaction date in YYYY-MM-DD format"
    )
    
    total_amount_raw: Optional[str] = Field(
        None,
        description="Raw total amount string as found on receipt (e.g., 'Rp 150.000')"
    )
    
    total_amount_value: Optional[float] = Field(
        None,
        description="Parsed total amount as a float (e.g., 150000.0)"
    )
    
    confidence_score: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score of the extraction (0.0 to 1.0)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "merchant_name": "INDOMARET CABANG SUDIRMAN",
                "transaction_date": "2026-01-11",
                "total_amount_raw": "Rp 150.000",
                "total_amount_value": 150000.0,
                "confidence_score": 0.85
            }
        }


class ExtractionResponseWithDebug(ExtractionResponse):
    """Extended response with debug information."""
    
    transaction_date_raw: Optional[str] = Field(
        None,
        description="Raw date string as found on receipt"
    )
    
    ocr_text: Optional[str] = Field(
        None,
        description="Full OCR text output (for debugging)"
    )
    
    processing_time_ms: Optional[float] = Field(
        None,
        description="Total processing time in milliseconds"
    )


class ErrorResponse(BaseModel):
    """Error response schema."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "InvalidImageError",
                "message": "Failed to process uploaded image",
                "detail": "Unsupported image format: .gif"
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field("healthy", description="Service status")
    version: str = Field(..., description="API version")
    ocr_engine: str = Field("paddleocr", description="OCR engine being used")
