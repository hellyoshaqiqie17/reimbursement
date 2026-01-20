"""
API Routes for Receipt OCR Pipeline.
"""
import time
import logging
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse

from app.api.schemas import (
    ExtractionResponse,
    ExtractionResponseWithDebug,
    ErrorResponse,
    HealthResponse
)
from app.preprocessing.preprocessor import ReceiptPreprocessor, load_image_from_bytes
from app.ocr.ocr_engine import ReceiptOCR
from app.parsing.extractors import ReceiptExtractor

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize components (lazy-loaded)
_preprocessor: Optional[ReceiptPreprocessor] = None
_ocr_engine: Optional[ReceiptOCR] = None
_extractor: Optional[ReceiptExtractor] = None


def get_preprocessor() -> ReceiptPreprocessor:
    """Get or create preprocessor instance."""
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = ReceiptPreprocessor()
    return _preprocessor


def get_ocr_engine() -> ReceiptOCR:
    """Get or create OCR engine instance."""
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = ReceiptOCR(lang="en", use_gpu=False)
    return _ocr_engine


def get_extractor() -> ReceiptExtractor:
    """Get or create extractor instance."""
    global _extractor
    if _extractor is None:
        _extractor = ReceiptExtractor()
    return _extractor


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check if the API is running and healthy"
)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        ocr_engine="paddleocr"
    )


@router.post(
    "/extract",
    response_model=ExtractionResponse,
    responses={
        200: {"model": ExtractionResponse, "description": "Successful extraction"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Processing error"}
    },
    summary="Extract Receipt Data",
    description="Upload a receipt image and extract merchant name, date, and total amount"
)
async def extract_receipt(
    file: UploadFile = File(..., description="Receipt image file (JPEG, PNG, etc.)"),
    debug: bool = Query(False, description="Include debug information in response")
):
    """
    Extract data from a receipt image.
    
    - **file**: Receipt image to process
    - **debug**: If true, includes OCR text and processing time in response
    """
    start_time = time.time()
    
    # Validate file type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "InvalidFileType",
                "message": f"Expected image file, got: {file.content_type}"
            }
        )
    
    try:
        # Read file content
        content = await file.read()
        
        if not content:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "EmptyFile",
                    "message": "Uploaded file is empty"
                }
            )
        
        logger.info(f"Processing receipt: {file.filename}, size: {len(content)} bytes")
        
        # Load image from bytes
        image = load_image_from_bytes(content)
        
        # Preprocess image
        preprocessor = get_preprocessor()
        processed_image = preprocessor.process(image)
        
        # Run OCR
        ocr_engine = get_ocr_engine()
        ocr_results = ocr_engine.extract_text(processed_image)
        
        if not ocr_results:
            return ExtractionResponse(
                merchant_name=None,
                transaction_date=None,
                total_amount_raw=None,
                total_amount_value=None,
                confidence_score=0.0
            )
        
        # Extract fields
        text_lines = ocr_engine.get_text_lines(ocr_results)
        extractor = get_extractor()
        extracted = extractor.extract_all(ocr_results, text_lines)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        logger.info(f"Extraction completed in {processing_time:.2f}ms: {extracted}")
        
        # Build response
        if debug:
            full_text = ocr_engine.get_full_text(ocr_results)
            return ExtractionResponseWithDebug(
                merchant_name=extracted["merchant_name"],
                transaction_date=extracted["transaction_date"],
                transaction_date_raw=extracted["transaction_date_raw"],
                total_amount_raw=extracted["total_amount_raw"],
                total_amount_value=extracted["total_amount_value"],
                confidence_score=extracted["confidence_score"],
                ocr_text=full_text,
                processing_time_ms=round(processing_time, 2)
            )
        else:
            return ExtractionResponse(
                merchant_name=extracted["merchant_name"],
                transaction_date=extracted["transaction_date"],
                total_amount_raw=extracted["total_amount_raw"],
                total_amount_value=extracted["total_amount_value"],
                confidence_score=extracted["confidence_score"]
            )
    
    except ValueError as e:
        logger.error(f"Value error processing receipt: {e}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "InvalidImageError",
                "message": str(e)
            }
        )
    
    except Exception as e:
        logger.exception(f"Unexpected error processing receipt: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "ProcessingError",
                "message": "Failed to process receipt image",
                "detail": str(e)
            }
        )
