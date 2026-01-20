"""
FastAPI Application Entry Point for Receipt OCR Pipeline.

Usage:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router as api_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown events."""
    # Startup
    logger.info("Starting Receipt OCR Pipeline API...")
    logger.info("API documentation available at /docs")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Receipt OCR Pipeline API...")


# Create FastAPI application
app = FastAPI(
    title="Receipt OCR Pipeline",
    description="""
## Receipt OCR Pipeline for Reimbursement System

This API automatically extracts data from receipt images for reimbursement requests:

- **Merchant Name**: Store/vendor name from receipt header
- **Transaction Date**: Date of purchase
- **Total Amount**: Grand total / final amount to pay

### Features
- Image preprocessing (deskewing, denoising, contrast enhancement)
- PaddleOCR-powered text extraction
- Indonesian currency format support (Rp, IDR)
- Keyword-based total amount detection

### Supported Image Formats
JPEG, PNG, TIFF, BMP, WebP
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(
    api_router,
    prefix="/api/v1",
    tags=["Receipt Extraction"]
)


# Root endpoint
@app.get("/", include_in_schema=False)
async def root():
    """Redirect to API documentation."""
    return JSONResponse(
        content={
            "message": "Receipt OCR Pipeline API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/api/v1/health"
        }
    )


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle uncaught exceptions."""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
