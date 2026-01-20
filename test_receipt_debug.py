"""
Test script to debug the tuple index error with real receipt image.
"""
import sys
import traceback
import numpy as np
import cv2

sys.path.insert(0, '.')

from app.preprocessing.preprocessor import ReceiptPreprocessor, load_image_from_bytes
from app.ocr.ocr_engine import ReceiptOCR
from app.parsing.extractors import ReceiptExtractor

def log(msg):
    print(msg, flush=True)
    with open("debug_log.txt", "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def test_with_image(image_path: str):
    # Clear log
    with open("debug_log.txt", "w", encoding="utf-8") as f:
        f.write("")
    
    log(f"Testing with image: {image_path}")
    
    try:
        # Load image
        log("Loading image...")
        image = cv2.imread(image_path)
        if image is None:
            log(f"ERROR: Could not load image from {image_path}")
            return
        log(f"Image shape: {image.shape}")
        
        # Preprocess
        log("Preprocessing...")
        preprocessor = ReceiptPreprocessor()
        processed = preprocessor.process(image)
        log(f"Processed shape: {processed.shape}")
        
        # OCR
        log("Running OCR...")
        ocr = ReceiptOCR(lang="en")
        ocr_results = ocr.extract_text(processed)
        log(f"OCR results count: {len(ocr_results)}")
        
        if ocr_results:
            log("First 5 OCR results:")
            for i, r in enumerate(ocr_results[:5]):
                log(f"  {i}: text='{r.text}', conf={r.confidence:.2f}")
        
        # Get text lines
        log("Getting text lines...")
        text_lines = ocr.get_text_lines(ocr_results)
        log(f"Text lines count: {len(text_lines)}")
        
        # Extract
        log("Extracting fields...")
        extractor = ReceiptExtractor()
        result = extractor.extract_all(ocr_results, text_lines)
        
        log("Extraction result:")
        for k, v in result.items():
            log(f"  {k}: {v}")
            
        log("SUCCESS!")
            
    except Exception as e:
        log(f"=== ERROR ===")
        log(f"Error type: {type(e).__name__}")
        log(f"Error message: {e}")
        log("Full traceback:")
        tb = traceback.format_exc()
        log(tb)

if __name__ == "__main__":
    # Test with user's uploaded image
    test_image = r"C:\Users\MSI MODERN 14\.gemini\antigravity\brain\43e0ddb9-fec7-4a05-a958-73336de759b6\uploaded_image_1768397828756.png"
    test_with_image(test_image)
