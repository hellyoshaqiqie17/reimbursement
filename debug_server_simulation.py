import numpy as np
import logging
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from app.ocr.ocr_engine import ReceiptOCR

def test_server_ocr():
    print("Initializing ReceiptOCR...")
    try:
        ocr_engine = ReceiptOCR()
        
        # Create dummy image (white background, black text)
        # Match the shape/type likely produced by preprocessor
        img = np.full((1000, 1000, 3), 255, dtype=np.uint8)
        
        print("Running extract_text...")
        results = ocr_engine.extract_text(img)
        
        print("OCR successful.")
        print(f"Found {len(results)} results.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_server_ocr()
