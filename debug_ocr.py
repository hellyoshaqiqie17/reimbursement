from paddleocr import PaddleOCR
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_ocr():
    print("Initializing PaddleOCR...")
    try:
        # Try with minimal arguments
        ocr = PaddleOCR(lang='en')
        print("Initialization successful.")
        
        # Create dummy image (white background, black text)
        img = np.full((100, 300, 3), 255, dtype=np.uint8)
        
        print("Running OCR with cls=False...")
        # Try calling with cls=False
        result = ocr.ocr(img, cls=False)
        print("OCR successful.")
        print(result)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ocr()
