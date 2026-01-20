"""
Minimal test that prints exact line number
"""
import sys
import linecache
sys.path.insert(0, '.')

import cv2
import traceback

try:
    from app.preprocessing.preprocessor import ReceiptPreprocessor
    from app.ocr.ocr_engine import ReceiptOCR
    from app.parsing.extractors import ReceiptExtractor
    
    image_path = r"C:\Users\MSI MODERN 14\.gemini\antigravity\brain\43e0ddb9-fec7-4a05-a958-73336de759b6\uploaded_image_1768397828756.png"
    
    print("Step 1: Loading image")
    image = cv2.imread(image_path)
    print(f"Image loaded: {image.shape}")
    
    print("Step 2: Preprocessing")
    preprocessor = ReceiptPreprocessor()
    processed = preprocessor.process(image)
    print(f"Processed: {processed.shape}")
    
    print("Step 3: OCR")
    ocr = ReceiptOCR(lang="en")
    ocr_results = ocr.extract_text(processed)
    print(f"OCR results: {len(ocr_results)}")
    
    print("Step 4: Get text lines")
    text_lines = ocr.get_text_lines(ocr_results)
    print(f"Text lines: {len(text_lines)}")
    
    print("Step 5: Extract")
    extractor = ReceiptExtractor()
    result = extractor.extract_all(ocr_results, text_lines)
    
    print("SUCCESS")
    for k, v in result.items():
        print(f"  {k}: {v}")
    
except Exception as e:
    ex_type, ex_value, ex_tb = sys.exc_info()
    # Get the innermost frame
    tb = traceback.extract_tb(ex_tb)
    for frame in tb:
        filename, lineno, funcname, text = frame
        print(f"File: {filename}")
        print(f"Line {lineno} in {funcname}")
        print(f"Code: {text}")
        print("---")
    print(f"Exception: {ex_type.__name__}: {ex_value}")
