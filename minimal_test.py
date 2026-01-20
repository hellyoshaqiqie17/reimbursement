"""
Minimal test to get exact traceback
"""
import sys
sys.path.insert(0, '.')

import cv2
import traceback

try:
    from app.preprocessing.preprocessor import ReceiptPreprocessor
    from app.ocr.ocr_engine import ReceiptOCR
    from app.parsing.extractors import ReceiptExtractor
    
    image_path = r"C:\Users\MSI MODERN 14\.gemini\antigravity\brain\43e0ddb9-fec7-4a05-a958-73336de759b6\uploaded_image_1768397828756.png"
    
    image = cv2.imread(image_path)
    preprocessor = ReceiptPreprocessor()
    processed = preprocessor.process(image)
    
    ocr = ReceiptOCR(lang="en")
    ocr_results = ocr.extract_text(processed)
    
    text_lines = ocr.get_text_lines(ocr_results)
    
    extractor = ReceiptExtractor()
    result = extractor.extract_all(ocr_results, text_lines)
    
    print("SUCCESS")
    print(result)
    
except Exception as e:
    tb = traceback.format_exc()
    # Write to file with ASCII encoding to avoid unicode issues
    with open("traceback.txt", "w", encoding="ascii", errors="replace") as f:
        f.write(tb)
    print("Error saved to traceback.txt")
    print(type(e).__name__, str(e))
