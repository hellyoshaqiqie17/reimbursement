"""
Test with user's specific image file - output to file
"""
import sys
sys.path.insert(0, '.')

import cv2

# Redirect all output to file
output_file = open("test_result.log", "w", encoding="utf-8")
def log(msg):
    output_file.write(str(msg) + "\n")
    output_file.flush()

try:
    from app.preprocessing.preprocessor import ReceiptPreprocessor
    from app.ocr.ocr_engine import ReceiptOCR
    from app.parsing.extractors import ReceiptExtractor

    image_path = r"c:\reimbursement\untuk_test\Screenshot 2026-01-11 181638.png"

    log(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        log("ERROR: Could not load image")
        exit(1)
    log(f"Image shape: {image.shape}")

    log("Preprocessing...")
    preprocessor = ReceiptPreprocessor()
    processed = preprocessor.process(image)
    log(f"Processed shape: {processed.shape}")

    log("Running OCR...")
    ocr = ReceiptOCR(lang="en")
    ocr_results = ocr.extract_text(processed)
    log(f"OCR results count: {len(ocr_results)}")

    if ocr_results:
        log("=== ALL OCR TEXT ===")
        for i, r in enumerate(ocr_results):
            log(f"{i}: '{r.text}' (conf={r.confidence:.2f})")
        
        log("=== TEXT LINES ===")
        text_lines = ocr.get_text_lines(ocr_results)
        for i, line in enumerate(text_lines):
            line_text = " ".join([r.text for r in line])
            log(f"Line {i}: {line_text}")
        
        log("=== EXTRACTION ===")
        extractor = ReceiptExtractor()
        result = extractor.extract_all(ocr_results, text_lines)
        for k, v in result.items():
            log(f"  {k}: {v}")
    else:
        log("NO OCR RESULTS - Text detection failed!")
        
    log("DONE")
    
except Exception as e:
    import traceback
    log(f"ERROR: {e}")
    log(traceback.format_exc())
finally:
    output_file.close()
    print("Results written to test_result.log")
