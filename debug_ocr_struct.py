"""
Get all keys and values from OCRResult
"""
import sys
sys.path.insert(0, '.')

import cv2

output_file = open("ocr_debug3.log", "w", encoding="utf-8")
def log(msg):
    output_file.write(str(msg) + "\n")
    output_file.flush()

try:
    from app.preprocessing.preprocessor import ReceiptPreprocessor
    from paddleocr import PaddleOCR

    image_path = r"c:\reimbursement\untuk_test\Screenshot 2026-01-11 181638.png"

    image = cv2.imread(image_path)
    preprocessor = ReceiptPreprocessor()
    processed = preprocessor.process(image)

    log("Running PaddleOCR...")
    ocr = PaddleOCR(lang="en")
    result = ocr.ocr(processed)
    
    ocr_result = result[0]
    
    log("All keys in OCRResult:")
    for key in ocr_result.keys():
        log(f"  {key}")
    
    log("\nAll key-value pairs:")
    for key, value in ocr_result.items():
        val_type = type(value).__name__
        if hasattr(value, '__len__') and not isinstance(value, str):
            log(f"  {key}: [{val_type}, len={len(value)}]")
            if len(value) > 0 and len(value) <= 30:
                for i, item in enumerate(value):
                    item_str = str(item)
                    if len(item_str) > 100:
                        item_str = item_str[:100] + "..."
                    log(f"    [{i}]: {item_str}")
        else:
            val_str = str(value)
            if len(val_str) > 100:
                val_str = val_str[:100] + "..."
            log(f"  {key}: {val_str}")
        
    log("DONE")
    
except Exception as e:
    import traceback
    log(f"ERROR: {e}")
    log(traceback.format_exc())
finally:
    output_file.close()
    print("Results written to ocr_debug3.log")
