"""
OCR Engine Module using PaddleOCR.
Optimized for receipt fonts including dot matrix and thermal print.
"""
# Disable OneDNN/MKLDNN to fix Cloud Run compatibility issues
import os
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'true'
os.environ['MKLDNN_CACHE_CAPACITY'] = '0'
os.environ['FLAGS_enable_pir_in_executor'] = '0'

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class OCRResult:
    """Represents a single OCR detection result."""
    
    def __init__(
        self,
        text: str,
        confidence: float,
        bbox,
        line_index: int = 0
    ):
        self.text = text
        self.confidence = confidence
        self._raw_bbox = bbox
        self.bbox = self._normalize_bbox(bbox)
        self.line_index = line_index
    
    def _normalize_bbox(self, bbox) -> List[List[float]]:
        """Normalize bbox to [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] format."""
        try:
            # Handle numpy array
            if isinstance(bbox, np.ndarray):
                bbox = bbox.tolist()
            
            if not bbox:
                return [[0, 0], [0, 0], [0, 0], [0, 0]]
            
            # Check if already in correct format (list of points)
            if isinstance(bbox, list) and len(bbox) >= 4:
                first = bbox[0]
                if isinstance(first, (list, tuple)) and len(first) >= 2:
                    # Already in [[x,y], ...] format
                    return [[float(p[0]), float(p[1])] for p in bbox[:4]]
            
            # Handle flat format [x1,y1,x2,y2,x3,y3,x4,y4]
            if isinstance(bbox, list) and len(bbox) >= 8:
                return [
                    [float(bbox[0]), float(bbox[1])],
                    [float(bbox[2]), float(bbox[3])],
                    [float(bbox[4]), float(bbox[5])],
                    [float(bbox[6]), float(bbox[7])]
                ]
            
            # Handle 4-element box [x_min, y_min, x_max, y_max]
            if isinstance(bbox, list) and len(bbox) == 4:
                x1, y1, x2, y2 = [float(v) for v in bbox]
                return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            
            logger.warning(f"Unknown bbox format: {type(bbox)}, value: {bbox}")
            return [[0, 0], [0, 0], [0, 0], [0, 0]]
            
        except Exception as e:
            logger.error(f"Error normalizing bbox: {e}")
            return [[0, 0], [0, 0], [0, 0], [0, 0]]
    
    @property
    def center_y(self) -> float:
        """Get the vertical center of the bounding box."""
        try:
            return sum(point[1] for point in self.bbox) / len(self.bbox)
        except (IndexError, TypeError, ZeroDivisionError):
            return 0.0
    
    @property
    def center_x(self) -> float:
        """Get the horizontal center of the bounding box."""
        try:
            return sum(point[0] for point in self.bbox) / len(self.bbox)
        except (IndexError, TypeError, ZeroDivisionError):
            return 0.0
    
    @property
    def left_x(self) -> float:
        """Get the leftmost x coordinate."""
        try:
            return min(point[0] for point in self.bbox)
        except (IndexError, TypeError, ValueError):
            return 0.0
    
    @property
    def right_x(self) -> float:
        """Get the rightmost x coordinate."""
        try:
            return max(point[0] for point in self.bbox)
        except (IndexError, TypeError, ValueError):
            return 0.0
    
    def __repr__(self) -> str:
        try:
            return f"OCRResult(text='{self.text}', confidence={self.confidence:.2f}, y={self.center_y:.0f})"
        except:
            return f"OCRResult(text='{self.text}', confidence={self.confidence:.2f})"


class ReceiptOCR:
    """
    OCR engine for receipt text extraction using PaddleOCR.
    
    Features:
    - Optimized for receipt fonts (dot matrix, thermal)
    - Returns text with position information for layout analysis
    - Organizes results by vertical position (reading order)
    """
    
    def __init__(
        self,
        lang: str = "en",
        use_gpu: bool = False,
        use_angle_cls: bool = True
    ):
        """
        Initialize PaddleOCR engine.
        
        Args:
            lang: Language code ("en" for English, "ch" for Chinese)
            use_gpu: Whether to use GPU acceleration
            use_angle_cls: Whether to use angle classification
        """
        self.lang = lang
        self.use_gpu = use_gpu
        self.use_angle_cls = use_angle_cls
        self._ocr = None
    
    def _get_ocr(self):
        """Lazy initialization of PaddleOCR to avoid slow import on startup."""
        if self._ocr is None:
            try:
                from paddleocr import PaddleOCR
                
                logger.info("Initializing PaddleOCR engine...")
                self._ocr = PaddleOCR(
                    lang=self.lang,
                )
                logger.info("PaddleOCR initialized successfully")
            except ImportError:
                logger.error("PaddleOCR not installed. Install with: pip install paddleocr paddlepaddle")
                raise ImportError("PaddleOCR is required. Install with: pip install paddleocr paddlepaddle")
        
        return self._ocr
    
    def extract_text(self, image: np.ndarray) -> List[OCRResult]:
        """
        Extract text from image with position information.
        
        Args:
            image: Preprocessed image as numpy array
            
        Returns:
            List of OCRResult objects sorted by vertical position
        """
        ocr = self._get_ocr()
        
        logger.info("Running OCR on image (no args)...")
        # Explicitly call without arguments as cls arg causes error
        result = ocr.ocr(image)
        
        if result is None or len(result) == 0 or result[0] is None:
            logger.warning("No text detected in image")
            return []
        
        # Parse PaddleOCR results
        ocr_results = []
        
        # Handle PaddleOCR v2.9+ / PaddleX result format
        # The result[0] can be a dict-like OCRResult object
        if len(result) > 0 and hasattr(result[0], 'keys'):
            data = result[0]
            # PaddleX uses plural key names: rec_texts, rec_scores, rec_polys
            boxes = data.get('rec_polys', data.get('dt_polys', []))
            texts = data.get('rec_texts', data.get('rec_text', []))
            scores = data.get('rec_scores', data.get('rec_score', []))
            
            logger.info(f"PaddleX format: {len(texts)} texts, {len(scores)} scores, {len(boxes)} boxes")
            
            for i in range(len(texts)):
                text = texts[i] if i < len(texts) else ""
                score = scores[i] if i < len(scores) else 0.0
                box = boxes[i] if i < len(boxes) else [[0,0],[0,0],[0,0],[0,0]]
                
                # Ensure box is list of points
                if isinstance(box, np.ndarray):
                    box = box.tolist()
                
                ocr_results.append(OCRResult(
                    text=str(text).strip(),
                    confidence=float(score),
                    bbox=box
                ))
                
        # Handle standard list-of-lists format
        elif len(result) > 0 and isinstance(result[0], list):
            for line in result[0]:
                try:
                    # Ensure line has at least 2 elements (bbox, text_info)
                    if len(line) < 2:
                        logger.warning(f"Skipping malformed line (len < 2): {line}")
                        continue
                        
                    bbox = line[0]
                    text_info = line[1]
                    
                    # Handle text_info structure
                    if isinstance(text_info, (list, tuple)):
                        if len(text_info) >= 2:
                            text = text_info[0]
                            confidence = text_info[1]
                        elif len(text_info) == 1:
                            text = text_info[0]
                            confidence = 0.0  # Default confidence
                        else:
                            logger.warning(f"Skipping empty text_info: {text_info}")
                            continue
                    else:
                        # Fallback if text_info is just a string
                        text = str(text_info)
                        confidence = 0.0
                
                    ocr_results.append(OCRResult(
                        text=text.strip(),
                        confidence=confidence,
                        bbox=bbox
                    ))
                except Exception as e:
                    logger.error(f"Error parsing line: {line}. Error: {e}")
                    continue
        
        # Sort by vertical position (top to bottom)
        ocr_results.sort(key=lambda r: r.center_y)
        
        # Assign line indices
        for i, result in enumerate(ocr_results):
            result.line_index = i
        
        logger.info(f"Extracted {len(ocr_results)} text regions")
        return ocr_results
    
    def get_text_lines(self, results: List[OCRResult], line_threshold: float = 20.0) -> List[List[OCRResult]]:
        """
        Group OCR results into logical lines based on vertical position.
        
        Args:
            results: List of OCRResult objects
            line_threshold: Maximum vertical distance to consider same line (pixels)
            
        Returns:
            List of lines, where each line is a list of OCRResults sorted left-to-right
        """
        if not results:
            return []
        
        lines: List[List[OCRResult]] = []
        current_line: List[OCRResult] = [results[0]]
        current_y = results[0].center_y
        
        for result in results[1:]:
            if abs(result.center_y - current_y) <= line_threshold:
                # Same line
                current_line.append(result)
            else:
                # New line - save current and start new
                current_line.sort(key=lambda r: r.left_x)  # Sort left-to-right
                lines.append(current_line)
                current_line = [result]
                current_y = result.center_y
        
        # Don't forget the last line
        if current_line:
            current_line.sort(key=lambda r: r.left_x)
            lines.append(current_line)
        
        return lines
    
    def get_full_text(self, results: List[OCRResult]) -> str:
        """
        Get all extracted text as a single string with line breaks.
        
        Args:
            results: List of OCRResult objects
            
        Returns:
            Full text with lines separated by newlines
        """
        lines = self.get_text_lines(results)
        text_lines = []
        
        for line in lines:
            line_text = " ".join(r.text for r in line)
            text_lines.append(line_text)
        
        return "\n".join(text_lines)
    
    def get_average_confidence(self, results: List[OCRResult]) -> float:
        """Calculate average confidence score across all results."""
        if not results:
            return 0.0
        return sum(r.confidence for r in results) / len(results)
