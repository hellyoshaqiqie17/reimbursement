"""
Image Preprocessing Module for Receipt OCR.
Handles common receipt issues: low light, crumpled paper, shadows, and skewed images.
"""
import cv2
import numpy as np
from typing import Tuple, Optional
import logging

from app.config import ImageConfig

logger = logging.getLogger(__name__)


class ReceiptPreprocessor:
    """
    Preprocesses receipt images for optimal OCR performance.
    
    Pipeline:
    1. Load and validate image
    2. Convert to grayscale
    3. Deskew (perspective transformation)
    4. Denoise and enhance contrast
    """
    
    def __init__(self):
        self.config = ImageConfig()
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Apply full preprocessing pipeline to an image.
        
        Args:
            image: Input image as numpy array (BGR format from cv2.imread)
            
        Returns:
            Preprocessed image ready for OCR
        """
        logger.info("Starting image preprocessing pipeline")
        
        # Step 1: Validate and resize if needed
        image = self._validate_and_resize(image)
        
        # Step 2: Convert to grayscale
        gray = self._to_grayscale(image)
        
        # Step 3: Deskew the image
        deskewed = self._deskew(gray)
        
        # Step 4: Denoise
        denoised = self._denoise(deskewed)
        
        # Step 5: Enhance contrast
        enhanced = self._enhance_contrast(denoised)
        
        # Step 6: Convert back to 3-channel for PaddleOCR compatibility
        # PaddleOCR expects 3-channel images
        if len(enhanced.shape) == 2:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        logger.info("Image preprocessing completed")
        return enhanced
    
    def _validate_and_resize(self, image: np.ndarray) -> np.ndarray:
        """Validate image dimensions and resize if too large."""
        if image is None or image.size == 0:
            raise ValueError("Invalid image: empty or None")
        
        height, width = image.shape[:2]
        
        # Check if resize is needed
        if width > self.config.MAX_WIDTH or height > self.config.MAX_HEIGHT:
            scale = min(
                self.config.MAX_WIDTH / width,
                self.config.MAX_HEIGHT / height
            )
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            logger.info(f"Image resized from {width}x{height} to {new_width}x{new_height}")
        
        return image
    
    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale if needed."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """
        Deskew the image using Hough Line Transform.
        Detects the dominant angle of text lines and rotates to correct.
        """
        # Apply edge detection
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        
        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=100,
            maxLineGap=10
        )
        
        if lines is None or len(lines) == 0:
            logger.info("No lines detected for deskewing, returning original")
            return image
        
        # Calculate angles of detected lines
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 != 0:  # Avoid division by zero
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                # Only consider near-horizontal lines (within ±45°)
                if -45 < angle < 45:
                    angles.append(angle)
        
        if not angles:
            logger.info("No suitable angles found for deskewing")
            return image
        
        # Use median angle to avoid outliers
        median_angle = np.median(angles)
        
        # Only correct if the skew is significant (> 0.5°)
        if abs(median_angle) < 0.5:
            logger.info(f"Skew angle {median_angle:.2f}° is minimal, skipping rotation")
            return image
        
        logger.info(f"Detected skew angle: {median_angle:.2f}°, correcting...")
        
        # Rotate the image to correct skew
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        
        # Calculate new image bounds to avoid clipping
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))
        
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (new_width, new_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated
    
    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply denoising to reduce noise from thermal/dot matrix printing."""
        return cv2.fastNlMeansDenoising(
            image,
            h=self.config.DENOISE_STRENGTH,
            templateWindowSize=7,
            searchWindowSize=21
        )
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
        This helps with receipts that have faded text or uneven lighting.
        """
        clahe = cv2.createCLAHE(
            clipLimit=self.config.CLAHE_CLIP_LIMIT,
            tileGridSize=self.config.CLAHE_TILE_SIZE
        )
        return clahe.apply(image)
    
    def apply_adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """
        Optional: Apply adaptive thresholding to create binary image.
        Useful for very low contrast receipts.
        """
        return cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2
        )


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from file path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Image as numpy array in BGR format
        
    Raises:
        ValueError: If image cannot be loaded or format not supported
    """
    import os
    
    # Check file exists
    if not os.path.exists(image_path):
        raise ValueError(f"Image file not found: {image_path}")
    
    # Check file extension
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in ImageConfig.SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported image format: {ext}. Supported: {ImageConfig.SUPPORTED_FORMATS}")
    
    # Load image
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    return image


def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Load an image from bytes (for API uploads).
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        Image as numpy array in BGR format
        
    Raises:
        ValueError: If image cannot be decoded
    """
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    
    # Decode image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Failed to decode image from bytes")
    
    return image
