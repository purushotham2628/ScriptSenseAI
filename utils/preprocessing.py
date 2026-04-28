"""
Image Preprocessing Module
Handles OCR-oriented preprocessing for noisy, low-contrast inscriptions.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class ImagePreprocessor:
    """Preprocesses images for ancient script analysis"""
    
    def __init__(self):
        """Initialize the image preprocessor"""
        self.original_image = None
        self.processed_image = None
        self.ocr_image = None
    
    def load_image(self, image_path: Optional[str] = None, image_bytes: Optional[bytes] = None) -> np.ndarray:
        """
        Load image from file or bytes
        
        Args:
            image_path: Path to image file
            image_bytes: Image as bytes (for uploaded files)
            
        Returns:
            Loaded image as numpy array
        """
        if image_path:
            # Load from file path
            self.original_image = cv2.imread(image_path)
        elif image_bytes:
            # Load from bytes
            nparr = np.frombuffer(image_bytes, np.uint8)
            self.original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if self.original_image is None:
            raise ValueError("Failed to load image")
        
        return self.original_image
    
    def convert_to_grayscale(self, image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Convert image to grayscale
        
        Args:
            image: Input image (uses self.original_image if None)
            
        Returns:
            Grayscale image
        """
        if image is None:
            image = self.original_image
        
        if len(image.shape) == 2:
            return image

        if image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray
    
    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        """
        Remove light noise from image using Gaussian blur.
        
        Args:
            image: Input image
            
        Returns:
            Denoised image
        """
        return cv2.GaussianBlur(image, (5, 5), 0)

    def normalize_brightness(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize brightness while preserving local inscription detail.

        Args:
            image: Input grayscale image

        Returns:
            Brightness-normalized grayscale image
        """
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
        Improves contrast for better text visibility
        
        Args:
            image: Input grayscale image
            
        Returns:
            CLAHE enhanced image
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        return enhanced
    
    def apply_thresholding(self, image: np.ndarray, method: str = 'otsu') -> np.ndarray:
        """
        Apply thresholding to binarize text
        
        Args:
            image: Input grayscale image
            method: 'otsu', 'adaptive', or 'binary'
            
        Returns:
            Thresholded binary image
        """
        if method == 'otsu':
            # Otsu's automatic thresholding
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        elif method == 'adaptive':
            # Adaptive thresholding handles stone texture and uneven lighting
            # without turning inscriptions into edge outlines.
            binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 31, 9)
        else:
            # Simple binary threshold
            _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
        
        return binary
    
    def apply_morphological_operations(self, image: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to clean up binary image
        
        Args:
            image: Binary image
            
        Returns:
            Processed image
        """
        # Create kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
        # Apply morphological opening (removes small noise)
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Apply morphological closing (fills small holes)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        return closed
    
    def resize_image(self, image: np.ndarray, width: int = 1280) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio
        
        Args:
            image: Input image
            width: Target width
            
        Returns:
            Resized image
        """
        height = int((width / image.shape[1]) * image.shape[0])
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        return resized

    def upscale_image(self, image: np.ndarray, scale: float = 2.0) -> np.ndarray:
        """
        Upscale image for OCR while preserving character contours.

        Args:
            image: Input image
            scale: Scale factor

        Returns:
            Upscaled image
        """
        if scale == 1.0:
            return image

        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    def preprocess_complete_pipeline(self, image_path: Optional[str] = None, 
                                    image_bytes: Optional[bytes] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete preprocessing pipeline
        
        Args:
            image_path: Path to image file
            image_bytes: Image bytes
            
        Returns:
            Tuple of (original_image, processed_image)
        """
        # Load image
        self.load_image(image_path, image_bytes)
        
        # Resize for consistent processing, then upscale for better OCR.
        resized = self.resize_image(self.original_image)
        upscaled = self.upscale_image(resized, scale=2.0)
        
        # Convert to grayscale.
        print("[preprocess] grayscale")
        gray = self.convert_to_grayscale(upscaled)

        print("[preprocess] brightness normalization")
        normalized = self.normalize_brightness(gray)

        print("[preprocess] CLAHE contrast enhancement")
        enhanced = self.apply_clahe(normalized)
        self.ocr_image = enhanced
        
        # Light denoising only; aggressive filters can erase shallow glyphs.
        print("[preprocess] light Gaussian blur")
        blurred = self.denoise_image(enhanced)
        
        # Binarize with adaptive thresholding for low-contrast inscriptions.
        print("[preprocess] adaptive Gaussian threshold")
        processed = self.apply_thresholding(blurred, method='adaptive')
        
        self.processed_image = processed
        
        return upscaled, processed
    
    def save_image(self, image: np.ndarray, output_path: str) -> None:
        """
        Save image to file
        
        Args:
            image: Image to save
            output_path: Output file path
        """
        cv2.imwrite(output_path, image)
    
    def get_image_stats(self, image: np.ndarray) -> dict:
        """
        Get image statistics for analysis
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with image stats
        """
        stats = {
            'shape': image.shape,
            'dtype': str(image.dtype),
            'min': float(np.min(image)),
            'max': float(np.max(image)),
            'mean': float(np.mean(image)),
            'std': float(np.std(image))
        }
        return stats
