"""
Fallback Text Recognition Module
===============================

This module provides a fallback text recognition solution using PyTesseract 
when EasyOCR is not available or fails due to network restrictions.

PyTesseract requires Tesseract OCR to be installed on the system, which is 
often available on corporate laptops or can be installed offline.
"""

import os
import cv2
import numpy as np
from typing import List
from pydantic import BaseModel


class Box(BaseModel):
    name: str = "box"
    xmin: int
    xmax: int
    ymin: int
    ymax: int

    @property
    def width(self):
        return max(self.xmax - self.xmin, 0)

    @property
    def height(self):
        return max(self.ymax - self.ymin, 0)


class Text(Box):
    name: str = "text"
    ocr: str = ""


class FallbackTextRecognizer:
    """
    Fallback text recognizer using PyTesseract when EasyOCR is not available.
    """
    instance = None

    def __init__(self):
        self.ocr_engine = self._initialize_ocr_engine()
    
    def _initialize_ocr_engine(self):
        """Initialize the best available OCR engine."""
        # Try EasyOCR first
        try:
            import easyocr
            model_storage_directory = os.path.expanduser("~/.EasyOCR/model")
            os.makedirs(model_storage_directory, exist_ok=True)
            
            reader = easyocr.Reader(
                lang_list=["en"], 
                download_enabled=False,
                model_storage_directory=model_storage_directory
            )
            print("✅ EasyOCR initialized successfully (offline mode)")
            return ("easyocr", reader)
            
        except Exception as e:
            print(f"⚠️ EasyOCR failed: {e}")
        
        # Fallback to PyTesseract
        try:
            import pytesseract
            from PIL import Image
            
            # Test if tesseract is available
            pytesseract.get_tesseract_version()
            print("✅ PyTesseract initialized successfully")
            return ("tesseract", pytesseract)
            
        except Exception as e:
            print(f"⚠️ PyTesseract failed: {e}")
        
        # Last resort: return a dummy engine
        print("❌ No OCR engine available. Text extraction will be limited.")
        return ("dummy", None)

    def process(self, image, text_list: list = None):
        """Process image and extract text regions."""
        if text_list is None:
            text_list = []
            
        engine_type, engine = self.ocr_engine
        
        if engine_type == "easyocr":
            return self._process_with_easyocr(image, engine)
        elif engine_type == "tesseract":
            return self._process_with_tesseract(image, engine)
        else:
            return self._process_with_dummy(image)
    
    def _process_with_easyocr(self, image, reader):
        """Process using EasyOCR."""
        texts = reader.readtext(image, min_size=1, text_threshold=0.3)
        
        output = []
        for (location, ocr_text, confidence) in texts:
            if confidence > 0.2:  # Filter low-confidence detections
                xmin = int(min(p[0] for p in location))
                xmax = int(max(p[0] for p in location))
                ymin = int(min(p[1] for p in location))
                ymax = int(max(p[1] for p in location))
                
                new_text = Text(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, ocr=ocr_text)
                output.append(new_text)
        
        return output
    
    def _process_with_tesseract(self, image, pytesseract):
        """Process using PyTesseract."""
        try:
            from PIL import Image
            
            # Convert OpenCV image to PIL
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
            
            # Get bounding box data
            data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
            
            output = []
            n_boxes = len(data['level'])
            
            for i in range(n_boxes):
                confidence = int(data['conf'][i])
                text = data['text'][i].strip()
                
                if confidence > 30 and text:  # Filter low confidence and empty text
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    
                    new_text = Text(
                        xmin=x, 
                        xmax=x + w, 
                        ymin=y, 
                        ymax=y + h, 
                        ocr=text
                    )
                    output.append(new_text)
            
            return output
            
        except Exception as e:
            print(f"PyTesseract processing failed: {e}")
            return []
    
    def _process_with_dummy(self, image):
        """Dummy processor that returns empty results."""
        print("⚠️ No OCR engine available - returning empty results")
        return []
    
    @classmethod
    def get_unique_instance(cls):
        if cls.instance is None:
            cls.instance = cls()
        return cls.instance


def draw(image, text_list: List[Text]):
    """Draw text bounding boxes on image."""
    for text in text_list:
        cv2.rectangle(
            image, (text.xmin, text.ymin), (text.xmax, text.ymax), (255, 0, 0), 4
        )
    return image


# For compatibility with existing code
TextRecognizer = FallbackTextRecognizer


if __name__ == "__main__":
    # Test the fallback recognizer
    print("🔍 Testing Fallback Text Recognizer")
    
    recognizer = FallbackTextRecognizer()
    
    # Create a dummy image with text for testing
    test_image = np.ones((200, 400, 3), dtype=np.uint8) * 255
    cv2.putText(test_image, "Test Text", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    results = recognizer.process(test_image)
    print(f"Found {len(results)} text regions")
    
    for i, text in enumerate(results):
        print(f"Text {i+1}: '{text.ocr}' at ({text.xmin}, {text.ymin}, {text.xmax}, {text.ymax})")