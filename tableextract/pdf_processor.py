try:
    import fitz  # PyMuPDF
except ImportError:
    try:
        import PyMuPDF as fitz
    except ImportError:
        print("Error: PyMuPDF not found. Please install with: pip install PyMuPDF")
        fitz = None
import cv2
import numpy as np
from typing import List, Tuple
import io
from PIL import Image
try:
    import streamlit as st
except ImportError:
    # Mock streamlit for testing
    class MockStreamlit:
        def error(self, msg):
            print(f"Error: {msg}")
    st = MockStreamlit()

class PDFProcessor:
    """
    Utility class for processing PDF documents and converting them to images
    for table extraction.
    """
    
    def __init__(self):
        self.supported_formats = ['.pdf']
    
    def is_pdf(self, file) -> bool:
        """Check if the uploaded file is a PDF"""
        if hasattr(file, 'name'):
            return file.name.lower().endswith('.pdf')
        return False
    
    def pdf_to_images(self, pdf_bytes: bytes, dpi: int = 200) -> List[Tuple[np.ndarray, int]]:
        """
        Convert PDF pages to images.
        
        Args:
            pdf_bytes: PDF file as bytes
            dpi: Resolution for conversion (default 200 DPI for good quality)
            
        Returns:
            List of tuples (image_array, page_number)
        """
        images = []
        
        try:
            # Open PDF from bytes
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            for page_num in range(len(pdf_document)):
                # Get the page
                page = pdf_document.load_page(page_num)
                
                # Create transformation matrix for the desired DPI
                mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
                
                # Render page to image
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                img_data = pix.tobytes("ppm")
                pil_img = Image.open(io.BytesIO(img_data))
                
                # Convert to numpy array (RGB)
                img_array = np.array(pil_img)
                
                images.append((img_array, page_num + 1))
            
            pdf_document.close()
            
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return []
        
        return images
    
    def extract_pdf_metadata(self, pdf_bytes: bytes) -> dict:
        """
        Extract metadata from PDF document.
        
        Args:
            pdf_bytes: PDF file as bytes
            
        Returns:
            Dictionary containing PDF metadata
        """
        try:
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            metadata = {
                'page_count': len(pdf_document),
                'title': pdf_document.metadata.get('title', 'Unknown'),
                'author': pdf_document.metadata.get('author', 'Unknown'),
                'creator': pdf_document.metadata.get('creator', 'Unknown'),
                'producer': pdf_document.metadata.get('producer', 'Unknown'),
                'creation_date': pdf_document.metadata.get('creationDate', 'Unknown'),
                'modification_date': pdf_document.metadata.get('modDate', 'Unknown'),
            }
            
            pdf_document.close()
            return metadata
            
        except Exception as e:
            st.error(f"Error extracting PDF metadata: {str(e)}")
            return {'page_count': 0, 'error': str(e)}
    
    def get_page_thumbnails(self, pdf_bytes: bytes, max_pages: int = 5, thumb_size: Tuple[int, int] = (200, 300)) -> List[Tuple[np.ndarray, int]]:
        """
        Generate thumbnail images for PDF pages for preview.
        
        Args:
            pdf_bytes: PDF file as bytes
            max_pages: Maximum number of pages to generate thumbnails for
            thumb_size: Size of thumbnails (width, height)
            
        Returns:
            List of tuples (thumbnail_array, page_number)
        """
        thumbnails = []
        
        try:
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            for page_num in range(min(len(pdf_document), max_pages)):
                page = pdf_document.load_page(page_num)
                
                # Create a smaller transformation matrix for thumbnails
                mat = fitz.Matrix(0.3, 0.3)  # 30% scale for thumbnails
                
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("ppm")
                pil_img = Image.open(io.BytesIO(img_data))
                
                # Resize to exact thumbnail size
                pil_img = pil_img.resize(thumb_size, Image.Resampling.LANCZOS)
                
                img_array = np.array(pil_img)
                thumbnails.append((img_array, page_num + 1))
            
            pdf_document.close()
            
        except Exception as e:
            st.error(f"Error generating thumbnails: {str(e)}")
            return []
        
        return thumbnails
    
    def extract_text_from_pdf(self, pdf_bytes: bytes, page_num: int = None) -> str:
        """
        Extract text content from PDF for additional context.
        
        Args:
            pdf_bytes: PDF file as bytes
            page_num: Specific page number to extract text from (1-based), None for all pages
            
        Returns:
            Extracted text content
        """
        try:
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            text_content = ""
            
            if page_num is not None:
                # Extract text from specific page
                if 1 <= page_num <= len(pdf_document):
                    page = pdf_document.load_page(page_num - 1)  # Convert to 0-based
                    text_content = page.get_text()
            else:
                # Extract text from all pages
                for page_num in range(len(pdf_document)):
                    page = pdf_document.load_page(page_num)
                    text_content += f"--- Page {page_num + 1} ---\n"
                    text_content += page.get_text() + "\n\n"
            
            pdf_document.close()
            return text_content
            
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return ""

# Global instance
pdf_processor = PDFProcessor()
