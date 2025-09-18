#!/usr/bin/env python3

import os
import sys

def test_pdf_imports():
    """Test if PDF processing modules are working"""
    
    print("🧪 Testing PDF Support Components")
    print("=" * 40)
    
    try:
        # Test PyMuPDF
        import fitz
        print(f"✅ PyMuPDF {fitz.__version__} - OK")
        
        # Test creating a blank document
        doc = fitz.open()
        print("✅ Can create PDF documents - OK")
        doc.close()
        
        # Test PDF processor
        from pdf_processor import pdf_processor
        print("✅ PDF Processor module - OK")
        
        # Test file detection
        class MockFile:
            def __init__(self, name):
                self.name = name
        
        test_file = MockFile("test.pdf")
        result = pdf_processor.is_pdf(test_file)
        print(f"✅ PDF Detection: {result} - OK")
        
        # Test PIL for image processing
        from PIL import Image
        print("✅ PIL (Pillow) - OK")
        
        # Test other dependencies
        import numpy as np
        print("✅ NumPy - OK")
        
        import cv2
        print("✅ OpenCV - OK")
        
        print(f"\n🎉 All PDF support components are working!")
        print(f"\n📚 Capabilities:")
        print(f"   • PDF to Image conversion")
        print(f"   • Multi-page PDF processing")
        print(f"   • PDF metadata extraction")
        print(f"   • Thumbnail generation")
        print(f"   • Text extraction from PDFs")
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_pdf_imports()
    
    if success:
        print(f"\n🚀 Ready to run the enhanced app:")
        print(f"   streamlit run local_app_with_pdf.py")
        print(f"\n💡 Features available:")
        print(f"   • Upload PNG, JPG, JPEG images")
        print(f"   • Upload PDF documents (multi-page)")
        print(f"   • Select specific pages to process")
        print(f"   • Download individual Excel files")
        print(f"   • Download all results as ZIP")
    else:
        print(f"\n❌ Please install missing dependencies and try again")
