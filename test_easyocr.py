#!/usr/bin/env python3
"""
EasyOCR Diagnostic Test
======================

This script tests EasyOCR initialization to diagnose the exact issue
causing the "failed to initialize easyocr" error.
"""

import os
import sys

def test_easyocr_initialization():
    """Test EasyOCR initialization with detailed error reporting."""
    print("🔍 EasyOCR Initialization Diagnostic")
    print("=" * 50)
    
    # Check Python environment
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Check model directory
    model_dir = os.path.expanduser("~/.EasyOCR/model")
    print(f"Model directory: {model_dir}")
    print(f"Model directory exists: {os.path.exists(model_dir)}")
    
    if os.path.exists(model_dir):
        files = os.listdir(model_dir)
        print(f"Files in model directory: {files}")
        
        # Check specific models
        craft_model = os.path.join(model_dir, "craft_mlt_25k.pth")
        english_model = os.path.join(model_dir, "english_g2.pth")
        
        print(f"CRAFT model exists: {os.path.exists(craft_model)}")
        print(f"English model exists: {os.path.exists(english_model)}")
        
        if os.path.exists(craft_model):
            print(f"CRAFT model size: {os.path.getsize(craft_model)} bytes")
        if os.path.exists(english_model):
            print(f"English model size: {os.path.getsize(english_model)} bytes")
    
    print("\n🧪 Testing EasyOCR import...")
    try:
        import easyocr
        print("✅ EasyOCR module imported successfully")
        print(f"EasyOCR version: {easyocr.__version__ if hasattr(easyocr, '__version__') else 'Unknown'}")
    except Exception as e:
        print(f"❌ Failed to import EasyOCR: {e}")
        return False
    
    print("\n🧪 Testing EasyOCR Reader initialization...")
    try:
        reader = easyocr.Reader(
            lang_list=["en"],
            download_enabled=False,
            model_storage_directory=model_dir,
            gpu=False,
            verbose=False
        )
        print("✅ EasyOCR Reader initialized successfully!")
        
        # Test a simple operation
        import numpy as np
        test_image = np.ones((100, 200, 3), dtype=np.uint8) * 255
        result = reader.readtext(test_image)
        print(f"✅ Test operation successful. Found {len(result)} text regions (expected 0 for blank image)")
        
        return True
        
    except Exception as e:
        print(f"❌ EasyOCR Reader initialization failed: {e}")
        print(f"Error type: {type(e).__name__}")
        
        # More detailed error analysis
        import traceback
        print("\n📋 Full error traceback:")
        traceback.print_exc()
        
        return False

def test_fallback_options():
    """Test fallback OCR options."""
    print("\n🔄 Testing fallback options...")
    
    # Test PyTesseract
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        print(f"✅ PyTesseract available (Tesseract {version})")
        return True
    except Exception as e:
        print(f"❌ PyTesseract not available: {e}")
        return False

def main():
    """Run all diagnostic tests."""
    print("🚀 Starting EasyOCR Diagnostic Tests\n")
    
    easyocr_success = test_easyocr_initialization()
    fallback_available = test_fallback_options()
    
    print("\n" + "=" * 50)
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 50)
    
    if easyocr_success:
        print("✅ EasyOCR is working correctly!")
        print("🎉 Your table extraction should work without issues.")
    elif fallback_available:
        print("⚠️ EasyOCR failed, but PyTesseract is available as fallback")
        print("📝 The app will use PyTesseract for text recognition")
    else:
        print("❌ Both EasyOCR and PyTesseract failed")
        print("💡 Recommendations:")
        print("   1. Run: python download_easyocr_models.py")
        print("   2. Install Tesseract OCR as fallback")
        print("   3. Check your Python environment and dependencies")
    
    print(f"\n📁 Model directory: {os.path.expanduser('~/.EasyOCR/model')}")
    print("🔍 Check this directory to verify model files exist")

if __name__ == "__main__":
    main()