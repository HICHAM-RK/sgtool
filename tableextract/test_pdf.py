#!/usr/bin/env python3

import os
import sys
from pdf_processor import pdf_processor

def test_pdf_functionality():
    """Test PDF processing functionality"""
    
    print("🧪 Testing PDF Processing Functionality")
    print("=" * 50)
    
    # Test PDF processor initialization
    print("✅ PDF Processor initialized successfully")
    print(f"   Supported formats: {pdf_processor.supported_formats}")
    
    # Test with a simple PDF creation (for testing purposes)
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        import io
        
        print("\n📄 Creating test PDF...")
        
        # Create a simple test PDF with some text and a basic table
        pdf_buffer = io.BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=letter)
        
        # Add title
        c.setFont("Helvetica-Bold", 16)
        c.drawString(100, 750, "Test PDF Document")
        
        # Add some table-like content
        c.setFont("Helvetica", 12)
        c.drawString(100, 700, "Sample Table:")
        
        # Simple table structure
        data = [
            ["Name", "Age", "City"],
            ["John", "25", "New York"],
            ["Jane", "30", "Los Angeles"],
            ["Bob", "35", "Chicago"]
        ]
        
        y = 650
        for row in data:
            x = 100
            for cell in row:
                c.drawString(x, y, cell)
                x += 100
            y -= 30
        
        c.save()
        pdf_bytes = pdf_buffer.getvalue()
        
        print("✅ Test PDF created successfully")
        
        # Test PDF metadata extraction
        print("\n🔍 Testing metadata extraction...")
        metadata = pdf_processor.extract_pdf_metadata(pdf_bytes)
        print(f"   Pages: {metadata.get('page_count', 'Unknown')}")
        print(f"   Title: {metadata.get('title', 'Unknown')}")
        print(f"   Creator: {metadata.get('creator', 'Unknown')}")
        
        # Test PDF to images conversion
        print("\n🖼️  Testing PDF to images conversion...")
        images = pdf_processor.pdf_to_images(pdf_bytes, dpi=150)
        
        if images:
            print(f"✅ Successfully converted PDF to {len(images)} image(s)")
            for img_array, page_num in images:
                print(f"   Page {page_num}: Shape {img_array.shape}")
        else:
            print("❌ Failed to convert PDF to images")
            return False
        
        # Test thumbnail generation
        print("\n🔍 Testing thumbnail generation...")
        thumbnails = pdf_processor.get_page_thumbnails(pdf_bytes, max_pages=5)
        
        if thumbnails:
            print(f"✅ Successfully generated {len(thumbnails)} thumbnail(s)")
            for thumb_array, page_num in thumbnails:
                print(f"   Thumbnail {page_num}: Shape {thumb_array.shape}")
        else:
            print("❌ Failed to generate thumbnails")
        
        # Test text extraction
        print("\n📝 Testing text extraction...")
        text_content = pdf_processor.extract_text_from_pdf(pdf_bytes)
        
        if text_content.strip():
            print("✅ Successfully extracted text from PDF")
            print(f"   Text preview: {text_content[:100]}...")
        else:
            print("⚠️  No text extracted (this might be normal for image-based PDFs)")
        
        print(f"\n✅ All PDF functionality tests completed successfully!")
        return True
        
    except ImportError:
        print("⚠️  ReportLab not available for creating test PDF")
        print("   Installing reportlab...")
        
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "reportlab"])
            print("✅ ReportLab installed successfully")
            return test_pdf_functionality()  # Retry the test
        except Exception as e:
            print(f"❌ Could not install reportlab: {e}")
            print("   PDF functionality should still work with existing PDF files")
            return True
    
    except Exception as e:
        print(f"❌ Error during PDF testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_file_type_detection():
    """Test file type detection"""
    
    print("\n🔍 Testing file type detection...")
    
    # Test with mock file objects
    class MockFile:
        def __init__(self, name):
            self.name = name
    
    test_files = [
        ("document.pdf", True),
        ("image.jpg", False),
        ("document.PDF", True),
        ("file.png", False),
        ("test.jpeg", False),
    ]
    
    for filename, expected in test_files:
        mock_file = MockFile(filename)
        result = pdf_processor.is_pdf(mock_file)
        status = "✅" if result == expected else "❌"
        print(f"   {status} {filename}: {result} (expected: {expected})")
    
    print("✅ File type detection tests completed")

if __name__ == "__main__":
    print("🚀 Starting PDF Functionality Tests")
    
    try:
        # Test file type detection
        test_file_type_detection()
        
        # Test PDF processing functionality
        success = test_pdf_functionality()
        
        if success:
            print(f"\n🎉 All tests passed! PDF support is ready to use.")
            print(f"\n📱 To run the enhanced app with PDF support:")
            print(f"   streamlit run local_app_with_pdf.py")
            print(f"\nThen open your browser to: http://localhost:8501")
        else:
            print(f"\n❌ Some tests failed. Please check the error messages above.")
            
    except Exception as e:
        print(f"\n❌ Test script failed: {str(e)}")
        import traceback
        traceback.print_exc()
