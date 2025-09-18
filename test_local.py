#!/usr/bin/env python3

import os
import sys
import cv2
import numpy as np

# Add paths to find the modules
sys.path.insert(0, os.path.join(os.getcwd(), 'streamlit'))
sys.path.insert(0, os.path.join(os.getcwd(), 'text_recognition'))
sys.path.insert(0, os.path.join(os.getcwd(), 'table_recognition'))

from classes import read_tables_from_list, read_texts_from_list
from utility import merge_text_table
from text_recognition import TextRecognizer
from table_recognition import TableRecognizer

def test_sample_image():
    """Test the functionality with the sample image."""
    
    # Load the sample image
    image_path = "sample.jpg"
    if not os.path.exists(image_path):
        print(f"Sample image not found: {image_path}")
        return
    
    print("Loading sample image...")
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image")
        return
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"Image loaded successfully. Shape: {image.shape}")
    
    try:
        # Test Table Recognition
        print("\n🤖 Testing Table Recognition...")
        table_recognizer = TableRecognizer.get_unique_instance()
        tables = table_recognizer.process(image, [])
        print(f"Found {len(tables)} table(s)")
        
        for i, table in enumerate(tables):
            print(f"  Table {i+1}: ({table.xmin}, {table.ymin}) to ({table.xmax}, {table.ymax})")
            print(f"    Contains {len(table.cells)} cells")
        
        # Test Text Recognition
        print("\n🤖 Testing Text Recognition...")
        text_recognizer = TextRecognizer.get_unique_instance()
        texts = text_recognizer.process(image, [])
        print(f"Found {len(texts)} text region(s)")
        
        for i, text in enumerate(texts[:5]):  # Show only first 5 texts
            print(f"  Text {i+1}: '{text.ocr}' at ({text.xmin}, {text.ymin})")
        
        if len(texts) > 5:
            print(f"  ... and {len(texts) - 5} more text regions")
        
        # Convert to expected format and merge
        print("\n🔄 Testing data integration...")
        
        table_dicts = []
        for table in tables:
            table_dict = {
                "name": "table",
                "xmin": table.xmin,
                "ymin": table.ymin, 
                "xmax": table.xmax,
                "ymax": table.ymax,
                "cells": []
            }
            for cell in table.cells:
                cell_dict = {
                    "xmin": cell.xmin,
                    "ymin": cell.ymin,
                    "xmax": cell.xmax,
                    "ymax": cell.ymax
                }
                table_dict["cells"].append(cell_dict)
            table_dicts.append(table_dict)

        text_dicts = []
        for text in texts:
            text_dict = {
                "name": "text",
                "xmin": text.xmin,
                "ymin": text.ymin,
                "xmax": text.xmax,
                "ymax": text.ymax,
                "ocr": text.ocr
            }
            text_dicts.append(text_dict)

        # Convert back to proper objects
        tables = read_tables_from_list(table_dicts)
        texts = read_texts_from_list(text_dicts)
        
        # Merge text with tables
        merge_text_table(tables, texts)
        
        print("✅ Data integration successful!")
        
        # Show results
        for i, table in enumerate(tables):
            print(f"\nTable {i+1} after text integration:")
            table.indexing()
            for j, cell in enumerate(table.cells[:3]):  # Show only first 3 cells
                print(f"  Cell {j+1}: '{cell.ocr}' at ({cell.xmin}, {cell.ymin})")
            if len(table.cells) > 3:
                print(f"  ... and {len(table.cells) - 3} more cells")
        
        print(f"\n✅ Test completed successfully!")
        print(f"📊 Summary:")
        print(f"   - {len(tables)} table(s) detected")
        print(f"   - {len(texts)} text region(s) detected")
        print(f"   - {sum(len(t.cells) for t in tables)} total cells found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing GO5-Project Local Setup")
    print("=" * 50)
    
    success = test_sample_image()
    
    if success:
        print("\n🎉 All tests passed! The application is ready to use.")
        print("\nTo run the Streamlit app:")
        print("   streamlit run local_app.py")
        print("\nThen open your browser to: http://localhost:8501")
    else:
        print("\n❌ Tests failed. Please check the error messages above.")
