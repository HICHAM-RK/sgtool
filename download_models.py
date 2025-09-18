import os
import sys

# Add the tableextract subdirectory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tableextract'))

def download_models():
    """Pre-download models during deployment"""
    print("Starting model download...")
    
    try:
        # Import and initialize text recognition
        from text_recognition import TextRecognizer
        print("Downloading text recognition models...")
        text_recognizer = TextRecognizer.get_unique_instance()
        print("Text recognition models loaded successfully!")
        
        # Import and initialize table recognition  
        from table_recognition import TableRecognizer
        print("Downloading table recognition models...")
        table_recognizer = TableRecognizer.get_unique_instance()
        print("Table recognition models loaded successfully!")
        
        print("All models downloaded successfully!")
        
    except Exception as e:
        print(f"Error downloading models: {e}")
        print("Models will be downloaded on first use instead.")

if __name__ == "__main__":
    download_models()