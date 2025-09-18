#!/usr/bin/env python3
"""
Manual EasyOCR Model Downloader
===============================

This script helps you manually download EasyOCR models when automatic 
downloading is blocked by corporate firewalls or network restrictions.

Usage:
    python download_easyocr_models.py

The script will:
1. Create the EasyOCR model directory
2. Download required models from GitHub
3. Place them in the correct location for offline use
"""

import os
import urllib.request
import urllib.error
from pathlib import Path


def download_file(url, destination, description=""):
    """Download a file with progress indication."""
    print(f"📥 Downloading {description}...")
    print(f"   From: {url}")
    print(f"   To: {destination}")
    
    try:
        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) // total_size)
                bar_length = 40
                filled_length = (bar_length * percent) // 100
                bar = '█' * filled_length + '-' * (bar_length - filled_length)
                print(f'\r   Progress: |{bar}| {percent}% Complete', end='', flush=True)
        
        urllib.request.urlretrieve(url, destination, progress_hook)
        print("\n   ✅ Download completed successfully!")
        return True
        
    except urllib.error.URLError as e:
        print(f"\n   ❌ Download failed: {e}")
        print("   💡 This might be due to network restrictions.")
        return False
    except Exception as e:
        print(f"\n   ❌ Unexpected error: {e}")
        return False


def main():
    print("🚀 EasyOCR Model Downloader")
    print("=" * 50)
    
    # Set up model directory
    model_dir = Path.home() / ".EasyOCR" / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Model directory: {model_dir}")
    
    # Define required models with their URLs
    models = {
        "craft_mlt_25k.pth": {
            "url": "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3.2/craft_mlt_25k.zip",
            "description": "Text Detection Model (CRAFT MLT 25K)",
            "is_zip": True
        },
        "english_g2.pth": {
            "url": "https://github.com/JaidedAI/EasyOCR/releases/download/v1.6.2/english_g2.zip", 
            "description": "English Text Recognition Model",
            "is_zip": True
        }
    }
    
    print("\n📋 Required Models:")
    for model_name, info in models.items():
        print(f"   • {model_name}: {info['description']}")
    
    print("\n🔍 Checking existing models...")
    
    downloaded_models = []
    failed_downloads = []
    
    for model_name, info in models.items():
        model_path = model_dir / model_name
        
        if model_path.exists():
            print(f"   ✅ {model_name} already exists")
            continue
            
        print(f"\n📥 Downloading {model_name}...")
        
        if info.get("is_zip", False):
            # For ZIP files, we need to download and extract
            zip_path = model_dir / f"{model_name}.zip"
            
            if download_file(info["url"], zip_path, info["description"]):
                # Extract the ZIP file
                try:
                    import zipfile
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(model_dir)
                    
                    # Remove the ZIP file after extraction
                    zip_path.unlink()
                    
                    # Check if the model file exists after extraction
                    if model_path.exists():
                        print(f"   ✅ {model_name} extracted successfully")
                        downloaded_models.append(model_name)
                    else:
                        print(f"   ⚠️ Model file not found after extraction")
                        failed_downloads.append(model_name)
                        
                except Exception as e:
                    print(f"   ❌ Failed to extract ZIP: {e}")
                    failed_downloads.append(model_name)
            else:
                failed_downloads.append(model_name)
        else:
            # Direct file download
            if download_file(info["url"], model_path, info["description"]):
                downloaded_models.append(model_name)
            else:
                failed_downloads.append(model_name)
    
    print("\n" + "=" * 50)
    print("📊 DOWNLOAD SUMMARY")
    print("=" * 50)
    
    if downloaded_models:
        print(f"✅ Successfully downloaded {len(downloaded_models)} model(s):")
        for model in downloaded_models:
            print(f"   • {model}")
    
    if failed_downloads:
        print(f"\n❌ Failed to download {len(failed_downloads)} model(s):")
        for model in failed_downloads:
            print(f"   • {model}")
    
    # Show alternative download instructions
    if failed_downloads:
        print("\n💡 ALTERNATIVE SOLUTION - Manual Download:")
        print("-" * 50)
        print("If the automatic download fails due to network restrictions,")
        print("you can manually download the models:")
        print()
        
        for model_name, info in models.items():
            if model_name in failed_downloads:
                print(f"1. Go to: {info['url']}")
                print(f"2. Download and extract to: {model_dir / model_name}")
                print()
        
        print("Alternative URLs (try these if GitHub is blocked):")
        print("• https://www.dropbox.com/s/q8ii4w3lqag1nfw/craft_mlt_25k.pth")
        print("• https://www.dropbox.com/s/ckb1lnzm1r3lv7t/english_g2.pth")
    
    print("\n🔧 NEXT STEPS:")
    print("-" * 50)
    if not failed_downloads:
        print("✅ All models downloaded successfully!")
        print("🚀 You can now run your table extraction tool offline.")
        print("💡 The app will use download_enabled=False automatically.")
    else:
        print("⚠️  Some models are missing. Your options:")
        print("   1. Try manual download using the URLs above")
        print("   2. Use a personal computer/network to download")
        print("   3. Contact IT to whitelist the EasyOCR repository")
    
    print(f"\n📁 Models location: {model_dir}")
    print("🔍 You can verify the files exist in this directory.")


if __name__ == "__main__":
    main()