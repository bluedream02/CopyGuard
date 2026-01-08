#!/usr/bin/env python3
"""
Extract image datasets from compressed archives.
Run this script once after cloning the repository.

Usage:
    cd dataset
    python extract_images.py
"""

import os
import zipfile
from pathlib import Path


def extract_archive(zip_path, extract_to=None):
    """
    Extract a zip archive.
    
    Args:
        zip_path: Path to zip file
        extract_to: Directory to extract to (default: same as zip name)
    """
    zip_path = Path(zip_path)
    
    if not zip_path.exists():
        print(f"❌ Error: {zip_path.name} not found")
        return False
    
    # Determine extraction directory
    if extract_to is None:
        extract_to = zip_path.stem  # Remove .zip extension
    
    extract_to = Path(extract_to)
    
    # Check if already extracted
    if extract_to.exists():
        print(f"⚠️  {extract_to.name}/ already exists. Skipping...")
        return True
    
    print(f"📦 Extracting {zip_path.name}...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('.')
        
        # Count extracted files
        file_count = sum(1 for _ in extract_to.rglob('*') if _.is_file())
        print(f"   ✅ Extracted successfully")
        print(f"   📊 {file_count} files extracted")
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to extract: {e}")
        return False


def main():
    """Main extraction process."""
    print("=" * 60)
    print("  Extracting Image Datasets")
    print("=" * 60)
    print()
    
    # Check if we're in the dataset directory
    if not Path("book_copyright_images.zip").exists():
        print("❌ Error: Please run this script from the dataset/ directory")
        print("   cd dataset && python extract_images.py")
        return 1
    
    # List of archives to extract
    archives = [
        "book_copyright_images.zip",
        "code_copyright_images.zip",
        "lyrics_copyright_images.zip",
        "news_copyright_images.zip"
    ]
    
    # Extract all archives
    results = []
    for archive in archives:
        success = extract_archive(archive)
        results.append((archive, success))
        print()
    
    # Verify extraction
    print("=" * 60)
    print("  Verification")
    print("=" * 60)
    print()
    
    all_present = True
    expected_dirs = [
        "book_copyright_images",
        "code_copyright_images",
        "lyrics_copyright_images",
        "news_copyright_images"
    ]
    
    for dirname in expected_dirs:
        if Path(dirname).exists():
            print(f"✅ {dirname}/")
        else:
            print(f"❌ {dirname}/ (missing)")
            all_present = False
    
    print()
    
    if all_present:
        print("🎉 All image datasets extracted successfully!")
        print()
        print("You can now run the evaluation scripts.")
        return 0
    else:
        print("⚠️  Some datasets failed to extract. Please check the errors above.")
        return 1


if __name__ == "__main__":
    exit(main())

