#!/bin/bash
# Extract all image datasets from compressed archives
# Run this script once after cloning the repository

echo "================================================"
echo "  Extracting Image Datasets"
echo "================================================"
echo ""

# Check if we're in the dataset directory
if [ ! -f "book_copyright_images.zip" ]; then
    echo "❌ Error: Please run this script from the dataset/ directory"
    echo "   cd dataset && bash extract_images.sh"
    exit 1
fi

# Function to extract and verify
extract_and_verify() {
    local zipfile=$1
    local dirname=${zipfile%.zip}
    
    echo "📦 Extracting $zipfile..."
    
    # Check if already extracted
    if [ -d "$dirname" ]; then
        echo "   ⚠️  $dirname already exists. Skipping..."
        return 0
    fi
    
    # Extract
    if unzip -q "$zipfile"; then
        echo "   ✅ Extracted successfully"
        
        # Count files
        local filecount=$(find "$dirname" -type f | wc -l)
        echo "   📊 $filecount files extracted"
    else
        echo "   ❌ Failed to extract $zipfile"
        return 1
    fi
    
    echo ""
}

# Extract all archives
extract_and_verify "book_copyright_images.zip"
extract_and_verify "code_copyright_images.zip"
extract_and_verify "lyrics_copyright_images.zip"
extract_and_verify "news_copyright_images.zip"

# Verify extraction
echo "================================================"
echo "  Verification"
echo "================================================"
echo ""

all_present=true

for dirname in book_copyright_images code_copyright_images lyrics_copyright_images news_copyright_images; do
    if [ -d "$dirname" ]; then
        echo "✅ $dirname/"
    else
        echo "❌ $dirname/ (missing)"
        all_present=false
    fi
done

echo ""

if [ "$all_present" = true ]; then
    echo "🎉 All image datasets extracted successfully!"
    echo ""
    echo "You can now run the evaluation scripts."
    exit 0
else
    echo "⚠️  Some datasets failed to extract. Please check the errors above."
    exit 1
fi

