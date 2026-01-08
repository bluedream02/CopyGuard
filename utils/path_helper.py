"""
Path helper utilities for dataset file paths.

This module provides functions to construct correct file paths
for text and image files in the dataset.
"""

import os
import warnings
from typing import Optional


def get_text_file_path(dataset_type: str, filename: str, base_dir: str = "dataset") -> str:
    """
    Get the path to a text file in the dataset.
    
    Args:
        dataset_type: Type of dataset ('book_copyright', 'code_copyright', 
                     'lyrics_copyright', 'news_copyright')
        filename: Name of the file (e.g., '1Q84.txt', 'bert.txt')
        base_dir: Base directory for dataset (default: 'dataset')
        
    Returns:
        Full path to the text file
    """
    # Ensure filename has .txt extension
    if not filename.endswith('.txt'):
        filename = filename + '.txt'
    
    return os.path.join(base_dir, dataset_type, filename)


def get_image_file_path(dataset_type: str, filename: str, image_mode: int = 0, 
                        base_dir: str = "dataset") -> str:
    """
    Get the path to an image file in the dataset.
    
    Args:
        dataset_type: Type of dataset ('book_copyright', 'code_copyright', 
                     'lyrics_copyright', 'news_copyright')
        filename: Base name of the file (e.g., '1Q84', 'bert')
        image_mode: Image mode (0, 1, or 2)
        base_dir: Base directory for dataset (default: 'dataset')
        
    Returns:
        Full path to the image file
    """
    # Construct image directory name
    image_dir = f"{dataset_type}_images"
    
    # Handle different naming conventions
    if dataset_type == "book_copyright":
        # Books use format: "Title_sample_1.png"
        if not filename.endswith('_sample_1.png'):
            # Extract base name if it's a .txt file
            base_name = filename.replace('.txt', '')
            image_filename = f"{base_name}_sample_1.png"
        else:
            image_filename = filename
    elif dataset_type == "code_copyright":
        # Code uses format: "name.png"
        if filename.endswith('.txt'):
            image_filename = filename.replace('.txt', '.png')
        elif not filename.endswith('.png'):
            image_filename = filename + '.png'
        else:
            image_filename = filename
    elif dataset_type == "lyrics_copyright":
        # Lyrics use format: "Title_sample_1.png"
        if not filename.endswith('_sample_1.png'):
            base_name = filename.replace('.txt', '')
            image_filename = f"{base_name}_sample_1.png"
        else:
            image_filename = filename
    elif dataset_type == "news_copyright":
        # News uses format: "Title_screenshot.png" or "Title_screenshot copy.png"
        if filename.endswith('_text.txt'):
            # Remove _text and add _screenshot
            base_name = filename.replace('_text.txt', '')
            image_filename = f"{base_name}_screenshot.png"
        elif filename.endswith('.txt'):
            base_name = filename.replace('.txt', '')
            image_filename = f"{base_name}_screenshot.png"
        elif not filename.endswith('.png'):
            image_filename = filename + '.png'
        else:
            image_filename = filename
    else:
        # Default: assume .png extension
        if not filename.endswith('.png'):
            image_filename = filename.replace('.txt', '.png') if filename.endswith('.txt') else filename + '.png'
        else:
            image_filename = filename
    
    return os.path.join(base_dir, image_dir, str(image_mode), image_filename)


def get_all_image_paths(dataset_type: str, filename: str, base_dir: str = "dataset") -> dict:
    """
    Get paths to all three image modes for a given file.
    
    Args:
        dataset_type: Type of dataset
        filename: Base name of the file
        base_dir: Base directory for dataset
        
    Returns:
        Dictionary with keys 'img_file_0', 'img_file_1', 'img_file_2'
    """
    return {
        'img_file_0': get_image_file_path(dataset_type, filename, 0, base_dir),
        'img_file_1': get_image_file_path(dataset_type, filename, 1, base_dir),
        'img_file_2': get_image_file_path(dataset_type, filename, 2, base_dir),
    }


def verify_file_paths(dataset_type: str, filename: str, base_dir: str = "dataset", 
                      warn_missing: bool = True) -> dict:
    """
    Verify that text and image files exist and return their paths.
    
    Args:
        dataset_type: Type of dataset
        filename: Base name of the file
        base_dir: Base directory for dataset
        warn_missing: Whether to warn about missing files
        
    Returns:
        Dictionary with file paths and existence status
    """
    txt_path = get_text_file_path(dataset_type, filename, base_dir)
    img_paths = get_all_image_paths(dataset_type, filename, base_dir)
    
    result = {
        'txt_file': txt_path,
        'txt_exists': os.path.exists(txt_path),
        'img_file_0': img_paths['img_file_0'],
        'img_0_exists': os.path.exists(img_paths['img_file_0']),
        'img_file_1': img_paths['img_file_1'],
        'img_1_exists': os.path.exists(img_paths['img_file_1']),
        'img_file_2': img_paths['img_file_2'],
        'img_2_exists': os.path.exists(img_paths['img_file_2']),
    }
    
    # Warn about missing files
    if warn_missing:
        if not result['txt_exists']:
            warnings.warn(f"Text file not found: {txt_path}")
        for mode in [0, 1, 2]:
            if not result[f'img_{mode}_exists']:
                warnings.warn(f"Image file not found: {result[f'img_file_{mode}']}")
    
    return result


if __name__ == "__main__":
    # Example usage
    print("Example: Book copyright")
    result = verify_file_paths("book_copyright", "1Q84.txt", "dataset")
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    print("\nExample: Code copyright")
    result = verify_file_paths("code_copyright", "bert.txt", "dataset")
    for key, value in result.items():
        print(f"  {key}: {value}")

