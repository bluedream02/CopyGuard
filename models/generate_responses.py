"""
Generate Baseline Responses (Without Defense)

This script generates model responses without any defense mechanism.
Used to establish baseline performance metrics.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.model_loader import ModelLoader
from utils.config_loader import load_config
from utils.path_helper import get_project_root


def load_dataset(dataset_path: str) -> List[Dict]:
    """
    Load dataset from JSON file.
    
    Args:
        dataset_path: Path to dataset JSON file
        
    Returns:
        List of dataset samples
    """
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("Dataset must be a list of samples")
    
    return data


def load_queries(query_path: str) -> Dict[str, List[str]]:
    """
    Load query seeds from JSON file.
    
    Args:
        query_path: Path to query seeds JSON file
        
    Returns:
        Dictionary mapping categories to query lists
    """
    with open(query_path, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    
    return queries


def apply_notice_mode(query: str, notice_mode: int, copyright_text: str = "") -> str:
    """
    Apply copyright notice to query based on notice mode.
    
    Args:
        query: Original query text
        notice_mode: Notice mode (0=none, 1=generic, 2=original)
        copyright_text: Original copyright text
        
    Returns:
        Query with copyright notice prepended
    """
    if notice_mode == 0:
        return query
    elif notice_mode == 1:
        return f"All rights reserved.\n{query}"
    elif notice_mode == 2:
        return f"{copyright_text}\n{query}"
    else:
        raise ValueError(f"Invalid notice_mode: {notice_mode}")


def get_image_path(sample: Dict, image_mode: int) -> str:
    """
    Get image path based on image mode.
    
    Args:
        sample: Dataset sample
        image_mode: Image mode (0=plain, 1=generic notice, 2=original notice)
        
    Returns:
        Path to image file
    """
    if image_mode == 0:
        return sample.get("img_file_0", "")
    elif image_mode == 1:
        return sample.get("img_file_1", "")
    elif image_mode == 2:
        return sample.get("img_file_2", "")
    else:
        raise ValueError(f"Invalid image_mode: {image_mode}")


def generate_responses(model, dataset: List[Dict], queries: Dict[str, List[str]],
                      image_mode: int, notice_mode: int, **gen_kwargs) -> List[Dict]:
    """
    Generate responses for all samples in dataset.
    
    Args:
        model: Loaded model instance
        dataset: List of dataset samples
        queries: Query dictionary
        image_mode: Image presentation mode
        notice_mode: Copyright notice mode
        **gen_kwargs: Additional generation parameters
        
    Returns:
        Dataset with responses added
    """
    results = []
    
    for sample in tqdm(dataset, desc="Generating responses"):
        # Get image path
        image_path = get_image_path(sample, image_mode)
        copyright_text = sample.get("copyright_text", "")
        
        # Generate responses for all queries
        responses = []
        
        for category, query_list in queries.items():
            for query in query_list:
                # Apply copyright notice
                modified_query = apply_notice_mode(query, notice_mode, copyright_text)
                
                try:
                    # Generate response
                    response = model.generate(
                        image_path=image_path,
                        query=modified_query,
                        **gen_kwargs
                    )
                    
                    responses.append({
                        "category": category,
                        "query": modified_query,
                        "response": response
                    })
                    
                except Exception as e:
                    print(f"Error generating response: {e}")
                    responses.append({
                        "category": category,
                        "query": modified_query,
                        "response": f"[ERROR] {str(e)}"
                    })
        
        # Add responses to sample
        sample_result = sample.copy()
        sample_result["responses"] = responses
        results.append(sample_result)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate baseline responses without defense mechanism"
    )
    
    # Model arguments
    parser.add_argument("--model-type", type=str, required=True,
                       help="Type of model (qwen, deepseek, llava, glm, gpt, etc.)")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to model checkpoint or model name")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, required=True,
                       help="Path to dataset JSON file")
    parser.add_argument("--queries", type=str, default=None,
                       help="Path to query seeds JSON (default: dataset/query/seeds.json)")
    
    # Output arguments
    parser.add_argument("--output", type=str, required=True,
                       help="Path to save output JSON file")
    
    # Notice configuration
    parser.add_argument("--image-mode", type=int, default=0, choices=[0, 1, 2],
                       help="Image mode: 0=plain, 1=generic notice, 2=original notice")
    parser.add_argument("--notice-mode", type=int, default=0, choices=[0, 1, 2],
                       help="Notice mode: 0=none, 1=generic, 2=original")
    
    # Generation parameters
    parser.add_argument("--max-new-tokens", type=int, default=512,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.2,
                       help="Sampling temperature")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on (cuda/cpu)")
    
    # Model-specific parameters
    parser.add_argument("--llava-repo-path", type=str, default=None,
                       help="Path to LLaVA repository (required for LLaVA)")
    parser.add_argument("--api-key", type=str, default=None,
                       help="API key for API-based models")
    parser.add_argument("--api-base", type=str, default=None,
                       help="API base URL for API-based models")
    
    args = parser.parse_args()
    
    # Load configuration
    project_root = get_project_root()
    
    # Set default queries path
    if args.queries is None:
        args.queries = os.path.join(project_root, "dataset", "query", "seeds.json")
    
    # Load dataset and queries
    print(f"Loading dataset from {args.dataset}...")
    dataset = load_dataset(args.dataset)
    print(f"Loaded {len(dataset)} samples")
    
    print(f"Loading queries from {args.queries}...")
    queries = load_queries(args.queries)
    print(f"Loaded {sum(len(q) for q in queries.values())} queries across {len(queries)} categories")
    
    # Load model
    print(f"Loading {args.model_type} model from {args.model_path}...")
    model_kwargs = {
        "device": args.device,
    }
    
    # Add model-specific parameters
    if args.model_type.lower() in ['llava', 'llava-1.5', 'llava-next']:
        if args.llava_repo_path:
            model_kwargs["llava_repo_path"] = args.llava_repo_path
        model_kwargs["temperature"] = args.temperature
    
    if args.model_type.lower() in ['gpt', 'gpt-4', 'gpt-4o', 'gpt-4o-mini', 'claude', 'gemini']:
        if args.api_key:
            model_kwargs["api_key"] = args.api_key
        if args.api_base:
            model_kwargs["api_base"] = args.api_base
    
    model = ModelLoader.load_model(args.model_type, args.model_path, **model_kwargs)
    
    # Generation parameters
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
    }
    
    if args.model_type.lower() not in ['llava', 'llava-1.5', 'llava-next']:
        gen_kwargs["temperature"] = args.temperature
    
    # Generate responses
    print(f"\nGenerating responses (image_mode={args.image_mode}, notice_mode={args.notice_mode})...")
    results = generate_responses(
        model=model,
        dataset=dataset,
        queries=queries,
        image_mode=args.image_mode,
        notice_mode=args.notice_mode,
        **gen_kwargs
    )
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"\nResults saved to {args.output}")
    print(f"Total samples processed: {len(results)}")
    print(f"Total responses generated: {sum(len(s['responses']) for s in results)}")


if __name__ == "__main__":
    main()

