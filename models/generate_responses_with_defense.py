"""
Generate Responses With CopyGuard Defense

This script generates model responses with the CopyGuard defense framework enabled.
Includes copyright detection, verification, and risk-based blocking.
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
from defense.copyguard import CopyGuard
from utils.config_loader import load_config
from utils.path_helper import get_project_root


def load_dataset(dataset_path: str) -> List[Dict]:
    """Load dataset from JSON file"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("Dataset must be a list of samples")
    
    return data


def load_queries(query_path: str) -> Dict[str, List[str]]:
    """Load query seeds from JSON file"""
    with open(query_path, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    
    return queries


def apply_notice_mode(query: str, notice_mode: int, copyright_text: str = "") -> str:
    """Apply copyright notice to query"""
    if notice_mode == 0:
        return query
    elif notice_mode == 1:
        return f"All rights reserved.\n{query}"
    elif notice_mode == 2:
        return f"{copyright_text}\n{query}"
    else:
        raise ValueError(f"Invalid notice_mode: {notice_mode}")


def get_image_path(sample: Dict, image_mode: int) -> str:
    """Get image path based on image mode"""
    if image_mode == 0:
        return sample.get("img_file_0", "")
    elif image_mode == 1:
        return sample.get("img_file_1", "")
    elif image_mode == 2:
        return sample.get("img_file_2", "")
    else:
        raise ValueError(f"Invalid image_mode: {image_mode}")


def generate_responses_with_defense(model, copyguard: CopyGuard,
                                    dataset: List[Dict], queries: Dict[str, List[str]],
                                    image_mode: int, notice_mode: int,
                                    block_risky_queries: bool = True,
                                    **gen_kwargs) -> List[Dict]:
    """
    Generate responses with CopyGuard defense enabled.
    
    Args:
        model: Loaded model instance
        copyguard: CopyGuard instance
        dataset: List of dataset samples
        queries: Query dictionary
        image_mode: Image presentation mode
        notice_mode: Copyright notice mode
        block_risky_queries: Whether to block risky queries
        **gen_kwargs: Additional generation parameters
        
    Returns:
        Dataset with responses and defense info
    """
    results = []
    
    for sample in tqdm(dataset, desc="Generating responses with defense"):
        # Get image and text
        image_path = get_image_path(sample, image_mode)
        text = sample.get("text", "")
        copyright_text = sample.get("copyright_text", "")
        
        # Analyze content with CopyGuard (only once per sample)
        content_analysis = copyguard.analyze_content(
            image_path=image_path,
            text=text,
            content_type=None  # Auto-detect or specify if needed
        )
        
        # Generate responses for all queries
        responses = []
        
        for category, query_list in queries.items():
            for query in query_list:
                # Apply copyright notice
                modified_query = apply_notice_mode(query, notice_mode, copyright_text)
                
                try:
                    # Process request through CopyGuard
                    defense_result = copyguard.process_request(
                        query=modified_query,
                        image_path=image_path,
                        text=text,
                        content_type=None
                    )
                    
                    # Check if query should be blocked
                    should_block = defense_result['should_block'] and block_risky_queries
                    
                    if should_block:
                        # Block risky query with reminder
                        response_text = f"[BLOCKED] {defense_result['reminder']}"
                    else:
                        # Generate response (potentially with reminder prepended)
                        if defense_result['reminder'] and content_analysis['is_protected']:
                            # Prepend reminder to query
                            query_with_reminder = defense_result['formatted_reminder']
                        else:
                            query_with_reminder = modified_query
                        
                        # Generate model response
                        response_text = model.generate(
                            image_path=image_path,
                            query=query_with_reminder,
                            **gen_kwargs
                        )
                    
                    # Store response with defense info
                    responses.append({
                        "category": category,
                        "query": modified_query,
                        "response": response_text,
                        "defense_info": {
                            "blocked": should_block,
                            "has_copyright": content_analysis['is_protected'],
                            "copyright_status": content_analysis['copyright_status'],
                            "has_notice": content_analysis['has_notice'],
                            "risk_level": defense_result['query_analysis']['risk_analysis']['risk_level'],
                            "reminder": defense_result['reminder'],
                            "suggested_query": defense_result.get('suggested_query', None)
                        }
                    })
                    
                except Exception as e:
                    print(f"Error generating response with defense: {e}")
                    responses.append({
                        "category": category,
                        "query": modified_query,
                        "response": f"[ERROR] {str(e)}",
                        "defense_info": {
                            "blocked": False,
                            "error": str(e)
                        }
                    })
        
        # Add responses to sample
        sample_result = sample.copy()
        sample_result["responses"] = responses
        # Also store content analysis at sample level
        sample_result["content_analysis"] = {
            "has_copyright": content_analysis['is_protected'],
            "copyright_status": content_analysis['copyright_status'],
            "has_notice": content_analysis['has_notice']
        }
        results.append(sample_result)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate responses with CopyGuard defense mechanism"
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
    
    # Defense configuration
    parser.add_argument("--enable-ocr", action="store_true", default=True,
                       help="Enable OCR-based copyright detection")
    parser.add_argument("--enable-verifier", action="store_true", default=True,
                       help="Enable copyright status verification")
    parser.add_argument("--enable-risk-analyzer", action="store_true", default=True,
                       help="Enable query risk analysis")
    parser.add_argument("--enable-reminder", action="store_true", default=True,
                       help="Enable copyright status reminders")
    parser.add_argument("--block-risky", action="store_true", default=True,
                       help="Block risky queries instead of just warning")
    
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
    
    # Initialize CopyGuard
    print("\nInitializing CopyGuard defense framework...")
    copyguard = CopyGuard(
        enable_ocr=args.enable_ocr,
        enable_verifier=args.enable_verifier,
        enable_risk_analyzer=args.enable_risk_analyzer,
        enable_reminder=args.enable_reminder
    )
    
    # Load model
    print(f"\nLoading {args.model_type} model from {args.model_path}...")
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
    
    # Generate responses with defense
    print(f"\nGenerating responses with CopyGuard defense...")
    print(f"  Image mode: {args.image_mode}, Notice mode: {args.notice_mode}")
    print(f"  Block risky queries: {args.block_risky}")
    
    results = generate_responses_with_defense(
        model=model,
        copyguard=copyguard,
        dataset=dataset,
        queries=queries,
        image_mode=args.image_mode,
        notice_mode=args.notice_mode,
        block_risky_queries=args.block_risky,
        **gen_kwargs
    )
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    # Print statistics
    total_responses = sum(len(s['responses']) for s in results)
    blocked_responses = sum(
        sum(1 for r in s['responses'] if r.get('defense_info', {}).get('blocked', False))
        for s in results
    )
    
    print(f"\nResults saved to {args.output}")
    print(f"Total samples processed: {len(results)}")
    print(f"Total responses generated: {total_responses}")
    print(f"Blocked responses: {blocked_responses} ({100*blocked_responses/total_responses:.1f}%)")


if __name__ == "__main__":
    main()

