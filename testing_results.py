"""
CLIP-based Semantic Face Search - Optimized Version

Search for people by text description using CLIP embeddings.
Optimizations:
- Model and embeddings cached at module level
- GPU/MPS acceleration
- Half precision (fp16)
- Precomputed normalized embeddings
"""

from transformers import AutoTokenizer, CLIPTextModelWithProjection
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import requests
import json
from concurrent.futures import ThreadPoolExecutor

#-------------------------------------------------------------------------#
# Device Setup
#-------------------------------------------------------------------------#

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

#-------------------------------------------------------------------------#
# Cached Model and Embeddings (loaded once at module import)
#-------------------------------------------------------------------------#

_model = None
_tokenizer = None
_yalies = None
_embeddings_normalized = None


def _load_model():
    """Load and cache the CLIP text model."""
    global _model, _tokenizer
    
    if _model is None:
        print(f"Loading CLIP text model (device: {device})...")
        _model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
        
        # Use half precision on GPU devices
        if device in ("mps", "cuda"):
            _model = _model.half()
        
        _model = _model.to(device)
        _model.eval()
        
        _tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        print("Model loaded!")
    
    return _model, _tokenizer


def _load_embeddings(filepath='yalie_embedding.json'):
    """Load and cache embeddings with precomputed normalization."""
    global _yalies, _embeddings_normalized
    
    if _yalies is None:
        print(f"Loading embeddings from {filepath}...")
        with open(filepath, 'r') as f:
            _yalies = json.load(f)
        
        # Convert to numpy array
        embeddings = np.array([y['embedding'] for y in _yalies], dtype=np.float32)
        
        # Precompute normalized embeddings (avoids doing this every search)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        _embeddings_normalized = embeddings / norms
        
        print(f"Loaded {len(_yalies)} embeddings!")
    
    return _yalies, _embeddings_normalized


#-------------------------------------------------------------------------#
# Search Functions
#-------------------------------------------------------------------------#

def search(query: str, k: int = 3):
    """
    Search for top-k similar faces given a text query.
    
    Args:
        query: Text description to search for
        k: Number of results to return
    
    Returns:
        List of dicts with 'yalie' and 'score' keys
    """
    # Load cached model and embeddings
    model, tokenizer = _load_model()
    yalies, embeddings_norm = _load_embeddings()
    
    # Encode query text
    inputs = tokenizer(query, padding=True, return_tensors="pt")
    
    # Move to device with correct dtype
    if device in ("mps", "cuda"):
        inputs = {k: v.to(device) for k, v in inputs.items()}
    else:
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.inference_mode():
        outputs = model(**inputs)
    
    # Get query vector and normalize
    query_vector = outputs.text_embeds.float().cpu().squeeze().numpy()
    query_norm = query_vector / np.linalg.norm(query_vector)
    
    # Compute cosine similarities (embeddings already normalized)
    similarities = np.dot(embeddings_norm, query_norm)
    
    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:k]
    
    # Build results
    results = []
    for idx in top_indices:
        results.append({
            'yalie': yalies[idx],
            'score': float(similarities[idx])
        })
    
    return results


def download_image(url, timeout=10):
    """Download a single image from URL."""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception:
        return None


def display_results(results, show_images=True):
    """Display search results with optional image display."""
    
    # Download images in parallel if needed
    if show_images:
        urls = [r['yalie'].get('image') for r in results]
        images = []
        
        with ThreadPoolExecutor(max_workers=len(results)) as executor:
            images = list(executor.map(lambda url: download_image(url) if url else None, urls))
    
    for i, result in enumerate(results):
        yalie = result['yalie']
        score = result['score']
        
        # Get name
        first = yalie.get('first_name', '')
        last = yalie.get('last_name', '')
        name = f"{first} {last}".strip() or 'Unknown'
        
        # Get additional info
        college = yalie.get('college', '')
        year = yalie.get('year', '')
        
        info_parts = [f"#{i+1}: {name}"]
        if college:
            info_parts.append(f"College: {college}")
        if year:
            info_parts.append(f"Year: {year}")
        info_parts.append(f"Similarity: {score:.4f}")
        
        print("\n" + " | ".join(info_parts))
        
        # Show image
        if show_images and images[i]:
            images[i].show()


def interactive_mode():
    """Interactive search REPL."""
    print("\n" + "="*50)
    print("YALIE SEARCH - Interactive Mode")
    print("="*50)
    print("Enter a description to search for people.")
    print("Commands: 'quit' to exit, 'k=N' to change result count")
    print("="*50 + "\n")
    
    # Pre-load model and embeddings
    _load_model()
    _load_embeddings()
    
    k = 10
    
    while True:
        try:
            query = input("\nSearch query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not query:
            continue
        
        if query.lower() == 'quit':
            print("Goodbye!")
            break
        
        if query.lower().startswith('k='):
            try:
                k = int(query[2:])
                print(f"Results count set to {k}")
            except ValueError:
                print("Invalid k value")
            continue
        
        print(f"\nSearching for: '{query}' (top {k} results)...")
        results = search(query, k=k)
        display_results(results, show_images=True)


#-------------------------------------------------------------------------#
# Entry Point
#-------------------------------------------------------------------------#

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Search for Yalies by description")
    parser.add_argument("--query", "-q", type=str, help="Search query")
    parser.add_argument("--k", "-k", type=int, default=10, help="Number of results (default: 10)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--no-images", action="store_true", help="Don't display images")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
    elif args.query:
        print(f"Searching for: '{args.query}'")
        print(f"Finding top {args.k} matches...\n")
        results = search(args.query, k=args.k)
        display_results(results, show_images=not args.no_images)
    else:
        # Prompt for input
        query = input("Enter search query: ").strip()
        if not query:
            print("No query provided. Exiting.")
            exit(1)
        
        print(f"\nSearching for: '{query}'")
        print(f"Finding top {args.k} matches...\n")
        results = search(query, k=args.k)
        display_results(results, show_images=not args.no_images)
