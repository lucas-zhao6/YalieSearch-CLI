"""
CLIP Image Embedding Generator - Optimized Version

Generates embeddings for all Yalie photos using OpenAI's CLIP model.
Optimizations:
- Parallel image downloads (ThreadPoolExecutor)
- Half precision (fp16) for faster inference
- Connection pooling (requests.Session)
- Retry logic with exponential backoff
- Checkpoint/resume support
- Progress bars (tqdm)
- torch.compile for PyTorch 2.0+
"""

import json
import os
import time
from PIL import Image
from transformers import AutoProcessor, CLIPVisionModelWithProjection
import requests
import torch
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

#-------------------------------------------------------------------------#
# Configuration
#-------------------------------------------------------------------------#

BATCH_SIZE = 64
MAX_WORKERS = 20  # Parallel download threads
MAX_RETRIES = 3
CHECKPOINT_EVERY = 5  # Save checkpoint every N batches
CHECKPOINT_FILE = "checkpoint.json"
IMAGE_SIZE = (224, 224)  # CLIP input size

#-------------------------------------------------------------------------#
# Device Setup
#-------------------------------------------------------------------------#

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")

#-------------------------------------------------------------------------#
# Model Loading
#-------------------------------------------------------------------------#

print("Loading CLIP model...")
model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")

# Use half precision for GPU devices (faster, less memory)
if device in ("mps", "cuda"):
    model = model.half()
    print("Using half precision (fp16)")

model = model.to(device)

# Try to use torch.compile for PyTorch 2.0+ (faster inference)
try:
    if hasattr(torch, 'compile'):
        model = torch.compile(model)
        print("Using torch.compile for optimized inference")
except Exception as e:
    print(f"torch.compile not available: {e}")

processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14", use_fast=True)

# Create a session for connection pooling
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS)
session.mount('http://', adapter)
session.mount('https://', adapter)

#-------------------------------------------------------------------------#
# Helper Functions
#-------------------------------------------------------------------------#

def download_image(url, timeout=10):
    """Download and preprocess a single image with retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert('RGB')
            # Pre-resize to reduce memory usage
            img = img.resize(IMAGE_SIZE, Image.LANCZOS)
            return img
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(0.5 * (2 ** attempt))  # Exponential backoff
            else:
                return None
    return None


def download_batch_parallel(batch, start_idx):
    """Download images in parallel using ThreadPoolExecutor."""
    images = {}
    
    def download_with_index(item):
        idx, yalie = item
        url = yalie.get('image')
        if url:
            img = download_image(url)
            if img:
                return (idx, img)
        return None
    
    items = [(start_idx + i, yalie) for i, yalie in enumerate(batch)]
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_with_index, item): item for item in items}
        
        for future in tqdm(as_completed(futures), total=len(futures), 
                          desc="  Downloading", leave=False):
            result = future.result()
            if result:
                idx, img = result
                images[idx] = img
    
    return images


def load_checkpoint():
    """Load checkpoint if it exists."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                checkpoint = json.load(f)
            print(f"Resuming from checkpoint: {len(checkpoint['embeddings'])} embeddings already done")
            return checkpoint
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
    return {"embeddings": {}, "last_batch": -1}


def save_checkpoint(embeddings, last_batch):
    """Save checkpoint to disk."""
    checkpoint = {
        "embeddings": embeddings,
        "last_batch": last_batch
    }
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f)


def delete_checkpoint():
    """Delete checkpoint file on successful completion."""
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("Checkpoint file deleted")


#-------------------------------------------------------------------------#
# Main Embedding Function
#-------------------------------------------------------------------------#

def clip_embed(batch_size=BATCH_SIZE):
    """Generate CLIP embeddings for all images with all optimizations."""
    
    # Load data
    with open('yalies.json', 'r') as f:
        yalies = json.load(f)
    
    total = len(yalies)
    total_batches = (total + batch_size - 1) // batch_size
    print(f"\nProcessing {total} people in {total_batches} batches...")
    
    # Load checkpoint if exists
    checkpoint = load_checkpoint()
    embeddings_dict = checkpoint['embeddings']
    start_batch = checkpoint['last_batch'] + 1
    
    if start_batch > 0:
        print(f"Skipping batches 1-{start_batch} (already completed)")
    
    # Process batches
    with tqdm(total=total_batches, desc="Overall progress", initial=start_batch) as pbar:
        for batch_num in range(start_batch, total_batches):
            batch_start = batch_num * batch_size
            batch_end = min(batch_start + batch_size, total)
            batch = yalies[batch_start:batch_end]
            
            pbar.set_postfix({"batch": f"{batch_num + 1}/{total_batches}"})
            
            # Skip indices we already have embeddings for
            batch_to_process = []
            indices_to_process = []
            for i, yalie in enumerate(batch):
                global_idx = batch_start + i
                if str(global_idx) not in embeddings_dict:
                    batch_to_process.append(yalie)
                    indices_to_process.append(global_idx)
            
            if not batch_to_process:
                pbar.update(1)
                continue
            
            # Download images in parallel
            images_dict = download_batch_parallel(batch_to_process, batch_start)
            
            if not images_dict:
                pbar.update(1)
                continue
            
            # Prepare images for model
            sorted_indices = sorted(images_dict.keys())
            images = [images_dict[idx] for idx in sorted_indices]
            
            # Process through model
            image_tensors = processor(images=images, return_tensors="pt")
            
            # Move to device with correct dtype
            if device in ("mps", "cuda"):
                image_tensors = {k: v.to(device).half() for k, v in image_tensors.items()}
            else:
                image_tensors = {k: v.to(device) for k, v in image_tensors.items()}
            
            with torch.inference_mode():
                outputs = model(**image_tensors)
            
            batch_embeddings = outputs.image_embeds.float().cpu().numpy()
            
            # Store embeddings
            for i, global_idx in enumerate(sorted_indices):
                embeddings_dict[str(global_idx)] = batch_embeddings[i].tolist()
            
            # Save checkpoint periodically
            if (batch_num + 1) % CHECKPOINT_EVERY == 0:
                save_checkpoint(embeddings_dict, batch_num)
                tqdm.write(f"  Checkpoint saved ({len(embeddings_dict)} embeddings)")
            
            pbar.update(1)
    
    # Build final output
    print("\nBuilding output JSON...")
    payloads = []
    
    for idx_str, embedding in tqdm(embeddings_dict.items(), desc="Building output"):
        idx = int(idx_str)
        if idx < len(yalies):
            yalie = yalies[idx].copy()
            yalie['embedding'] = embedding
            payloads.append(yalie)
    
    # Write output (no indent for smaller file)
    with open('yalie_embedding.json', 'w') as f:
        json.dump(payloads, f)
    
    # Clean up checkpoint
    delete_checkpoint()
    
    print(f"\n✅ Done! Saved {len(payloads)} embeddings to yalie_embedding.json")
    print(f"   ({total - len(payloads)} people skipped due to missing/failed images)")
    
    return payloads


#-------------------------------------------------------------------------#
# Entry Point
#-------------------------------------------------------------------------#

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate CLIP embeddings for Yalie photos")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, 
                        help=f"Batch size for processing (default: {BATCH_SIZE})")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS,
                        help=f"Number of parallel download workers (default: {MAX_WORKERS})")
    args = parser.parse_args()
    
    MAX_WORKERS = args.workers
    clip_embed(batch_size=args.batch_size)
