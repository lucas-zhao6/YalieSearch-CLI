"""
Visualize the CLIP embedding space for Yalie photos - Optimized Version

Features:
- UMAP for fast dimensionality reduction (much faster than t-SNE for large datasets)
- Interactive HTML with thumbnail images on hover
- Caching of thumbnails and coordinates
- Parallel downloads with progress bars
- Clustering visualization
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import requests
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse

#-------------------------------------------------------------------------#
# Configuration
#-------------------------------------------------------------------------#

THUMBNAIL_DIR = "thumbnails"
COORDS_CACHE = "coords_2d.npy"
THUMBNAIL_SIZE = (64, 64)
MAX_WORKERS = 20

#-------------------------------------------------------------------------#
# Data Loading
#-------------------------------------------------------------------------#

def load_embeddings(filepath='yalie_embedding.json'):
    """Load embeddings from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


#-------------------------------------------------------------------------#
# Dimensionality Reduction
#-------------------------------------------------------------------------#

def reduce_dimensions(embeddings, method='umap', force=False):
    """Reduce embedding dimensions to 2D using UMAP (fast) or t-SNE."""
    
    # Check cache first
    if os.path.exists(COORDS_CACHE) and not force:
        print(f"Loading cached coordinates from {COORDS_CACHE}...")
        return np.load(COORDS_CACHE)
    
    print(f"Reducing {embeddings.shape[0]} embeddings from {embeddings.shape[1]}D to 2D using {method.upper()}...")
    
    if method == 'umap':
        try:
            import umap
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=15,
                min_dist=0.1,
                metric='cosine',
                random_state=42,
                verbose=True
            )
        except ImportError:
            print("UMAP not installed. Falling back to t-SNE...")
            print("Install with: pip install umap-learn")
            method = 'tsne'
    
    if method == 'tsne':
        from sklearn.manifold import TSNE
        # Use PCA first for speed with large datasets
        if embeddings.shape[0] > 1000:
            from sklearn.decomposition import PCA
            print("Running PCA first (50 components) for speed...")
            pca = PCA(n_components=50, random_state=42)
            embeddings = pca.fit_transform(embeddings)
            print(f"PCA complete. Variance explained: {pca.explained_variance_ratio_.sum():.2%}")
        
        perplexity = min(30, len(embeddings) - 1)
        reducer = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=42,
            max_iter=1000,
            verbose=1
        )
    
    reduced = reducer.fit_transform(embeddings)
    
    # Cache the results
    np.save(COORDS_CACHE, reduced)
    print(f"Saved coordinates to {COORDS_CACHE}")
    
    return reduced


#-------------------------------------------------------------------------#
# Thumbnail Management
#-------------------------------------------------------------------------#

def get_thumbnail_path(yalie_id):
    """Get the local path for a cached thumbnail."""
    return os.path.join(THUMBNAIL_DIR, f"{yalie_id}.jpg")


def download_thumbnail(url, yalie_id, size=THUMBNAIL_SIZE):
    """Download and cache a thumbnail image."""
    thumb_path = get_thumbnail_path(yalie_id)
    
    # Return cached if exists
    if os.path.exists(thumb_path):
        return thumb_path
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert('RGB')
        img.thumbnail(size, Image.LANCZOS)
        img.save(thumb_path, 'JPEG', quality=85)
        return thumb_path
    except Exception as e:
        return None


def download_all_thumbnails(yalies, force=False):
    """Download all thumbnails in parallel with progress bar."""
    os.makedirs(THUMBNAIL_DIR, exist_ok=True)
    
    # Check what needs downloading
    to_download = []
    for y in yalies:
        yalie_id = y.get('id') or y.get('netid') or str(hash(y.get('image', '')))
        thumb_path = get_thumbnail_path(yalie_id)
        if force or not os.path.exists(thumb_path):
            to_download.append((y.get('image'), yalie_id))
    
    if not to_download:
        print(f"All {len(yalies)} thumbnails already cached!")
        return
    
    print(f"Downloading {len(to_download)} thumbnails ({len(yalies) - len(to_download)} cached)...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(download_thumbnail, url, yid): yid 
            for url, yid in to_download if url
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            future.result()  # Wait for completion


def get_thumbnail_base64(yalie):
    """Get base64-encoded thumbnail for embedding in HTML."""
    yalie_id = yalie.get('id') or yalie.get('netid') or str(hash(yalie.get('image', '')))
    thumb_path = get_thumbnail_path(yalie_id)
    
    if os.path.exists(thumb_path):
        with open(thumb_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    return None


#-------------------------------------------------------------------------#
# Visualization Functions
#-------------------------------------------------------------------------#

def plot_basic(coords, output_file='embedding_space_basic.png'):
    """Create a basic scatter plot of the embedding space."""
    plt.figure(figsize=(14, 12))
    plt.scatter(coords[:, 0], coords[:, 1], alpha=0.5, s=15, c='steelblue')
    plt.title(f"CLIP Embedding Space ({len(coords)} people)", fontsize=14)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved: {output_file}")


def plot_with_clusters(coords, yalies, output_file='embedding_space_clusters.png'):
    """Create a scatter plot colored by detected clusters."""
    from sklearn.cluster import KMeans
    
    # Detect clusters
    n_clusters = min(10, len(coords) // 50)  # Reasonable number of clusters
    if n_clusters < 2:
        n_clusters = 2
    
    print(f"Clustering into {n_clusters} groups...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(coords)
    
    plt.figure(figsize=(14, 12))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap='tab10', 
                         alpha=0.6, s=15)
    plt.colorbar(scatter, label='Cluster')
    plt.title(f"CLIP Embedding Space - {n_clusters} Clusters ({len(coords)} people)", fontsize=14)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved: {output_file}")


def plot_interactive_html(coords, yalies, output_file='embedding_visualization.html'):
    """Create an interactive HTML visualization with thumbnails on hover."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly not installed. Run: pip install plotly")
        return
    
    print("Building interactive HTML with thumbnail hover...")
    
    # Prepare data for each point
    hover_texts = []
    image_tags = []
    
    for i, y in enumerate(tqdm(yalies, desc="Preparing hover data")):
        name = f"{y.get('first_name', '')} {y.get('last_name', '')}".strip() or 'Unknown'
        college = y.get('college', 'N/A')
        year = y.get('year', 'N/A')
        major = y.get('major', 'N/A')
        
        # Get base64 thumbnail
        b64 = get_thumbnail_base64(y)
        if b64:
            img_tag = f'<img src="data:image/jpeg;base64,{b64}" width="80" height="80"><br>'
        else:
            img_tag = ""
        
        hover_text = f"{img_tag}<b>{name}</b><br>College: {college}<br>Year: {year}"
        hover_texts.append(hover_text)
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        mode='markers',
        marker=dict(
            size=6,
            color=np.arange(len(coords)),
            colorscale='Viridis',
            opacity=0.7
        ),
        text=hover_texts,
        hovertemplate='%{text}<extra></extra>',
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
        )
    ))
    
    fig.update_layout(
        title=f'CLIP Embedding Space - {len(yalies)} Yalie Photos (Hover for details)',
        xaxis_title='UMAP Dimension 1',
        yaxis_title='UMAP Dimension 2',
        width=1200,
        height=900,
        hovermode='closest',
        template='plotly_white'
    )
    
    fig.write_html(output_file)
    print(f"Saved: {output_file}")
    print("Open this file in a browser to explore with thumbnail hover!")


#-------------------------------------------------------------------------#
# Analysis
#-------------------------------------------------------------------------#

def analyze_embeddings(embeddings, yalies):
    """Print statistics about the embedding space."""
    print("\n" + "="*60)
    print("EMBEDDING SPACE ANALYSIS")
    print("="*60)
    
    print(f"\nDataset size: {len(yalies)} people")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    # Compute pairwise similarities (sample if too large)
    n = len(embeddings)
    sample_size = min(1000, n)
    if n > sample_size:
        sample_idx = np.random.choice(n, sample_size, replace=False)
        sample = embeddings[sample_idx]
    else:
        sample = embeddings
    
    # Normalize embeddings
    norms = np.linalg.norm(sample, axis=1)
    normalized = sample / norms[:, np.newaxis]
    
    # Compute cosine similarities
    similarities = np.dot(normalized, normalized.T)
    
    # Get off-diagonal elements
    mask = ~np.eye(len(similarities), dtype=bool)
    off_diag = similarities[mask]
    
    print(f"\nCosine Similarity Statistics (sampled {sample_size} pairs):")
    print(f"  Mean: {off_diag.mean():.4f}")
    print(f"  Std:  {off_diag.std():.4f}")
    print(f"  Min:  {off_diag.min():.4f}")
    print(f"  Max:  {off_diag.max():.4f}")
    
    # Interpretation
    print("\n" + "-"*60)
    if off_diag.std() < 0.1:
        print("⚠️  Low variance - embeddings may be too similar")
    elif off_diag.std() > 0.15:
        print("✓  Good variance - embeddings show diversity")
    
    if off_diag.mean() > 0.8:
        print("⚠️  High similarity - faces may be hard to distinguish")
    elif off_diag.mean() < 0.5:
        print("✓  Good spread - faces are well separated")
    else:
        print("~  Moderate similarity - typical for face embeddings")
    print("-"*60 + "\n")


#-------------------------------------------------------------------------#
# Main
#-------------------------------------------------------------------------#

def main(force=False, skip_thumbnails=False):
    """Run the full visualization pipeline."""
    
    print("Loading embeddings...")
    yalies = load_embeddings()
    embeddings = np.array([y['embedding'] for y in yalies], dtype=np.float32)
    
    # Analyze
    analyze_embeddings(embeddings, yalies)
    
    # Download thumbnails (needed for hover)
    if not skip_thumbnails:
        download_all_thumbnails(yalies, force=force)
    
    # Reduce dimensions
    coords_2d = reduce_dimensions(embeddings, method='umap', force=force)
    
    # Create visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60 + "\n")
    
    print("1. Basic scatter plot...")
    plot_basic(coords_2d)
    
    print("\n2. Clustered scatter plot...")
    plot_with_clusters(coords_2d, yalies)
    
    print("\n3. Interactive HTML with thumbnail hover...")
    plot_interactive_html(coords_2d, yalies)
    
    print("\n" + "="*60)
    print("✅ DONE! Generated files:")
    print("   - embedding_space_basic.png")
    print("   - embedding_space_clusters.png")
    print("   - embedding_visualization.html (interactive with thumbnails)")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize CLIP embedding space")
    parser.add_argument("--force", "-f", action="store_true", 
                        help="Force re-computation (ignore cache)")
    parser.add_argument("--skip-thumbnails", action="store_true",
                        help="Skip thumbnail download (faster, no hover images)")
    
    args = parser.parse_args()
    main(force=args.force, skip_thumbnails=args.skip_thumbnails)
