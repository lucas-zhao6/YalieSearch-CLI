# YalieSearch

A semantic face search CLI that uses OpenAI's CLIP model to find Yale students by text description. Describe what you're looking for in natural language, and the system finds the closest matches.

## Features

- **Semantic Search** — Search using natural language descriptions like "person with glasses and curly hair"
- **CLIP Embeddings** — Uses OpenAI's CLIP ViT-Large model for powerful vision-language understanding
- **GPU Acceleration** — Supports CUDA and Apple Silicon (MPS) with half-precision for fast inference
- **Interactive Mode** — REPL for continuous searching without reloading the model
- **Embedding Visualization** — Explore the embedding space with interactive UMAP plots

## Installation

```bash
# Clone the repo
git clone https://github.com/lucas-zhao6/YalieSearch-CLI.git
cd yalie_search

# Install dependencies
pip install -r requirements.txt

# Set your Yalies API key (optional, for data fetching)
export YALIES_API_KEY="your_api_key_here"
```

### Requirements

- Python 3.8+
- ~4GB disk space for model + embeddings
- GPU recommended but not required

## Quick Start

```bash
# Search for someone
python testing_results.py -q "person wearing glasses"

# Interactive search mode
python testing_results.py -i
```

## CLI Reference

### 1. Data Fetching — `data_prep.py`

Fetches student data from the Yalies API.

```bash
python data_prep.py [OPTIONS]

Options:
  -f, --force       Force re-fetch (ignore cache)
  -p, --pages N     Maximum pages to fetch (default: 100)
  -w, --workers N   Parallel workers (default: 10)
```

**Features:**
- Parallel page fetching with ThreadPoolExecutor
- Automatic retry with exponential backoff
- Caching to `yalies.json`
- Progress bar with tqdm

---

### 2. Embedding Generation — `model.py`

Generates CLIP embeddings for all photos.

```bash
python model.py [OPTIONS]

Options:
  --batch-size N    Batch size for processing (default: 64)
  --workers N       Parallel download workers (default: 20)
```

**Features:**
- GPU/MPS acceleration with half-precision (fp16)
- Parallel image downloads
- Checkpoint/resume support
- torch.compile optimization (PyTorch 2.0+)
- Saves to `yalie_embedding.json`

---

### 3. Search — `testing_results.py`

Search for people by text description.

```bash
python testing_results.py [OPTIONS]

Options:
  -q, --query TEXT  Search query
  -k, --k N         Number of results (default: 10)
  -i, --interactive Interactive REPL mode
  --no-images       Don't display result images
```

**Examples:**

```bash
# Single query
python testing_results.py -q "asian woman with long hair" -k 5

# Interactive mode
python testing_results.py -i

# In interactive mode:
# > asian man with beard
# > k=20              (change result count)
# > quit              (exit)
```

**Features:**
- Cached model and embeddings (fast subsequent searches)
- Cosine similarity ranking
- Opens result images in your default viewer

---

### 4. Visualization — `visualize_embeddings.py`

Visualize the embedding space.

```bash
python visualize_embeddings.py [OPTIONS]

Options:
  -f, --force           Force re-computation (ignore cache)
  --skip-thumbnails     Skip thumbnail download
```

**Generates:**
- `embedding_space_basic.png` — Basic scatter plot
- `embedding_space_clusters.png` — K-means clustered view
- `embedding_visualization.html` — Interactive HTML with thumbnail hover

**Features:**
- UMAP dimensionality reduction (faster than t-SNE)
- Cached coordinates and thumbnails
- Parallel thumbnail downloads
- Embedding space statistics

## Pipeline

Run these scripts in order for a fresh setup:

```bash
# 1. Fetch data from Yalies API
python data_prep.py

# 2. Generate CLIP embeddings (takes a while)
python model.py

# 3. Search!
python testing_results.py -i

# 4. (Optional) Visualize the embedding space
python visualize_embeddings.py
```

## Project Structure

```
yalie_search/
├── data_prep.py              # Yalies API data fetcher
├── model.py                  # CLIP embedding generator
├── testing_results.py        # Semantic search CLI
├── visualize_embeddings.py   # Embedding visualization
├── yalies.json               # Cached student data
├── yalie_embedding.json      # Generated embeddings
├── thumbnails/               # Cached photo thumbnails
├── coords_2d.npy             # Cached UMAP coordinates
└── requirements.txt
```

## How It Works

1. **CLIP** (Contrastive Language-Image Pretraining) learns a shared embedding space for images and text
2. Photos are encoded as 768-dimensional vectors using CLIP's vision encoder
3. Search queries are encoded using CLIP's text encoder
4. Cosine similarity finds the closest photo embeddings to the query embedding

## Performance

| Operation | Time (M1 Mac) | Time (CUDA GPU) |
|-----------|---------------|-----------------|
| Model load | ~5s | ~3s |
| Single search | ~50ms | ~20ms |
| Embed 1000 photos | ~3 min | ~1 min |

## References

- [OpenAI CLIP](https://github.com/openai/CLIP)
- [Hugging Face CLIP](https://huggingface.co/docs/transformers/model_doc/clip)
- [UMAP](https://umap-learn.readthedocs.io/)
