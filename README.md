# Grounded Output Explorer

An interactive Streamlit app and CLI that builds a three-layer “grounding graph” for a model response:
- Inputs: chunked prompt sentences
- Evidence: Valyu search results (clustered)
- Outputs: chunked model response (clustered)

Edges show cosine similarities between output clusters and inputs/evidence. The app helps you quickly see which parts of a response are grounded vs weak vs hallucinated.

## What’s new
- Switched generation to Google Gemini (fixed model) with a built-in safety wrapper and auto‑retry.
- Added a “Demo Prompt Library” with safe, neutral prompts that avoid safety blocks.
- Requirements updated to include `google-generativeai` and `importlib-metadata` (for Python 3.9 compatibility).
- Added `Performance_benchmarking.py` to compare “all sources” vs “pruned sources” using TF‑IDF graphs and a local token estimate.

## Features
- Layered, playable graph (drag nodes, reset layers)
- Vertical or horizontal layer orientation
- Output and evidence clustering to keep graphs small (5–30 clusters)
- Similarity-labeled edges with width scaled by score
- Node colors for status: grounded (green), weak (orange), hallucinated (red)
- Hover tooltips and a built-in legend
 - Generate output via Gemini with a safety wrapper and see finish_reason + safety ratings
 - Demo prompts to get started without hitting safety blocks

## How it works (pipeline)
1. Chunk the input prompt and model output into sentences (NLTK)
2. Retrieve evidence via Valyu Search API
3. Embed sentences with Sentence-Transformers (all-MiniLM-L6-v2)
4. Cluster output and evidence with KMeans
5. Compute cosine similarities on embeddings/centroids
6. Classify outputs as grounded/weak/hallucinated using thresholds
7. Build a NetworkX graph and render it with PyVis in Streamlit

## Project structure
```
hack2/
  main.py                     # CLI pipeline: builds graph.html
  streamlit_app.py            # Streamlit UI: interactive graph with Gemini generation
  Performance_benchmarking.py # Streamlit UI: compares ALL vs PRUNED sources (local token estimate)
  requirements.txt            # Python dependencies
  README.md                   # This file
```

## Prerequisites
- macOS (tested locally) or any OS with Python 3.9+
- Python 3.9 recommended (matches the bundled virtualenv in this repo)
- A Valyu API key
- A Gemini API key (GEMINI_API_KEY)

## Setup
Create a virtual environment (recommended) and install dependencies:

```zsh
# From the repo root
cd "hack2"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Provide your API keys via environment or a .env file:

```zsh
# Option A: export in your shell session
export VALYU_API_KEY="<your-key-here>"
export GEMINI_API_KEY="<your-gemini-key>"

# Option B: create a .env file in hack2/
cat > .env << 'EOF'
VALYU_API_KEY=<your-key-here>
GEMINI_API_KEY=<your-gemini-key>
EOF
```

## Run the Streamlit app (recommended)

```zsh
# From hack2/ (venv activated)
streamlit run streamlit_app.py
```

Open the URL shown in the terminal (typically http://localhost:8501).

### Using the UI
- Choose a prompt from “Demo Prompt Library” (or write your own) and click “Generate Output (Gemini)”.
- Optionally leave “Apply safety wrapper” enabled to reduce safety blocks.
- Tune sliders:
  - Search results (Valyu)
  - Evidence clusters / Output clusters
  - Edge similarity threshold
  - Grounded / Weak thresholds
- Choose layer orientation: vertical (rows) or horizontal (columns)
- Click “Build Graph”
- Drag nodes to explore; click “Reset layers” to re-snap them to their rows/columns

The legend explains node colors and edge styles. Edge width increases with similarity; labels show the score.

When generation completes, you’ll also see:
- finish_reason (e.g., STOP or SAFETY)
- safety category ratings (if provided by Gemini)

## Run the CLI
The CLI builds a static HTML file (graph.html) for a sample prompt/response pair.

```zsh
# From hack2/ (venv activated)
python main.py
# => Wrote graph to graph.html
open graph.html  # macOS convenience
```

## Configuration notes
- Embedding model: all-MiniLM-L6-v2 (Sentence-Transformers)
- Similarity: cosine via dot product on L2-normalized embeddings
- Clustering: scikit-learn KMeans for evidence and output sentences
- Thresholds: configurable in UI; in CLI they’re defined near the top of main.py

## Performance benchmarking (optional)
`Performance_benchmarking.py` lets you:
- Retrieve sources from Valyu
- Build a TF‑IDF similarity graph and prune to one node per connected component
- Compare a local “token usage” estimate between ALL vs PRUNED sources

Run it with Streamlit:

```zsh
# From hack2/ (venv activated)
streamlit run Performance_benchmarking.py
```

Notes:
- This module currently focuses on local TF‑IDF similarity and a rough token estimate. You can adapt it to call Gemini if desired (ensure your GEMINI_API_KEY is configured and avoid placing secrets directly in source files).
- If you see any hard-coded keys in the file, move them into your .env or shell environment.

## Troubleshooting
- “Missing VALYU_API_KEY or GEMINI_API_KEY”: set them in your shell or .env, then restart
- Port 8501 already in use: stop the prior Streamlit process or run with `--server.port 8502`
- NLTK punkt download: first run will fetch it; internet access required once
- SSL/urllib3 OpenSSL warning (macOS): benign locally, can be ignored
- scikit-learn/NumPy matmul warnings: cosmetic for this workflow
- Blank/partial graph: ensure search returned results and thresholds aren’t set too high
- Gemini “finish_reason=2” (SAFETY/blocked): use the Demo Prompt Library, keep “Apply safety wrapper” on, and phrase prompts as high‑level, educational, and non‑actionable. Avoid medical/legal/security instructions or words like “bypass”, “exploit”, etc.
- `module importlib.metadata has no attribute packages_distributions`: we include `importlib-metadata` in requirements for Python 3.9. Ensure dependencies are installed and up to date.

## Security
- Don’t commit your .env or share your API keys (Valyu or Gemini)
- The app sends your prompt to Valyu’s search endpoint to retrieve public evidence
- Avoid hard-coding secrets in source files; prefer environment variables or .env

## Acknowledgements
- Valyu (search API)
- Sentence-Transformers, scikit-learn, NetworkX, PyVis, Streamlit

## License
This project inherits the repository’s license. If none is specified, treat it as “All rights reserved” or add a LICENSE file of your choice.
