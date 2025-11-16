# Grounded Output Explorer

An interactive Streamlit app and CLI that builds a three-layer “grounding graph” for a model response:
- Inputs: chunked prompt sentences
- Evidence: Valyu search results (clustered)
- Outputs: chunked model response (clustered)

Edges show cosine similarities between output clusters and inputs/evidence. The app helps you quickly see which parts of a response are grounded vs weak vs hallucinated.

## Features
- Layered, playable graph (drag nodes, reset layers)
- Vertical or horizontal layer orientation
- Output and evidence clustering to keep graphs small (5–30 clusters)
- Similarity-labeled edges with width scaled by score
- Node colors for status: grounded (green), weak (orange), hallucinated (red)
- Hover tooltips and a built-in legend

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
  main.py             # CLI pipeline: builds graph.html
  streamlit_app.py    # Streamlit UI: interactive graph
  requirements.txt    # Python dependencies
  README.md           # This file
```

## Prerequisites
- macOS (tested locally) or any OS with Python 3.9+
- Python 3.9 recommended (matches the bundled virtualenv in this repo)
- A Valyu API key

## Setup
Create a virtual environment (recommended) and install dependencies:

```zsh
# From the repo root
cd "hack2"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Provide your Valyu API key via environment or a .env file:

```zsh
# Option A: export in your shell session
export VALYU_API_KEY="<your-key-here>"

# Option B: create a .env file in hack2/
cat > .env << 'EOF'
VALYU_API_KEY=<your-key-here>
EOF
```

## Run the Streamlit app (recommended)

```zsh
# From hack2/ (venv activated)
streamlit run streamlit_app.py
```

Open the URL shown in the terminal (typically http://localhost:8501).

### Using the UI
- Enter your prompt and the model’s output in the sidebar
- Tune sliders:
  - Search results (Valyu)
  - Evidence clusters / Output clusters
  - Edge similarity threshold
  - Grounded / Weak thresholds
- Choose layer orientation: vertical (rows) or horizontal (columns)
- Click “Build Graph”
- Drag nodes to explore; click “Reset layers” to re-snap them to their rows/columns

The legend explains node colors and edge styles. Edge width increases with similarity; labels show the score.

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

## Troubleshooting
- “Missing VALYU_API_KEY”: set it in your shell or .env, then restart
- Port 8501 already in use: stop the prior Streamlit process or run with `--server.port 8502`
- NLTK punkt download: first run will fetch it; internet access required once
- SSL/urllib3 OpenSSL warning (macOS): benign locally, can be ignored
- scikit-learn/NumPy matmul warnings: cosmetic for this workflow
- Blank/partial graph: ensure search returned results and thresholds aren’t set too high

## Security
- Don’t commit your .env or share your VALYU_API_KEY
- The app sends your prompt to Valyu’s search endpoint to retrieve public evidence

## Acknowledgements
- Valyu (search API)
- Sentence-Transformers, scikit-learn, NetworkX, PyVis, Streamlit

## License
This project inherits the repository’s license. If none is specified, treat it as “All rights reserved” or add a LICENSE file of your choice.
