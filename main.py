import json
import os
import requests
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize

# Suppress tokenizers parallelism warning to avoid noisy logs
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import networkx as nx
from pyvis.network import Network
from valyu import Valyu
from dotenv import load_dotenv

# -------------------------------------------------------
# 0. Setup
# -------------------------------------------------------

load_dotenv()

# Ensure NLTK punkt is available (avoid repeated downloads and incorrect resource name)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

EMBED_MODEL = "all-MiniLM-L6-v2"
SIM_THRESHOLD_EDGE = 0.50
SIM_THRESHOLD_GROUNDED = 0.75
SIM_THRESHOLD_WEAK = 0.45

# -------------------------------------------------------
# 1. Valyu Search API (placeholder)
# -------------------------------------------------------

def run_valyu_search(prompt_text: str):
    """
    Runs Valyu search with the prompt provided
    """
    valyu_api_key = os.getenv("VALYU_API_KEY", "")
    if not valyu_api_key:
        raise RuntimeError("Missing VALYU_API_KEY. Set it in environment or .env file.")
    valyu = Valyu(api_key=valyu_api_key)

    response = valyu.search(
        query=prompt_text,
        max_num_results=20,          
    )
    return response.results


# -------------------------------------------------------
# 2. Chunking
# -------------------------------------------------------

def chunk_text(text: str):
    sentences = sent_tokenize(text)
    return [s.strip() for s in sentences if len(s.strip()) > 0]


# -------------------------------------------------------
# 3. Embedding
# -------------------------------------------------------

def embed_chunks(model, chunks):
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings


# -------------------------------------------------------
# 4. Similarity
# -------------------------------------------------------

def cosine_sim_matrix(A, B):
    return np.dot(A, B.T)


# -------------------------------------------------------
# 5. Classification of output chunks
# -------------------------------------------------------

def classify_similarity(max_sim):
    if max_sim >= SIM_THRESHOLD_GROUNDED:
        return "grounded"
    elif max_sim >= SIM_THRESHOLD_WEAK:
        return "weak"
    else:
        return "hallucinated"


# -------------------------------------------------------
# 6. Graph Building
# -------------------------------------------------------

def build_graph(prompt_chunks, output_chunks, search_chunks,
        sim_output_prompt, sim_output_search):
    G = nx.Graph()

    # Add nodes
    for c in prompt_chunks:
        # Treat prompts as inputs in the visualization
        G.add_node(c["id"], text=c["text"], type="input")

    for c in output_chunks:
        G.add_node(c["id"], text=c["text"], type="output", status=c["status"])

    for c in search_chunks:
        G.add_node(c["id"], text=c["text"], type="evidence", cluster=c["cluster"])

    # Add edges: output -> input
    for oi, oc in enumerate(output_chunks):
        for pi, pc in enumerate(prompt_chunks):
            sim = sim_output_prompt[oi][pi]
            if sim >= SIM_THRESHOLD_EDGE:
                # add label and edge type for clear visualization later
                G.add_edge(oc["id"], pc["id"], weight=float(sim), label=f"{float(sim):.2f}", etype="output_input")

    # Add edges: output -> search evidence
    for oi, oc in enumerate(output_chunks):
        for si, sc in enumerate(search_chunks):
            sim = sim_output_search[oi][si]
            if sim >= SIM_THRESHOLD_EDGE:
                G.add_edge(oc["id"], sc["id"], weight=float(sim), cluster=sc["cluster"], label=f"{float(sim):.2f}", etype="output_evidence")

    return G

# -------------------------------------------------------
# 7. Graph Visualization
# -------------------------------------------------------

def visualize_graph(G, out_file="graph.html"):
    # Directed graph for clarity; we'll enforce a hierarchical (layered) layout
    net = Network(height="850px", width="100%", directed=True, notebook=False)
    net.from_nx(G)

    # Helper for readable labels
    def trunc(text, n=80):
        return text if len(text) <= n else text[: n - 1] + "…"

    # Palette for evidence clusters
    cluster_palette = [
        "#9e9ac8", "#fdae6b", "#74c476", "#6baed6", "#fb6a4a",
        "#c7e9c0", "#9ecae1", "#fdd0a2", "#a1d99b", "#bcbddc",
    ]

    # Map node id -> type for edge styling
    node_type = {n["id"]: n.get("type") for n in net.nodes}

    # Style nodes with levels, labels, titles, colors
    for node in net.nodes:
        ntype = node.get("type")
        text = node.get("text", "")
        node_id = node.get("id")

        if ntype == "input":
            node["level"] = 1
            node["shape"] = "box"
            node["color"] = "#6baed6"
            node["label"] = f"Input {node_id}:\n" + trunc(text)
            node["title"] = f"Input {node_id}\n\n" + text
        elif ntype == "evidence":
            node["level"] = 2
            node["shape"] = "ellipse"
            cluster = node.get("cluster")
            if isinstance(cluster, int) and cluster >= 0:
                node["color"] = cluster_palette[cluster % len(cluster_palette)]
                node["group"] = f"cluster-{cluster}"
                node["label"] = f"Evidence c{cluster}:\n" + trunc(text)
                node["title"] = f"Evidence {node_id} (cluster {cluster})\n\n" + text
            else:
                node["color"] = "#9e9ac8"
                node["label"] = "Evidence:\n" + trunc(text)
                node["title"] = f"Evidence {node_id}\n\n" + text
        elif ntype == "output":
            node["level"] = 3
            node["shape"] = "box"
            status = node.get("status", "")
            if status == "grounded":
                node["color"] = "#31a354"
            elif status == "weak":
                node["color"] = "#fd8d3c"
            else:
                node["color"] = "#de2d26"
            node["label"] = f"Output {node_id} ({status}):\n" + trunc(text)
            node["title"] = f"Output {node_id} [{status}]\n\n" + text

    # Manually place nodes in three clear layers and disable physics.
    # Stagger nodes within each layer; spread evidence by cluster to reduce edge overlap.
    layers = {1: [], 2: [], 3: []}
    for n in net.nodes:
        lvl = n.get("level")
        if lvl in layers:
            layers[lvl].append(n)

    # Determine horizontal spacing and centering
    base_spacing = 260
    y_for_level = {1: 0, 2: 320, 3: 640}

    # Get clusters for evidence layer to create columns
    evidence_nodes = layers[2]
    clusters = sorted({n.get("cluster") for n in evidence_nodes if isinstance(n.get("cluster"), int)})
    cluster_index = {c: i for i, c in enumerate(clusters)}
    num_cluster_cols = max(1, len(clusters))

    # Compute max counts for centering per layer
    max_counts = {
        1: max(1, len(layers[1])),
        2: max(1, len(evidence_nodes)),
        3: max(1, len(layers[3]))
    }

    def centered_start(count, spacing):
        return -((count - 1) * spacing) / 2.0

    # Place prompt and output with slight zig-zag to avoid straight lines
    for lvl in (1, 3):
        nodes = layers[lvl]
        spacing = base_spacing
        start_x = centered_start(len(nodes), spacing)
        for idx, n in enumerate(nodes):
            jitter = ((-1) ** idx) * 40  # alternate left/right for slight zig-zag
            n["x"] = start_x + idx * spacing + jitter
            n["y"] = y_for_level[lvl]
            n["fixed"] = True
            n["physics"] = False

    # Place evidence by cluster columns with per-cluster local spacing
    if num_cluster_cols > 0:
        col_spacing = max(base_spacing, 220)
        col_start_x = centered_start(num_cluster_cols, col_spacing)
        # bucket evidence nodes per cluster
        bucket = {c: [] for c in clusters} if clusters else {0: evidence_nodes}
        if clusters:
            for n in evidence_nodes:
                c = n.get("cluster")
                if c in bucket:
                    bucket[c].append(n)
        else:
            bucket[0] = evidence_nodes

        for c, nodes in bucket.items():
            col_idx = cluster_index.get(c, 0)
            x_center = col_start_x + col_idx * col_spacing
            # local vertical staggering within layer to reduce overlap
            local_spacing = 90
            local_start = -((len(nodes) - 1) * local_spacing) / 2.0
            for i, n in enumerate(nodes):
                n["x"] = x_center + ((-1) ** i) * 35  # small horizontal jitter
                n["y"] = y_for_level[2] + local_start + i * local_spacing
                n["fixed"] = True
                n["physics"] = False

    # Turn off physics globally to keep layout stable
    try:
        net.toggle_physics(False)
    except Exception:
        pass

    # Label and color edges, show arrows and similarity labels
    for edge in net.edges:
        src_type = node_type.get(edge["from"]) 
        dst_type = node_type.get(edge["to"])   
        etype = edge.get("etype")

        if not edge.get("label") and edge.get("weight") is not None:
            edge["label"] = f"{float(edge['weight']):.2f}"

        # Edge width by similarity for visual strength
        weight = float(edge.get("weight") or 0.0)
        edge["width"] = max(1, min(6, int(1 + 5 * weight)))

        if etype == "output_input" or (src_type == "output" and dst_type == "input"):
            edge["color"] = {"color": "#3182bd"}  # blue
            edge["arrows"] = "to"
            edge["dashes"] = True
            edge["smooth"] = {"type": "curvedCCW", "roundness": 0.25}
        elif etype == "output_evidence" or (src_type == "output" and dst_type == "evidence"):
            edge["color"] = {"color": "#e6550d"}  # orange
            edge["arrows"] = "to"
            edge["smooth"] = {"type": "curvedCW", "roundness": 0.25}
        else:
            edge["color"] = {"color": "#999999"}

        edge["font"] = {"align": "middle", "size": 12, "background": "#ffffff"}

    net.show(out_file, notebook=False)
    print(f"Wrote graph to {out_file}")


# -------------------------------------------------------
# Helper utilities for clustering visualization
# -------------------------------------------------------

def l2_normalize_rows(M: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    return M / denom


def top_representatives(embeddings: np.ndarray, indices: np.ndarray, center: np.ndarray, topn: int = 2):
    if len(indices) == 0:
        return []
    center_norm = center / (np.linalg.norm(center) + 1e-12)
    sims = embeddings[indices] @ center_norm
    order = np.argsort(-sims)
    top_idx = indices[order[: min(topn, len(order))]]
    return top_idx.tolist()


# -------------------------------------------------------
# 8. Main pipeline
# -------------------------------------------------------

def main():
    # ---------------------------------------------------
    # Input: prompt + model output
    # ---------------------------------------------------
    full_prompt = """
    Explain the risks of AI systems making decisions in high-stakes domains and
    how external evidence can be used to improve interpretability.
    """

    model_output = """
    AI systems in healthcare can misdiagnose conditions without clear reasoning.
    Sometimes they rely on patterns not visible in the training data.
    Using external corpora allows us to ground model claims in real evidence.
    """

    # ---------------------------------------------------
    # Step 1: Single Valyu Search Call
    # ---------------------------------------------------
    search_results = run_valyu_search(full_prompt)

    # ---------------------------------------------------
    # Step 2: Chunking
    # ---------------------------------------------------
    prompt_chunks = [{"id": f"p{i}", "text": t}
                     for i, t in enumerate(chunk_text(full_prompt))]

    output_chunks = [{"id": f"o{i}", "text": t}
                     for i, t in enumerate(chunk_text(model_output))]

    search_chunks = []
    for i, r in enumerate(search_results):
        # Valyu returns SearchResult objects; use attribute access instead of dict subscripting
        sentences = chunk_text(r.content)
        for j, s in enumerate(sentences):
            search_chunks.append({"id": f"s{i}_{j}", "text": s, "cluster": None})

    # ---------------------------------------------------
    # Step 3: Embedding
    # ---------------------------------------------------
    model = SentenceTransformer(EMBED_MODEL)

    emb_prompt = embed_chunks(model, prompt_chunks)
    emb_output = embed_chunks(model, output_chunks)
    emb_search = embed_chunks(model, search_chunks)

    # ---------------------------------------------------
    # Step 4: Cluster outputs and search evidence to reduce node count
    # ---------------------------------------------------
    max_k = 25
    # Evidence clustering
    k_search = min(max_k, max(1, len(search_chunks)))
    if k_search > 0 and len(search_chunks) > k_search:
        km_search = KMeans(n_clusters=k_search, n_init="auto")
        labels_search = km_search.fit_predict(emb_search)
        centers_search = l2_normalize_rows(km_search.cluster_centers_)
    else:
        # No clustering needed; treat each as its own center
        labels_search = np.arange(len(search_chunks))
        centers_search = emb_search
        k_search = len(search_chunks)

    # Output clustering
    k_out = min(max_k, max(1, len(output_chunks)))
    if k_out > 0 and len(output_chunks) > k_out:
        km_out = KMeans(n_clusters=k_out, n_init="auto")
        labels_out = km_out.fit_predict(emb_output)
        centers_out = l2_normalize_rows(km_out.cluster_centers_)
    else:
        labels_out = np.arange(len(output_chunks))
        centers_out = emb_output
        k_out = len(output_chunks)

    # ---------------------------------------------------
    # Step 5: Create clustered nodes with representative text
    # ---------------------------------------------------
    # Evidence cluster nodes
    evidence_nodes = []
    for cidx in range(k_search):
        member_idx = np.where(labels_search == cidx)[0]
        if len(member_idx) == 0:
            continue
        reps = top_representatives(emb_search, member_idx, centers_search[cidx], topn=2)
        rep_texts = [search_chunks[i]["text"] for i in reps]
        summary = (" | ").join(rep_texts) if rep_texts else search_chunks[member_idx[0]]["text"]
        evidence_nodes.append({
            "id": f"sC{cidx}",
            "text": f"(n={len(member_idx)}) " + summary,
            "cluster": int(cidx)
        })

    # Output cluster nodes
    output_nodes = []
    for cidx in range(k_out):
        member_idx = np.where(labels_out == cidx)[0]
        if len(member_idx) == 0:
            continue
        reps = top_representatives(emb_output, member_idx, centers_out[cidx], topn=2)
        rep_texts = [output_chunks[i]["text"] for i in reps]
        summary = (" | ").join(rep_texts) if rep_texts else output_chunks[member_idx[0]]["text"]
        output_nodes.append({
            "id": f"oC{cidx}",
            "text": f"(n={len(member_idx)}) " + summary,
            "status": ""
        })

    # ---------------------------------------------------
    # Step 6: Similarity between cluster centroids
    # ---------------------------------------------------
    sim_output_prompt = cosine_sim_matrix(centers_out, emb_prompt)
    sim_output_search = cosine_sim_matrix(centers_out, centers_search)

    # Classify output clusters based on centroid similarity
    for i, on in enumerate(output_nodes):
        max_p = np.max(sim_output_prompt[i]) if sim_output_prompt.shape[1] > 0 else 0.0
        max_s = np.max(sim_output_search[i]) if sim_output_search.shape[1] > 0 else 0.0
        on["status"] = classify_similarity(float(max(max_p, max_s)))

    # ---------------------------------------------------
    # Step 7: Build graph (cluster-level)
    # ---------------------------------------------------
    G = build_graph(prompt_chunks, output_nodes, evidence_nodes,
                    sim_output_prompt, sim_output_search)

    # ---------------------------------------------------
    # Step 8: Visualize
    # ---------------------------------------------------
    visualize_graph(G)


# -------------------------------------------------------
# Entry point
# -------------------------------------------------------
if __name__ == "__main__":
    main()
