import os
import json
import tempfile
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import nltk
from nltk.tokenize import sent_tokenize
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from valyu import Valyu
from pyvis.network import Network
import networkx as nx

# -----------------------------
# Setup
# -----------------------------
load_dotenv()

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_MODEL = "all-MiniLM-L6-v2"

# -----------------------------
# Helpers (pipeline)
# -----------------------------

def get_valyu():
    key = os.getenv("VALYU_API_KEY", "")
    if not key:
        st.error("VALYU_API_KEY not found. Add it to your environment or a .env file.")
        st.stop()
    return Valyu(api_key=key)


def run_valyu_search(query: str, max_results: int = 20):
    client = get_valyu()
    resp = client.search(query=query, max_num_results=max_results)
    return resp.results


def chunk_text(text: str):
    sentences = sent_tokenize(text)
    return [s.strip() for s in sentences if len(s.strip()) > 0]


def embed_chunks(model, chunks):
    texts = [c["text"] for c in chunks]
    return model.encode(texts, normalize_embeddings=True)


def cosine_sim_matrix(A, B):
    return np.dot(A, B.T)


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


def classify_similarity(max_sim, t_grounded: float, t_weak: float):
    if max_sim >= t_grounded:
        return "grounded"
    elif max_sim >= t_weak:
        return "weak"
    else:
        return "hallucinated"


def build_graph(input_nodes, output_nodes, evidence_nodes,
                sim_output_input, sim_output_evidence,
                sim_edge_threshold: float):
    G = nx.Graph()

    for c in input_nodes:
        G.add_node(c["id"], text=c["text"], type="input")
    for c in output_nodes:
        G.add_node(c["id"], text=c["text"], type="output", status=c.get("status", ""))
    for c in evidence_nodes:
        G.add_node(c["id"], text=c["text"], type="evidence", cluster=c.get("cluster"))

    # output -> input
    for oi, oc in enumerate(output_nodes):
        for ii, ic in enumerate(input_nodes):
            sim = sim_output_input[oi][ii]
            if sim >= sim_edge_threshold:
                G.add_edge(oc["id"], ic["id"], weight=float(sim), label=f"{float(sim):.2f}", etype="output_input")

    # output -> evidence
    for oi, oc in enumerate(output_nodes):
        for si, sc in enumerate(evidence_nodes):
            sim = sim_output_evidence[oi][si]
            if sim >= sim_edge_threshold:
                G.add_edge(oc["id"], sc["id"], weight=float(sim), cluster=sc.get("cluster"), label=f"{float(sim):.2f}", etype="output_evidence")

    return G


def visualize_graph(G, orientation: str = "vertical"):
    net = Network(height="850px", width="100%", directed=True, notebook=False)
    net.from_nx(G)

    def trunc(text, n=80):
        return text if len(text) <= n else text[: n - 1] + "…"

    cluster_palette = [
        "#9e9ac8", "#fdae6b", "#74c476", "#6baed6", "#fb6a4a",
        "#c7e9c0", "#9ecae1", "#fdd0a2", "#a1d99b", "#bcbddc",
    ]

    node_type = {n["id"]: n.get("type") for n in net.nodes}

    for node in net.nodes:
        ntype = node.get("type")
        text = node.get("text", "")
        node_id = node.get("id")

        # Small, minimal nodes with short labels and colors only
        node["shape"] = "dot"
        node["size"] = 12

        if ntype == "input":
            node["color"] = "#6baed6"
            node["label"] = "Input"
            node["title"] = "Input\n\n" + trunc(text)
        elif ntype == "evidence":
            cluster = node.get("cluster")
            if isinstance(cluster, int) and cluster >= 0:
                node["color"] = cluster_palette[cluster % len(cluster_palette)]
                node["group"] = f"cluster-{cluster}"
                node["label"] = "Evidence"
                node["title"] = f"Evidence (cluster {cluster})\n\n" + trunc(text, 240)
            else:
                node["color"] = "#9e9ac8"
                node["label"] = "Evidence"
                node["title"] = "Evidence\n\n" + trunc(text, 240)
        elif ntype == "output":
            status = node.get("status", "")
            if status == "grounded":
                node["color"] = "#31a354"
            elif status == "weak":
                node["color"] = "#fd8d3c"
            else:
                node["color"] = "#de2d26"
            node["label"] = "Output"
            node["title"] = f"Output [{status}]\n\n" + trunc(text, 240)

    # Enable physics and dragging for a smooth, playable graph
    net.set_options("""
    {
      "interaction": { "dragNodes": true, "dragView": true },
      "physics": {
        "enabled": true,
        "stabilization": { "enabled": true, "iterations": 300 },
        "barnesHut": { "gravitationalConstant": -8000, "springLength": 160, "springConstant": 0.04 }
      },
      "nodes": { "font": { "size": 10 } },
      "edges": { "smooth": { "enabled": true, "type": "dynamic" } }
    }
    """)

    # Enforce layered layout with interactivity. Choose rows (vertical) or columns (horizontal).
    buckets = {"input": [], "evidence": [], "output": []}
    for n in net.nodes:
        t = n.get("type")
        if t in buckets:
            buckets[t].append(n)

    def centered_start(count, spacing):
        return -((count - 1) * spacing) / 2.0

    base_spacing = 200
    if orientation == "horizontal":
        # Inputs (left), Evidence (center), Outputs (right)
        x_for_type = {"input": -600, "evidence": 0, "output": 600}
        # Spread vertically within column; fix X, allow Y dragging
        for t, nodes in buckets.items():
            if not nodes:
                continue
            start_y = centered_start(len(nodes), base_spacing)
            for idx, n in enumerate(nodes):
                n["x"] = x_for_type.get(t, 0)
                n["y"] = start_y + idx * base_spacing
                n["fixed"] = {"x": True, "y": False}
                n["physics"] = True
    else:
        # Default vertical: Inputs (top), Evidence (middle), Outputs (bottom)
        y_for_type = {"input": 0, "evidence": 300, "output": 600}
        # Spread horizontally within row; fix Y, allow X dragging
        for t, nodes in buckets.items():
            if not nodes:
                continue
            start_x = centered_start(len(nodes), base_spacing)
            for idx, n in enumerate(nodes):
                n["y"] = y_for_type.get(t, 0)
                n["x"] = start_x + idx * base_spacing
                n["fixed"] = {"x": False, "y": True}
                n["physics"] = True

    for edge in net.edges:
        src_type = node_type.get(edge["from"]) 
        dst_type = node_type.get(edge["to"])   
        etype = edge.get("etype")

        if not edge.get("label") and edge.get("weight") is not None:
            edge["label"] = f"{float(edge['weight']):.2f}"

        weight = float(edge.get("weight") or 0.0)
        edge["width"] = max(1, min(6, int(1 + 5 * weight)))

        if etype == "output_input" or (src_type == "output" and dst_type == "input"):
            edge["color"] = {"color": "#3182bd"}
            edge["arrows"] = "to"
            edge["dashes"] = True
            edge["smooth"] = {"type": "curvedCCW", "roundness": 0.25}
            edge["title"] = f"Output → Input\nSimilarity: {edge.get('label','')}"
        elif etype == "output_evidence" or (src_type == "output" and dst_type == "evidence"):
            edge["color"] = {"color": "#e6550d"}
            edge["arrows"] = "to"
            edge["smooth"] = {"type": "curvedCW", "roundness": 0.25}
            edge["title"] = f"Output → Evidence\nSimilarity: {edge.get('label','')}"
        else:
            edge["color"] = {"color": "#999999"}

        edge["font"] = {"align": "middle", "size": 10, "background": "#ffffff"}

    # Return HTML content to embed
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
        out_path = tmp.name
    net.show(out_path, notebook=False)
    with open(out_path, "r", encoding="utf-8") as f:
        html = f.read()
    try:
        os.remove(out_path)
    except Exception:
        pass

    # Inject a floating "Reset layers" control that snaps nodes back to layered positions
    type_by_id = {n["id"]: n.get("type") for n in net.nodes}
    inject_tpl = """
            <style>
                #resetLayersBtn {
                    position: fixed; bottom: 30px; right: 30px; z-index: 9999;
                    background: #333; color: #fff; border: none; padding: 10px 14px;
                    border-radius: 6px; cursor: pointer; opacity: 0.85;
                }
                #resetLayersBtn:hover { opacity: 1.0; }
            </style>
            <button id="resetLayersBtn" title="Snap nodes back to Input/Evidence/Output layers">Reset layers</button>
            <script>
                const TYPE_BY_ID = {TYPE_BY_ID_JSON};
                const ORIENTATION = "{ORIENTATION}";
                const Y_FOR_TYPE = {"input": 0, "evidence": 300, "output": 600};
                const X_FOR_TYPE = {"input": -600, "evidence": 0, "output": 600};
                function resetLayers() {
                    try {
                        const ids = Object.keys(TYPE_BY_ID);
                        const posMap = network.getPositions(ids);
                        ids.forEach(id => {
                            const t = TYPE_BY_ID[id];
                            const cur = posMap[id] || {x: 0, y: 0};
                            if (ORIENTATION === 'horizontal') {
                                const targetX = X_FOR_TYPE[t] ?? 0;
                                network.moveNode(id, targetX, cur.y);
                            } else {
                                const targetY = Y_FOR_TYPE[t] ?? 0;
                                network.moveNode(id, cur.x, targetY);
                            }
                        });
                    } catch (e) { console.error('Reset failed', e); }
                }
                document.getElementById('resetLayersBtn').addEventListener('click', resetLayers);
            </script>
        """
    inject_js = inject_tpl.replace("{TYPE_BY_ID_JSON}", json.dumps(type_by_id)).replace("{ORIENTATION}", orientation)

    if "</body>" in html:
        html = html.replace("</body>", inject_js + "</body>")
    else:
        html += inject_js

    return html


def render_legend():
        # HTML legend with colored dots and style descriptions
        return f"""
        <div style='font-size: 14px; line-height: 1.4;'>
            <strong>Legend</strong>
            <div style='margin-top:6px;'>
                <span style='display:inline-block;width:12px;height:12px;border-radius:50%;background:#6baed6;margin-right:6px;'></span>
                Input node
            </div>
            <div>
                <span style='display:inline-block;width:12px;height:12px;border-radius:50%;background:#9e9ac8;margin-right:6px;'></span>
                Evidence node (cluster colors vary)
            </div>
            <div>
                <span style='display:inline-block;width:12px;height:12px;border-radius:50%;background:#31a354;margin-right:6px;'></span>
                Output (grounded)
            </div>
            <div>
                <span style='display:inline-block;width:12px;height:12px;border-radius:50%;background:#fd8d3c;margin-right:6px;'></span>
                Output (weak)
            </div>
            <div>
                <span style='display:inline-block;width:12px;height:12px;border-radius:50%;background:#de2d26;margin-right:6px;'></span>
                Output (hallucinated)
            </div>
            <div style='margin-top:6px;'>
                <span style='border-bottom:2px dashed #3182bd; padding-right:8px; margin-right:6px;'></span>
                Output → Input edge (dashed, blue)
            </div>
            <div>
                <span style='border-bottom:2px solid #e6550d; padding-right:8px; margin-right:6px;'></span>
                Output → Evidence edge (orange)
            </div>
            <div style='margin-top:6px; color:#555;'>
                Edge width scales with similarity; label shows similarity score. Hover nodes/edges for details.
            </div>
        </div>
        """


# -----------------------------
# Streamlit UI
# -----------------------------

def default_texts():
    prompt = (
        "Explain the risks of AI systems making decisions in high-stakes domains and "
        "how external evidence can be used to improve interpretability."
    )
    output = (
        "AI systems in healthcare can misdiagnose conditions without clear reasoning. "
        "Sometimes they rely on patterns not visible in the training data. "
        "Using external corpora allows us to ground model claims in real evidence."
    )
    return prompt, output


st.set_page_config(page_title="Grounding Graph UI", layout="wide")

st.title("Grounded Output Explorer")

with st.sidebar:
    st.header("Inputs")
    in_prompt_default, out_text_default = default_texts()
    full_prompt = st.text_area("Input (Prompt)", value=in_prompt_default, height=150)
    model_output = st.text_area("Output (Model Response)", value=out_text_default, height=150)

    st.header("Settings")
    max_results = st.slider("Search results (Valyu)", 5, 50, 20, step=5)
    k_evidence = st.slider("Evidence clusters", 5, 30, 20, step=1)
    k_output = st.slider("Output clusters", 5, 30, 20, step=1)

    sim_edge = st.slider("Edge similarity threshold", 0.0, 1.0, 0.50, step=0.01)
    t_grounded = st.slider("Grounded threshold", 0.0, 1.0, 0.75, step=0.01)
    t_weak = st.slider("Weak threshold", 0.0, 1.0, 0.45, step=0.01)

    orientation_choice = st.radio(
        "Layer orientation",
        ("Vertical (rows)", "Horizontal (columns)"),
        index=0,
    )
    layer_orientation = "vertical" if orientation_choice.startswith("Vertical") else "horizontal"

    go = st.button("Build Graph")

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Pipeline")
    st.markdown("- Chunk the input/output\n- Retrieve evidence via Valyu\n- Cluster output/evidence\n- Compute similarities\n- Render interactive graph")
    st.caption("Provide VALYU_API_KEY in your environment or a .env file.")

with col_right:
    if go:
        with st.spinner("Running search, embeddings, clustering, and rendering…"):
            # Chunk inputs
            input_chunks = [{"id": f"p{i}", "text": t} for i, t in enumerate(chunk_text(full_prompt))]
            output_chunks = [{"id": f"o{i}", "text": t} for i, t in enumerate(chunk_text(model_output))]

            # Search results -> sentence chunks
            results = run_valyu_search(full_prompt, max_results=max_results)
            evidence_chunks = []
            for i, r in enumerate(results):
                sents = chunk_text(r.content)
                for j, s in enumerate(sents):
                    evidence_chunks.append({"id": f"s{i}_{j}", "text": s, "cluster": None})

            # Embeddings
            model = SentenceTransformer(DEFAULT_MODEL)
            emb_inputs = embed_chunks(model, input_chunks)
            emb_outputs = embed_chunks(model, output_chunks)
            emb_evidence = embed_chunks(model, evidence_chunks)

            # Cluster evidence
            k_se = min(max(1, k_evidence), len(evidence_chunks))
            if len(evidence_chunks) > k_se:
                km_se = KMeans(n_clusters=k_se, n_init="auto")
                labels_se = km_se.fit_predict(emb_evidence)
                centers_se = l2_normalize_rows(km_se.cluster_centers_)
            else:
                labels_se = np.arange(len(evidence_chunks))
                centers_se = emb_evidence
                k_se = len(evidence_chunks)

            # Cluster outputs
            k_so = min(max(1, k_output), len(output_chunks))
            if len(output_chunks) > k_so:
                km_so = KMeans(n_clusters=k_so, n_init="auto")
                labels_so = km_so.fit_predict(emb_outputs)
                centers_so = l2_normalize_rows(km_so.cluster_centers_)
            else:
                labels_so = np.arange(len(output_chunks))
                centers_so = emb_outputs
                k_so = len(output_chunks)

            # Build representative nodes
            evidence_nodes = []
            for cidx in range(k_se):
                member_idx = np.where(labels_se == cidx)[0]
                if len(member_idx) == 0:
                    continue
                reps = top_representatives(emb_evidence, member_idx, centers_se[cidx], topn=2)
                rep_texts = [evidence_chunks[i]["text"] for i in reps]
                summary = (" | ").join(rep_texts) if rep_texts else evidence_chunks[member_idx[0]]["text"]
                evidence_nodes.append({
                    "id": f"sC{cidx}",
                    "text": f"(n={len(member_idx)}) " + summary,
                    "cluster": int(cidx)
                })

            output_nodes = []
            for cidx in range(k_so):
                member_idx = np.where(labels_so == cidx)[0]
                if len(member_idx) == 0:
                    continue
                reps = top_representatives(emb_outputs, member_idx, centers_so[cidx], topn=2)
                rep_texts = [output_chunks[i]["text"] for i in reps]
                summary = (" | ").join(rep_texts) if rep_texts else output_chunks[member_idx[0]]["text"]
                output_nodes.append({
                    "id": f"oC{cidx}",
                    "text": f"(n={len(member_idx)}) " + summary,
                    "status": ""
                })

            # Similarity (centroids)
            sim_out_in = cosine_sim_matrix(centers_so, emb_inputs)
            sim_out_ev = cosine_sim_matrix(centers_so, centers_se)

            # Classify outputs
            for i, on in enumerate(output_nodes):
                max_p = np.max(sim_out_in[i]) if sim_out_in.shape[1] > 0 else 0.0
                max_s = np.max(sim_out_ev[i]) if sim_out_ev.shape[1] > 0 else 0.0
                on["status"] = classify_similarity(float(max(max_p, max_s)), t_grounded, t_weak)

            # Build graph & visualize
            G = build_graph(input_nodes=input_chunks, output_nodes=output_nodes, evidence_nodes=evidence_nodes,
                            sim_output_input=sim_out_in, sim_output_evidence=sim_out_ev,
                            sim_edge_threshold=sim_edge)
            html = visualize_graph(G, orientation=layer_orientation)

        # Show legend above the graph
        st.markdown(render_legend(), unsafe_allow_html=True)
        components.html(html, height=900, scrolling=True)
    else:
        st.info("Adjust settings and click 'Build Graph' to render.")
        # Show legend early so users know what to expect
        st.markdown(render_legend(), unsafe_allow_html=True)
