import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import requests
import json

# ------------------------------
# GEMINI API CONFIG
# ------------------------------
GEMINI_API_KEY = "AIzaSyCld9tkQmFhLf17Wnmla1MU0pR0bVZs2RM"
GEMINI_MODEL = "gemini-2.5-flash"  # or another available Gemini model

import requests
import requests
import streamlit as st
import requests
import streamlit as st

def gemini_reasoning_answer_debug(
    query, 
    sources_texts, 
    max_output_tokens=500, 
    max_source_chars=1000, 
    max_total_chars=5000,
    temperature=0.0
):
    """
    Generate an answer using Gemini API from a list of source texts and a question.
    
    Parameters:
        query (str): The user question.
        sources_texts (List[str]): List of source strings.
        max_output_tokens (int): Maximum tokens for Gemini to generate.
        max_source_chars (int): Maximum characters to take from each source (truncation).
        max_total_chars (int): Maximum combined characters for all sources.
        temperature (float): Temperature for generation (0 for deterministic).
    
    Returns:
        dict: {"answer": str, "tokens": int}
    """
    if not sources_texts:
        return {"answer": "No sources available for reasoning.", "tokens": 0}

    # 1️⃣ Truncate individual sources
    truncated_sources = [s[:max_source_chars] for s in sources_texts]

    # 2️⃣ Combine sources safely
    combined_text = "\n".join(truncated_sources)[:max_total_chars]

    # 3️⃣ Construct Gemini API payload
    payload = {
        "system_instruction": {"parts": [{"text": "You are a helpful assistant."}]},
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            f"Using the sources below, answer the question concisely.\n\n"
                            f"Sources:\n{combined_text}\n\nQuestion:\n{query}"
                        )
                    }
                ],
            }
        ],
        "generationConfig": {
            "maxOutputTokens": max_output_tokens,
            "temperature": temperature
        }
    }

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}

    # Recursive text extraction for Gemini's nested content
    def extract_text_from_content(content):
        """
        Recursively extract all 'text' fields from Gemini candidate content,
        including nested 'parts' and nested 'content'.
        """
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            return "".join(extract_text_from_content(c) for c in content)
        elif isinstance(content, dict):
            text_part = extract_text_from_content(content.get("text"))
            content_part = extract_text_from_content(content.get("content"))
            parts_part = extract_text_from_content(content.get("parts"))
            return text_part + content_part + parts_part
        else:
            return ""

    try:
        response = requests.post(url, headers=headers, json=payload)

        # Streamlit debug info
        st.text(f"Status code: {response.status_code}")
        st.text(f"Partial response: {response.text[:500]}")  # first 500 chars
        response.raise_for_status()

        data = response.json()
        st.json(data)  # full Gemini response for debugging

        candidates = data.get("candidates", [])
        if not candidates:
            return {"answer": "No answer returned by Gemini.", "tokens": 0}

        candidate = candidates[0]
        answer_text = extract_text_from_content(candidate)

        if not answer_text.strip():
            return {"answer": "Gemini returned empty answer.", "tokens": 0}

        tokens_used = len(answer_text.split())  # approximate token count
        return {"answer": answer_text, "tokens": tokens_used}

    except Exception as e:
        st.error(f"Gemini API call failed: {e}")
        if 'response' in locals():
            st.text(f"Response content: {response.text[:500]}")
        return {"answer": f"Gemini API call failed: {e}", "tokens": 0}

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def local_reasoning_answer_debug(query, sources_texts, max_sources=5):
    """
    Local reasoning that only returns the approximate token count of the most relevant sources.
    
    Parameters:
        query (str): The user question.
        sources_texts (List[str]): List of source strings.
        max_sources (int): Maximum number of top sources to consider.
        
    Returns:
        dict: {"tokens": int}
    """
    if not sources_texts:
        return {"tokens": 0}

    # Compute TF-IDF vectors for sources + query
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(sources_texts + [query])

    # Cosine similarity of query to each source
    query_vec = tfidf_matrix[-1]
    sources_vec = tfidf_matrix[:-1]
    sim_scores = cosine_similarity(query_vec, sources_vec).flatten()

    # Rank sources
    top_indices = sim_scores.argsort()[::-1][:max_sources]

    # Sum approximate tokens of top sources (split by spaces)
    tokens_used = sum(len(sources_texts[i].split()) for i in top_indices)

    return {"tokens": tokens_used}
# ------------------------------
# TF-IDF similarity graph
# ------------------------------
def build_similarity_graph(texts, threshold=0.30):
    if not texts:
        return nx.Graph()
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(texts)
    sim_matrix = cosine_similarity(tfidf)
    n = sim_matrix.shape[0]

    G = nx.Graph()
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        for j in range(i+1, n):
            if sim_matrix[i, j] >= threshold:
                G.add_edge(i, j, weight=float(sim_matrix[i, j]))
    return G

# ------------------------------
# Pick one node per connected component + all unconnected
# ------------------------------
def prune_sources_by_components(G, sources):
    if G.number_of_nodes() == 0:
        return sources, list(range(len(sources)))

    components = list(nx.connected_components(G))
    selected_indices = []

    for comp in components:
        comp = list(comp)
        chosen = random.choice(comp)
        selected_indices.append(chosen)

    selected_indices = list(set(selected_indices))
    pruned_sources = [sources[i] for i in selected_indices]
    return pruned_sources, selected_indices


# ------------------------------
# STREAMLIT UI
# ------------------------------
st.title("Valyu Search + Gemini Reasoning (All vs Pruned Sources)")

query = st.text_input("Enter your question:")

if st.button("Run Comparison"):
    # 1️⃣ Search Valyu
    from valyu import Valyu
    VALYU_API_KEY = "b99mDE2Cm42BdzZloqbvV2H1Wkwr1ga4r1pS9q9b"
    client = Valyu(api_key=VALYU_API_KEY)

    with st.spinner("Searching Valyu..."):
        try:
            result = client.search(query=query, max_num_results=20)
            sources_raw = result.results
        except Exception as e:
            st.error(f"Search failed: {e}")
            sources_raw = []

    sources = []
    for i, src in enumerate(sources_raw):
        sources.append({
            "text": getattr(src, "content", ""),  
            "score": getattr(src, "relevance_score", 0.0),
            "id": getattr(src, "id", f"no-id-{i}"),
            "url": getattr(src, "url", "No URL")
        })

    if not sources:
        st.warning("No sources returned. Using mock sources for testing.")
        sources = [
            {"text": "This is document 1 about AI.", "score": 0.9, "id": "doc1", "url": "url1"},
            {"text": "This is document 2 about machine learning.", "score": 0.8, "id": "doc2", "url": "url2"},
            {"text": "Document 3 discusses neural networks.", "score": 0.85, "id": "doc3", "url": "url3"},
        ]

    st.subheader("🔍 Retrieved Sources")
    st.markdown(f"Found {len(sources)} sources.")
    for s in sources:
        st.markdown(f"- **{s['id']}** | score: {s['score']:.2f}  \nURL: {s['url']}")

    # 3️⃣ Build TF-IDF graph
    texts = [s["text"] for s in sources]
    G = build_similarity_graph(texts, threshold=0.30)

    st.subheader("📈 TF-IDF Similarity Graph")
    fig, ax = plt.subplots(figsize=(8,6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, font_size=8)
    st.pyplot(fig)

    # 4️⃣ Gemini reasoning with all sources
    st.subheader("🧠 Gemini Reasoning with ALL sources")
    full_answer = local_reasoning_answer_debug(query, texts)
    st.markdown("#### Approximate Token Usage")
    st.write(f"Tokens: {full_answer['tokens']}")

    # 5️⃣ Prune sources
    pruned_sources, picked_indices = prune_sources_by_components(G, sources)
    pruned_texts = [s["text"] for s in pruned_sources]

    st.subheader("✨ Pruned Sources (1 per connected component + unconnected)")
    for s in pruned_sources:
        st.markdown(f"- **{s['id']}** | score: {s['score']:.2f}  \nURL: {s['url']}")

    # 6️⃣ Gemini reasoning with pruned sources
    st.subheader("🧠 Gemini Reasoning with PRUNED sources")
    pruned_answer = local_reasoning_answer_debug(query, pruned_texts)
    st.markdown("#### Answer")
    st.markdown("#### Approximate Token Usage")
    st.write(f"Tokens: {pruned_answer['tokens']}")
