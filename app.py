# Pediatric Ortho Assistant (optimized)

import os
import json
import hashlib
from pathlib import Path
from typing import List, Tuple

import streamlit as st
import numpy as np
import faiss
import chardet
import textstat
from openai import OpenAI

# -----------------------
# CONFIG
# -----------------------
DATA_DIR = Path("AAOS_Peds")     # <- your text folder
INDEX_DIR = Path("index_store")
INDEX_DIR.mkdir(exist_ok=True, parents=True)
INDEX_PATH = INDEX_DIR / "kb.faiss"
META_PATH  = INDEX_DIR / "kb_meta.json"

EMBED_MODEL = "text-embedding-3-small"  # fast & accurate enough for RAG
CHAT_MODEL  = "gpt-4o-mini"             # fast, high quality

# Secrets: support both st.secrets["openai"]["api_key"] and st.secrets["OPENAI_API_KEY"]
OPENAI_API_KEY = (
    st.secrets.get("openai", {}).get("api_key")
client = OpenAI(api_key=OPENAI_API_KEY))

st.set_page_config(page_title="Pediatric Ortho Assistant", layout="centered")
st.title("🦴 Pediatric Ortho Assistant")
st.markdown("Ask a question about pediatric bone injuries or treatments:")

# -----------------------
# HELPERS
# -----------------------
def read_file_with_detected_encoding(file_path: Path) -> str:
    raw = file_path.read_bytes()
    det = chardet.detect(raw) or {}
    enc = det.get("encoding") or "utf-8"
    try:
        return raw.decode(enc, errors="ignore")
    except Exception:
        return raw.decode("utf-8", errors="ignore")

def paragraph_chunk(text: str, chunk_chars=1000, overlap=120) -> List[str]:
    # Prefer paragraph boundaries; fall back to sliding window
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paras:
        txt = text.strip()
        return [txt[i:i+chunk_chars] for i in range(0, len(txt), chunk_chars - overlap)]
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 2 <= chunk_chars:
            cur = (cur + "\n\n" + p) if cur else p
        else:
            if cur:
                chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)
    # If any chunks are still huge, window them
    final = []
    for c in chunks:
        if len(c) <= chunk_chars + 200:
            final.append(c)
        else:
            for i in range(0, len(c), chunk_chars - overlap):
                final.append(c[i:i+chunk_chars])
    # Light filter to avoid tiny fragments
    return [c for c in final if len(c) >= 200]

def load_documents(folder: Path) -> Tuple[List[str], List[dict]]:
    chunks, metas = [], []
    for p in sorted(folder.glob("*.txt")):
        try:
            txt = read_file_with_detected_encoding(p)
            for i, ch in enumerate(paragraph_chunk(txt, chunk_chars=1100, overlap=140)):
                chunks.append(ch)
                metas.append({"path": str(p), "chunk_idx": i})
        except Exception:
            continue
    return chunks, metas

def dedupe(chunks: List[str], metas: List[dict]) -> Tuple[List[str], List[dict]]:
    seen, out_c, out_m = set(), [], []
    for c, m in zip(chunks, metas):
        h = hashlib.blake2b(c.encode("utf-8"), digest_size=16).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        out_c.append(c)
        out_m.append(m)
    return out_c, out_m

def embed_batched(texts: List[str], batch_size=128) -> np.ndarray:
    vecs = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vecs.extend([d.embedding for d in resp.data])
        st.progress(min(1.0, (i + len(batch)) / max(1, total)), text=f"Embedding {i + len(batch)}/{total}")
    arr = np.array(vecs, dtype="float32")
    # Normalize for cosine similarity
    faiss.normalize_L2(arr)
    return arr

def corpus_signature(folder: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(folder.glob("*.txt")):
        h.update(p.name.encode())
        h.update(p.read_bytes())  # content-based
    return h.hexdigest()

@st.cache_resource(show_spinner=False)
def load_or_build_index() -> Tuple[faiss.Index, List[dict]]:
    sig = corpus_signature(DATA_DIR)
    if INDEX_PATH.exists() and META_PATH.exists():
        try:
            meta = json.loads(META_PATH.read_text())
            if meta.get("signature") == sig:
                index = faiss.read_index(str(INDEX_PATH))
                return index, meta["metas"]
        except Exception:
            pass

    # Rebuild
    chunks, metas = load_documents(DATA_DIR)
    chunks, metas = dedupe(chunks, metas)
    if not chunks:
        raise RuntimeError("No chunks produced from the corpus. Check your .txt files.")

    vecs = embed_batched(chunks, batch_size=128)
    index = faiss.IndexFlatIP(vecs.shape[1])  # inner product with normalized vectors == cosine sim
    index.add(vecs)

    faiss.write_index(index, str(INDEX_PATH))
    META_PATH.write_text(json.dumps({"signature": sig, "metas": metas, "count": len(chunks)}, ensure_ascii=False))
    # Keep chunks in a separate cache slot so we don't bloat META
    st.session_state["_chunks_cache"] = chunks  # lightweight in-memory cache for this session
    return index, metas

def ask_question(question: str, index: faiss.Index, metas: List[dict], chunks: List[str], k=4) -> str:
    # Embed the query
    q = client.embeddings.create(model=EMBED_MODEL, input=question).data[0].embedding
    q = np.array([q], dtype="float32")
    faiss.normalize_L2(q)

    # Search
    _, idxs = index.search(q, k)
    chosen = [chunks[i] for i in idxs[0] if 0 <= i < len(chunks)]
    context = "\n\n---\n\n".join(chosen)

    messages = [
        {"role": "system", "content": (
            "You are a pediatric orthopedic guide. Speak clearly to parents using plain English. "
            "Use short words and short sentences. Use line breaks between ideas. "
            "Use bulleted or numbered lists for treatment steps or symptoms. "
            "Avoid medical terms unless you explain them. "
            "Write at a 5th–6th grade reading level. Keep the tone kind and calm. "
            "Only use the provided context. If the context does not contain the answer, say so. Never guess."
        )},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]

    # Keep retrying until grade target met (with a small cap)
    for _ in range(3):
        r = client.chat.completions.create(model=CHAT_MODEL, messages=messages)
        answer = r.choices[0].message.content.strip()
        grade = textstat.flesch_kincaid_grade(answer)
        if grade <= 7.5:
            break

    disclaimer = (
        "\n\n📢 This is general information, not medical advice. "
        "See https://orthokids.org or https://orthoinfo.aaos.org for more info. "
        "Talk to your child's doctor for specific care or emergencies."
    )
    return answer + disclaimer

# -----------------------
# UI + FLOW
# -----------------------
with st.expander("⚙️ Knowledge Base Controls", expanded=False):
    if st.button("Rebuild index now"):
        # Clear cache + force rebuild on next call
        try:
            INDEX_PATH.unlink(missing_ok=True)
            META_PATH.unlink(missing_ok=True)
        except Exception:
            pass
        st.cache_resource.clear()
        st.success("Cleared. The index will rebuild on the next load.")

question = st.text_input("Enter your question:")

# Load / build index once (cached)
try:
    with st.spinner("🔄 Preparing knowledge base…"):
        index, metas = load_or_build_index()
        # Retrieve chunks either from session (set during build) or re-load minimal memory
        chunks = st.session_state.get("_chunks_cache")
        if chunks is None:
            # Only reload text for display; does not re-embed
            chunks, _ = load_documents(DATA_DIR)
        st.success(f"KB ready. {index.ntotal} chunks indexed.")
except Exception as e:
    st.error(f"⚠️ Failed to prepare KB: {e}")
    chunks, index, metas = [], None, []

if question and index is not None and len(chunks) == index.ntotal and index.ntotal > 0:
    with st.spinner("🤖 Thinking…"):
        answer = ask_question(question, index, metas, chunks, k=4)
        st.markdown("### 💬 Answer")
        st.write(answer)
elif question and (index is None or index.ntotal == 0):
    st.warning("⏳ KB not ready yet. Try rebuilding from the controls above.")
