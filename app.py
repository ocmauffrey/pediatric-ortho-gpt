# PICU Parent Assistant
# Adapted from Pediatric Ortho Assistant
# Knowledge base: validated PICU/critical care parent resources (SCCM, AAP, ICUsteps, etc.)

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
DATA_DIR   = Path("PICU_Resources")   # <- folder with your validated .txt source files
INDEX_DIR  = Path("index_store")
INDEX_DIR.mkdir(exist_ok=True, parents=True)
INDEX_PATH = INDEX_DIR / "kb.faiss"
META_PATH  = INDEX_DIR / "kb_meta.json"

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL  = "gpt-4o-mini"

# Load OpenAI API key from secrets (supports both formats)
if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
    OPENAI_API_KEY = st.secrets["openai"]["api_key"]
else:
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(
    page_title="PICU Family Guide",
    page_icon="💙",
    layout="centered"
)

st.title("💙 PICU Family Guide")
st.markdown(
    "This tool helps families understand what is happening in the Pediatric Intensive Care Unit (PICU). "
    "Ask a question below and get a clear, plain-language answer based on trusted medical resources."
)

# Emotional safety notice at the top — always visible
st.info(
    "**You are not alone.** This tool provides general information only. "
    "For questions about your child's specific condition, treatment, or prognosis, "
    "please speak directly with your care team — they are your best resource. "
    "If you are feeling overwhelmed, ask to speak with our social worker or chaplain.",
    icon="🤝"
)

# -----------------------
# TOPIC GUARDRAILS
# -----------------------
# Questions the assistant should not attempt to answer — routed to care team instead.
OUT_OF_SCOPE_KEYWORDS = [
    "will my child survive", "is my child going to die", "how long does he have",
    "how long does she have", "prognosis", "chances of survival", "make it",
    "odds", "percent chance", "will they recover", "brain dead", "brain death",
    "withdraw", "withdrawing care", "comfort care", "hospice", "end of life",
    "code", "code status", "dnr", "do not resuscitate"
]

def is_out_of_scope(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in OUT_OF_SCOPE_KEYWORDS)

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
    final = []
    for c in chunks:
        if len(c) <= chunk_chars + 200:
            final.append(c)
        else:
            for i in range(0, len(c), chunk_chars - overlap):
                final.append(c[i:i+chunk_chars])
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
    faiss.normalize_L2(arr)
    return arr

def corpus_signature(folder: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(folder.glob("*.txt")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
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

    chunks, metas = load_documents(DATA_DIR)
    chunks, metas = dedupe(chunks, metas)
    if not chunks:
        raise RuntimeError("No chunks produced from the corpus. Check your .txt files in PICU_Resources/.")

    vecs = embed_batched(chunks, batch_size=128)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    faiss.write_index(index, str(INDEX_PATH))
    META_PATH.write_text(json.dumps({"signature": sig, "metas": metas, "count": len(chunks)}, ensure_ascii=False))
    st.session_state["_chunks_cache"] = chunks
    return index, metas

def ask_question(question: str, index: faiss.Index, metas: List[dict], chunks: List[str], k=5) -> str:
    q = client.embeddings.create(model=EMBED_MODEL, input=question).data[0].embedding
    q = np.array([q], dtype="float32")
    faiss.normalize_L2(q)

    _, idxs = index.search(q, k)
    chosen = [chunks[i] for i in idxs[0] if 0 <= i < len(chunks)]
    context = "\n\n---\n\n".join(chosen)

    messages = [
        {"role": "system", "content": (
            "You are a compassionate family guide for the Pediatric Intensive Care Unit (PICU). "
            "You help parents and caregivers understand what is happening during their child's ICU stay. "
            "\n\n"
            "TONE: Always warm, calm, and gentle. These families are under enormous stress. "
            "Acknowledge that this is a hard situation before launching into information. "
            "Never be clinical or cold. Never be falsely cheerful. "
            "\n\n"
            "LANGUAGE: Use plain English at a 5th–6th grade reading level. "
            "Short sentences. Short words. Use line breaks generously between ideas. "
            "Use bulleted lists for steps or equipment explanations. "
            "If you must use a medical term, explain it immediately in plain language. "
            "\n\n"
            "SCOPE: Only use the provided context to answer. "
            "Do NOT speculate about a child's prognosis, chances of recovery, or survival. "
            "Do NOT interpret specific lab values, vitals, or test results. "
            "If the context does not contain the answer, say so clearly and kindly, "
            "and direct the family to ask their care team. Never guess. "
            "\n\n"
            "ESCALATION: End every response by reminding families that their nurse or doctor "
            "is always the right person to ask about their child's specific situation."
        )},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]

    for _ in range(3):
        r = client.chat.completions.create(model=CHAT_MODEL, messages=messages)
        answer = r.choices[0].message.content.strip()
        grade = textstat.flesch_kincaid_grade(answer)
        if grade <= 7.5:
            break

    disclaimer = (
        "\n\n---\n"
        "📋 **This is general information, not medical advice.** "
        "It is based on trusted resources including the Society of Critical Care Medicine (SCCM) "
        "and the American Academy of Pediatrics (AAP). "
        "Always talk to your child's care team for guidance specific to your child."
    )
    return answer + disclaimer

# -----------------------
# UI + FLOW
# -----------------------

# Language selector (preserves your multilingual capability)
language = st.selectbox(
    "🌐 Preferred language for answers:",
    ["English", "Spanish", "Mandarin", "French", "Arabic", "Portuguese", "Haitian Creole", "Other"],
    index=0
)

# Common question prompts to lower barrier to use
st.markdown("**Not sure what to ask? Try one of these:**")
col1, col2 = st.columns(2)
with col1:
    if st.button("What does a ventilator do?"):
        st.session_state["prefill"] = "What does a ventilator do and why might my child need one?"
    if st.button("Why are there so many alarms?"):
        st.session_state["prefill"] = "Why do the monitors alarm so often? Should I be worried every time?"
with col2:
    if st.button("How can I help my child?"):
        st.session_state["prefill"] = "What can I do to help my child while they are in the PICU?"
    if st.button("What is a central line?"):
        st.session_state["prefill"] = "What is a central line and why does my child have one?"

prefill_val = st.session_state.pop("prefill", "")
question = st.text_input("Or type your own question here:", value=prefill_val)

if language != "English" and question:
    question = question + f" (Please answer in {language}.)"

# Admin controls
with st.expander("⚙️ Knowledge Base Controls", expanded=False):
    if st.button("Rebuild index now"):
        try:
            INDEX_PATH.unlink(missing_ok=True)
            META_PATH.unlink(missing_ok=True)
        except Exception:
            pass
        st.cache_resource.clear()
        st.success("Cleared. The index will rebuild on the next question.")

# Load / build index
try:
    with st.spinner("🔄 Preparing knowledge base…"):
        index, metas = load_or_build_index()
        chunks = st.session_state.get("_chunks_cache")
        if chunks is None:
            chunks, _ = load_documents(DATA_DIR)
        st.success(f"Ready — {index.ntotal} knowledge chunks indexed.")
except Exception as e:
    st.error(f"⚠️ Failed to prepare knowledge base: {e}")
    chunks, index, metas = [], None, []

# Answer flow
if question and index is not None and len(chunks) == index.ntotal and index.ntotal > 0:

    # Guardrail check before hitting the model
    if is_out_of_scope(question):
        st.warning(
            "💙 This is a question that only your child's care team can answer — "
            "it depends on things specific to your child that we don't have access to here. "
            "Please ask your doctor or nurse directly. They want to talk with you.",
            icon="🤝"
        )
    else:
        with st.spinner("Finding an answer…"):
            answer = ask_question(question, index, metas, chunks, k=5)
            st.markdown("### 💬 Answer")
            st.write(answer)

elif question and (index is None or index.ntotal == 0):
    st.warning("⏳ Knowledge base not ready yet. Try rebuilding from the controls above.")
