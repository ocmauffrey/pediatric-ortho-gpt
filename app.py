# Setup OpenAI + dependencies
import streamlit as st
from openai import OpenAI
import numpy as np
import faiss
import chardet
from pathlib import Path
import textstat

client = OpenAI(api_key=st.secrets["openai"]["api_key"])

# Load documents from txt files
def read_file_with_detected_encoding(file_path):
    with open(file_path, "rb") as f:
        raw_data = f.read()
        detected = chardet.detect(raw_data)
        return raw_data.decode(detected["encoding"])

def load_documents(folder_path: str):
    chunks = []
    for file in Path(folder_path).glob("*.txt"):
        try:
            text = read_file_with_detected_encoding(file)
            split_text = text.split("\n\n")
            split_text = [s.strip() for s in split_text if s.strip()]
            chunks.extend(split_text)
        except:
            continue
    return chunks

def embed_chunks(chunks):
    vectors = []
    for chunk in chunks:
        try:
            response = client.embeddings.create(
                model="text-embedding-ada-002",  # safer fallback model
                input=chunk
            )
            vectors.append(response.data[0].embedding)
        except:
            continue
    return np.array(vectors, dtype=np.float32)

# Ask the assistant
def ask_question(question, chunks, embeddings, k=3):
    if embeddings.shape[0] == 0:
        return "❌ No knowledge base available to answer this question."

    embed_response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=question
    )
    query_embedding = np.array([embed_response.data[0].embedding], dtype=np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    _, indices = index.search(query_embedding, k)
    selected_chunks = [chunks[i] for i in indices[0]]
    context = "\n\n".join(selected_chunks)

    messages = [
        {"role": "system", "content": (
            "You are a pediatric orthopedic guide. Speak clearly to parents using plain English. "
            "Use short words and short sentences. Use line breaks between ideas for readability. "
            "Use bulleted or numbered lists for treatment steps or symptoms. Avoid medical terms unless you explain them. "
            "Write at a 5th to 6th grade reading level. Keep each answer kind, calm, and easy to follow. "
            "Only use the provided context. Never guess."
        )},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]

    for _ in range(3):  # retry if reading grade is too high
        chat_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        response_text = chat_response.choices[0].message.content.strip()
        grade = textstat.flesch_kincaid_grade(response_text)
        if grade <= 7.5:
            break

    disclaimer = (
        "\n\n📢 This is general information, not medical advice. "
        "Visit https://orthokids.org or https://orthoinfo.aaos.org for more info. "
        "Talk to your child's doctor for specific care or emergencies."
    )

    return response_text + disclaimer

# === Streamlit UI ===
st.set_page_config(page_title="Pediatric Ortho Assistant", layout="centered")
st.title("🦴 Pediatric Ortho Assistant")
st.markdown("Ask a question about pediatric bone injuries or treatments:")

question = st.text_input("Enter your question:")

# Load knowledge base once
if 'chunks' not in st.session_state:
    with st.spinner("🔄 Loading knowledge base..."):
        chunks = load_documents("AAOS_Peds")  # replace with your folder
        embeddings = embed_chunks(chunks)
        if len(embeddings) == 0:
            st.error("⚠️ No embeddings were generated. Check your .txt files or API access.")
        else:
            st.session_state.chunks = chunks
            st.session_state.embeddings = embeddings

# Handle question
if question and "chunks" in st.session_state and "embeddings" in st.session_state:
    with st.spinner("🤖 Thinking..."):
        answer = ask_question(question, st.session_state.chunks, st.session_state.embeddings)
        st.markdown("### 💬 Answer:")
        st.write(answer)
elif question:
    st.warning("⏳ Please wait for the knowledge base to finish loading.")
