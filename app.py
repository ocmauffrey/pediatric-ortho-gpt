import streamlit as st
import faiss
import numpy as np
import textstat
from pathlib import Path
from openai import OpenAI
import chardet

# Setup OpenAI
import streamlit as st
from openai import OpenAI

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
                model="text-embedding-3-small",
                input=chunk
            )
            vectors.append(response.data[0].embedding)
        except:
            continue
    return np.array(vectors, dtype=np.float32)

# Ask the assistant
def ask_question(question, chunks, embeddings, k=3):
    embed_response = client.embeddings.create(
        model="text-embedding-3-small",
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
            "Use short words and short sentences. Use line breaks between ideas for readability."
            "Use bulleted or numbered lists for treatment steps or symptoms. Avoid medical terms unless you explain them. "
            "Write at a 5th to 6th grade reading level. Keep each answer kind, calm, and easy to follow. "
            "Only use the provided context. Never guess."
        )},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]

    for _ in range(3):  # retry if grade too high
        chat_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        response_text = chat_response.choices[0].message.content.strip()
        grade = textstat.flesch_kincaid_grade(response_text)
        if grade <= 7.5:
            break

    disclaimer = (
        "\n\n📢 This is general information, not medical advice." 
        "Please visit https://orthokids.org or https://www.orthoinfo.org for more information!"
        "Please talk to your child's doctor for specific care or emergencies."
    )

    return response_text + disclaimer

# === Streamlit UI ===
st.set_page_config(page_title="Pediatric Ortho Assistant", layout="centered")
st.title("🦴 Pediatric Ortho Assistant")
st.markdown("Ask a question about pediatric bone injuries or treatments:")

question = st.text_input("Enter your question:")
if 'chunks' not in st.session_state:
    with st.spinner("Loading knowledge base..."):
        st.session_state.chunks = load_documents("AAOS_Peds")  # replace with your folder
        st.session_state.embeddings = embed_chunks(st.session_state.chunks)

if question:
    with st.spinner("Thinking..."):
        answer = ask_question(question, st.session_state.chunks, st.session_state.embeddings)
        st.markdown("### 💬 Answer:")
        st.write(answer)
