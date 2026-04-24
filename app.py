import os
import time
import uuid
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
import chromadb

# ---------------- LOAD ENV ---------------- #
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    api_key=groq_api_key,
    model="llama-3.1-8b-instant"
)

# ---------------- RAG SETUP ---------------- #
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("support_kb")

# ---------------- STREAMLIT CONFIG ---------------- #
st.set_page_config(page_title="AI Support Agent", layout="wide")
st.title("🤖 AI Support Resolution Agent")

# ---------------- SESSION STATE ---------------- #
if "messages" not in st.session_state:
    st.session_state.messages = []

if "feedback" not in st.session_state:
    st.session_state.feedback = {}

# ---------------- SIDEBAR ---------------- #
framework = st.sidebar.selectbox(
    "Select Framework",
    ["LangChain", "CrewAI", "LangGraph"]
)

st.sidebar.markdown("### 📂 Upload CSV for Knowledge Base")
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# ---------------- SAFETY FILTER ---------------- #
def is_unsafe(query):
    unsafe_words = ["hack", "attack", "steal", "fraud", "illegal", "bypass"]
    return any(word in query.lower() for word in unsafe_words)

# ---------------- RAG SEARCH ---------------- #
def search_kb(query):
    try:
        results = collection.query(
            query_embeddings=[embed_model.encode(query).tolist()],
            n_results=1
        )

        if results and results["documents"][0]:
            return results["documents"][0][0]
    except:
        pass
    return None

# ---------------- STREAMING ---------------- #
def stream_text(text):
    placeholder = st.empty()
    output = ""

    for char in text:
        output += char
        placeholder.markdown(output + "▌")
        time.sleep(0.01)

    return output

# ---------------- AGENT ---------------- #
def agent(query):

    if is_unsafe(query):
        return "❌ Request blocked due to safety policy."

    kb_answer = search_kb(query)

    if kb_answer:
        return f"📘 Answer (RAG): {kb_answer}"

    prompt = f"""
You are a safe AI customer support assistant.

Rules:
- Do not hallucinate
- If unsure, say escalate
- Be concise

Framework: {framework}

User: {query}
"""

    res = llm.invoke(prompt)
    return res.content

# ---------------- CSV INGESTION ---------------- #
if file is not None:
    if st.sidebar.button("Generate Embeddings"):
        df = pd.read_csv(file)

        for i, row in df.iterrows():
            text = " ".join(str(x) for x in row.values)
            emb = embed_model.encode(text).tolist()

            collection.add(
                documents=[text],
                embeddings=[emb],
                ids=[str(uuid.uuid4())]
            )

        st.sidebar.success("Embeddings stored successfully!")

# ---------------- CHAT DISPLAY ---------------- #
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # 👍 👎 FEEDBACK FIXED HERE
        if msg["role"] == "assistant":

            msg_id = msg["id"]

            col1, col2 = st.columns(2)

            with col1:
                if st.button("👍 Helpful", key=f"up_{msg_id}"):
                    st.session_state.feedback[msg_id] = "positive"
                    st.success("Thanks for feedback 👍")

            with col2:
                if st.button("👎 Not Helpful", key=f"down_{msg_id}"):
                    st.session_state.feedback[msg_id] = "negative"
                    st.warning("Feedback recorded 👎")

# ---------------- INPUT ---------------- #
query = st.chat_input("Ask your support question...")

if query:

    # store user message
    st.session_state.messages.append({
        "id": str(uuid.uuid4()),
        "role": "user",
        "content": query
    })

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent(query)

        final_text = stream_text(response)

    # store assistant message
    st.session_state.messages.append({
        "id": str(uuid.uuid4()),
        "role": "assistant",
        "content": final_text
    })

    st.rerun()