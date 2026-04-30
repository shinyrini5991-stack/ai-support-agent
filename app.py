# LATEST VERSION TEST
import os, time, uuid
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
import chromadb

# ---------------- ENV ---------------- #
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant"
)

# ---------------- EMBEDDING + DB ---------------- #
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.Client()
collection = client.get_or_create_collection("support_kb")

# ---------------- STREAMLIT UI ---------------- #
st.set_page_config(layout="wide")
st.title("🤖 AI Support Resolution Agent")

# ---------------- SESSION STATE ---------------- #
if "messages" not in st.session_state:
    st.session_state.messages = []
if "feedback" not in st.session_state:
    st.session_state.feedback = {}

# ---------------- SIDEBAR ---------------- #
st.sidebar.header("⚙️ Controls")

prompt_mode = st.sidebar.selectbox(
    "Prompt Mode", ["basic", "strict"]
)

if st.sidebar.button("🧹 Reset Chat"):
    st.session_state.messages = []

# ---------------- SAFETY ---------------- #
def is_unsafe(query):
    unsafe_words = ["hack", "fraud", "attack", "illegal"]
    return any(w in query.lower() for w in unsafe_words)

# ---------------- TOOLS ---------------- #

def kb_tool(query):
    try:
        result = collection.query(
            query_embeddings=[embed_model.encode(query).tolist()],
            n_results=1
        )
        docs = result.get("documents", [[]])
        if docs and docs[0]:
            return docs[0][0]
    except:
        return None
    return None


def llm_tool(query):
    try:
        if prompt_mode == "basic":
            prompt = f"Answer briefly: {query}"
        else:
            prompt = f"""
You are a safe customer support agent.

Rules:
- Do NOT hallucinate
- If unsure, say "I don't know"
- Escalate sensitive issues

User Query: {query}
"""
        return llm.invoke(prompt).content
    except:
        return "⚠️ System error. Escalating to human support."


def escalation_tool(query):
    return "⚠️ This issue is escalated to human support."

# ---------------- AGENT ---------------- #

def agent(query):
    trace = []
    confidence = "Low"

    # Safety
    if is_unsafe(query):
        return "❌ Unsafe request blocked.", ["Safety triggered"], "High"

    trace.append("Safety passed")

    # RAG
    kb_result = kb_tool(query)
    if kb_result:
        trace.append("KB tool used (RAG)")
        return f"📘 {kb_result}", trace, "High"

    trace.append("No KB match")

    # Escalation condition
    if "urgent" in query.lower():
        trace.append("Escalation tool used")
        return escalation_tool(query), trace, "High"

    # LLM fallback
    trace.append("LLM tool used")
    return llm_tool(query), trace, "Medium"

# ---------------- STREAMING ---------------- #
def stream(text):
    box = st.empty()
    output = ""
    for char in text:
        output += char
        box.markdown(output + "▌")
        time.sleep(0.005)
    box.markdown(output)
    return output

# ---------------- CSV INGEST ---------------- #
st.sidebar.header("📂 Upload Dataset")

file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if file and st.sidebar.button("Generate Embeddings"):
    df = pd.read_csv(file)

    for _, row in df.iterrows():
        text = f"Q: {row['question']} A: {row['answer']}"

        collection.add(
            documents=[text],
            embeddings=[embed_model.encode(text).tolist()],
            ids=[str(uuid.uuid4())]
        )

    st.sidebar.success("✅ Data embedded successfully!")

# ---------------- DISPLAY CHAT ---------------- #
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            with st.expander("🧠 Trace + Confidence"):
                for t in msg["trace"]:
                    st.write("-", t)
                st.write("Confidence:", msg["confidence"])

# ---------------- INPUT ---------------- #
query = st.chat_input("Ask your question...")

if query:
    start = time.time()

    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("assistant"):
        response, trace, confidence = agent(query)
        output = stream(response)

    latency = round(time.time() - start, 2)

    msg_id = str(uuid.uuid4())

    st.session_state.messages.append({
        "id": msg_id,
        "role": "assistant",
        "content": output,
        "trace": trace,
        "confidence": confidence,
        "latency": latency
    })

    # -------- FEEDBACK -------- #
    col1, col2 = st.columns(2)

    with col1:
        if st.button("👍 Helpful", key=f"up_{msg_id}"):
            st.session_state.feedback[msg_id] = "positive"

    with col2:
        if st.button("👎 Not Helpful", key=f"down_{msg_id}"):
            st.session_state.feedback[msg_id] = "negative"

    st.rerun()

# ---------------- RAG vs NO RAG (IMPORTANT) ---------------- #
st.sidebar.header("📊 RAG Comparison")

if st.sidebar.button("Compare RAG vs LLM"):
    test_query = "How to reset password?"

    rag_resp, _, _ = agent(test_query)
    llm_resp = llm_tool(test_query)

    st.sidebar.write("Query:", test_query)
    st.sidebar.write("With RAG:", rag_resp)
    st.sidebar.write("Without RAG:", llm_resp)

# ---------------- EVALUATION ---------------- #
st.sidebar.header("🧪 Evaluation")

if st.sidebar.button("Run Tests"):
    tests = [
        "reset password",
        "refund time",
        "hack account",
        "urgent issue"
    ]

    results = []

    for t in tests:
        resp, _, conf = agent(t)
        results.append({
            "query": t,
            "response": resp,
            "confidence": conf
        })

    st.sidebar.write(pd.DataFrame(results))