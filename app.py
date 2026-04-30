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

    # ✅ Improved error handling
    except Exception as e:
        return f"⚠️ LLM error: {str(e)}. Escalating to human support."


def escalation_tool(query):
    return "⚠️ This issue is escalated to human support."

# ---------------- AGENT (UPDATED - PHASE 5 COMPLETE) ---------------- #

def agent(query, step=0):
    trace = []
    confidence = "Low"

    # 🔁 LOOP PREVENTION
    if step > 3:
        return "⚠️ Too many steps. Escalating to human support.", ["Loop detected"], "High"

    # 🔐 SAFETY CHECK
    if is_unsafe(query):
        trace.append("❌ Safety triggered → Unsafe query blocked")
        return "❌ Unsafe request blocked.", trace, "High"

    trace.append("✅ Safety check passed")

    # 📘 TOOL 1: KB (RAG)
    kb_result = kb_tool(query)
    if kb_result:
        trace.append("🛠 Tool Selected: KB Tool (RAG)")
        trace.append("📄 Reason: Found relevant answer in knowledge base")
        return f"📘 {kb_result}", trace, "High"

    trace.append("❌ No KB match found")

    # 🚨 TOOL 2: ESCALATION
    if "urgent" in query.lower():
        trace.append("🛠 Tool Selected: Escalation Tool")
        trace.append("📄 Reason: Query marked as urgent")
        return escalation_tool(query), trace, "High"

    # 🤖 TOOL 3: LLM
    trace.append("🛠 Tool Selected: LLM Tool")
    trace.append("📄 Reason: No KB match, fallback to LLM")

    response = llm_tool(query)
    return response, trace, "Medium"

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

# ---------------- TOOL DEMONSTRATION (PHASE 5 REQUIREMENT) ---------------- #

st.sidebar.header("⚠️ Tool Evaluation")

# ❌ Incorrect tool usage demo
if st.sidebar.button("Test Incorrect Tool Usage"):
    test_query = "How do I reset my password?"

    wrong = llm_tool(test_query)  # forcing wrong tool
    correct, trace, _ = agent(test_query)

    st.sidebar.write("Query:", test_query)
    st.sidebar.write("❌ Wrong Tool (LLM):", wrong)
    st.sidebar.write("✅ Correct Tool (KB):", correct)

    st.sidebar.write("🧠 Trace:")
    for t in trace:
        st.sidebar.write("-", t)

# ✅ Correct tool usage demo
if st.sidebar.button("Test Tool Selection"):
    test_queries = [
        "reset password",
        "urgent account issue",
        "what is AI?"
    ]

    for q in test_queries:
        resp, trace, conf = agent(q)

        st.sidebar.write("------")
        st.sidebar.write("Query:", q)
        st.sidebar.write("Response:", resp)
        st.sidebar.write("Confidence:", conf)
        for t in trace:
            st.sidebar.write("-", t)

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

    # 👍👎 Feedback
    col1, col2 = st.columns(2)

    with col1:
        if st.button("👍 Helpful", key=f"up_{msg_id}"):
            st.session_state.feedback[msg_id] = "positive"

    with col2:
        if st.button("👎 Not Helpful", key=f"down_{msg_id}"):
            st.session_state.feedback[msg_id] = "negative"

    st.rerun()

# ---------------- RAG vs NO RAG ---------------- #
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