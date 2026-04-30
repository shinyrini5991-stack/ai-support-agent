# 🤖 AI Support Resolution Agent

An intelligent **Customer Support AI Agent** designed to automate real-world support workflows using LLMs, Retrieval-Augmented Generation (RAG), and tool-based reasoning.

---

## 🚀 Project Overview

This project simulates a **real customer support system** where users can:

* Ask support-related questions
* Get accurate answers from a knowledge base
* Receive safe, policy-compliant responses
* Escalate sensitive issues to human support

The system is built with a strong focus on:

* ✅ Reliability
* ✅ Explainability
* ✅ Safety-first behaviour
* ✅ Practical usability

---

## 👤 User Persona

**Primary User:**
Customers seeking help with:

* Account issues
* Payments & refunds
* Subscriptions
* Delivery problems

---

## 🎯 Problem Statement

Customer support teams face:

* High volume of repetitive queries
* Delayed responses
* Inconsistent answers

👉 This AI agent solves these problems by:

* Automating responses
* Using knowledge grounding (RAG)
* Ensuring safe and controlled outputs

---

## 🧠 System Architecture

```mermaid
flowchart TD
    A[User Input] --> B[Safety Check]
    B --> C{Unsafe?}
    C -- Yes --> D[Block Request]
    C -- No --> E[Retriever (Chroma DB)]
    E --> F{KB Match?}
    F -- Yes --> G[Return KB Answer]
    F -- No --> H{Urgent?}
    H -- Yes --> I[Escalate to Human]
    H -- No --> J[LLM (Groq)]
    J --> K[Final Response]
```

---

## ⚙️ Tech Stack

* **LLM**: Groq (LLaMA models)
* **Framework**: LangChain
* **Vector DB**: Chroma
* **Embeddings**: Sentence Transformers
* **UI**: Streamlit
* **Language**: Python

---

## 🧩 Features

### ✅ Core Capabilities

* LLM-based intelligent responses
* Retrieval-Augmented Generation (RAG)
* CSV-based knowledge ingestion
* Semantic search using embeddings

### 🔐 Safety Features

* Blocks unsafe queries (fraud, hacking, etc.)
* No hallucinated policies
* Escalates sensitive issues
* No personal data stored

### 🛠 Tool-Based Architecture

* KB Retrieval Tool
* LLM Response Tool
* Escalation Tool

### 🧠 Explainability

* Step-by-step trace of reasoning
* Confidence levels (High / Medium / Low)

### 💬 User Experience

* Chat interface (Streamlit)
* Streaming responses
* Feedback system (👍 / 👎)

---

## 📂 Dataset Preparation

A structured **customer support knowledge base** is used:

```csv
id,question,answer
1,How do I reset my password?,Go to Settings → Account → Reset Password.
2,Refund time,Refunds are processed within 5–7 days.
3,Cancel subscription,Go to Settings → Subscription → Cancel.
```

👉 Data is:

* Embedded using sentence transformers
* Stored in Chroma vector database
* Retrieved using semantic similarity

---

## 🔄 Workflow

1. User submits query
2. Safety check is applied
3. System searches knowledge base (RAG)
4. If found → return grounded answer
5. Else → use LLM
6. If sensitive → escalate

---

## 📊 RAG vs Non-RAG Comparison

| Query          | Without RAG    | With RAG            |
| -------------- | -------------- | ------------------- |
| Reset password | Generic answer | Exact steps from KB |
| Refund time    | Approximate    | Accurate (5–7 days) |

---

## 🧪 Evaluation

### Test Scenarios

* Password reset
* Refund query
* Unsafe request (hack account)
* Urgent escalation

### Metrics

* Response accuracy
* Safety compliance
* Latency
* Consistency

---

## ⚠️ Limitations

* Limited dataset size
* No real backend integration
* Basic memory (session-based only)
* No advanced multi-agent orchestration

---

## 🚀 Future Improvements

* Add full conversation memory
* Integrate real backend APIs
* Use advanced orchestration (LangGraph / CrewAI)
* Improve retrieval with chunking
* Add confidence scoring model

---

## ▶️ How to Run

```bash
pip install streamlit langchain-groq sentence-transformers chromadb python-dotenv pandas
streamlit run app.py
```

---

## 🔑 Environment Setup

Create `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

---

## 📦 Project Structure

```
ai-support-agent/
│
├── app.py
├── support_kb.csv
├── README.md
├── requirements.txt
└── .env
```

---

## 🎯 Success Criteria

* Accurate responses from KB
* Safe handling of harmful queries
* Proper escalation when needed
* Clear explainability (trace + confidence)

---

## 📌 Conclusion

This project demonstrates a **production-style AI support agent** that combines:

* LLM reasoning
* Knowledge retrieval
* Safety-first design
* Explainable decision-making

👉 Making it suitable for real-world customer support automation.

---
