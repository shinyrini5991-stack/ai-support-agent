# 🤖 AI Support Resolution Agent

## 📌 Project Overview
This is an AI-powered customer support agent built using LangChain and Groq API.  
It helps users resolve support queries like account issues, billing problems, and technical support.

---

## ⚙️ Features
- Natural language customer support
- Safe response handling (refuses harmful requests)
- Knowledge-based responses (RAG-style logic)
- Escalation for unresolved issues
- No personal data storage

---

## 🧠 Tech Stack
- Python
- LangChain
- Groq LLM (Llama 3.1)
- dotenv (.env for API key)

---

## 📁 Project Structure
ai-support-agent/
│
├── app.py
├── .env
└── README.md


---

## 🔐 Setup Instructions

### 1. Install dependencies
- pip install langchain langchain-groq python-dotenv


### 2. Add API Key
Create `.env` file:


### 3. Run the project
python app.py


---

## 💬 Example Queries
- "How do I reset my password?"
- "I was charged twice"
- "My account is locked"
- "Hack my account" ❌ (will be refused)

---

## 🛡️ Safety Features
- Refuses unsafe or illegal requests
- Does not hallucinate policies
- Escalates unresolved issues
- Does not store personal data

---

## 🚀 Future Improvements
- Add full RAG with vector database
- Add Streamlit UI chatbot
- Add ticketing system integration

## 🏗️ System Architecture

## 🏗️ System Architecture

```mermaid
flowchart TD

A["User Query"] --> B["Input Handler"]
B --> C["Safety Filter"]
C --> D{"Unsafe Request?"}

D -- Yes --> E["Refuse Response"]
D -- No --> F["Intent Detection"]

F --> G["Retriever (RAG)"]
G --> H{"Relevant Info Found?"}

H -- Yes --> I["Return KB Answer"]
H -- No --> J["Groq LLM"]

J --> K["Response Generator"]
I --> K

K --> L["User Output"]
