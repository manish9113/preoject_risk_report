# 🧠 AI-Powered Project Risk Management System

This project is an **AI-powered multi-agent system** built using **CrewAI**, designed to assess, manage, and mitigate risks in real-time within IT projects. The system leverages **locally hosted LLMs via Ollama**, **Pinecone** for vector storage, and **Streamlit** for an interactive front-end interface.

---

## 🚀 Features

- 🧑‍💼 **Multi-Agent Architecture** using CrewAI
- 🛠️ **Custom Tools** for each agent tailored to specific tasks
- 📊 Real-time **project status analysis**
- 📉 **Risk scoring & mitigation strategies**
- 🌐 **Market analysis** based on financial trends
- 🧠 **LLM Responses** using local **Ollama models**
- 🧬 **Semantic Search** & vector memory via **Pinecone**
- 🌐 Intuitive **Streamlit UI** to interact with the system

---

## 🏗️ Tech Stack

| Component         | Tech Used           |
|------------------|---------------------|
| **Framework**     | [CrewAI](https://docs.crewai.com) |
| **Frontend**      | [Streamlit](https://streamlit.io) |
| **LLM**           | [Ollama](https://ollama.com) (locally hosted) |
| **Vector DB**     | [Pinecone](https://www.pinecone.io) |
| **Language**      | Python |
| **Tooling**       | Custom tools for each agent |

---

## 🧠 Agents & Responsibilities

1. **Project Risk Manager**
   - Oversees overall risk management
   - Delegates tasks to specialized agents

2. **Market Analysis Agent**
   - Fetches market data
   - Assesses external economic risks

3. **Risk Scoring Agent**
   - Analyzes investment, financial, and transactional risks
   - Uses vector similarity with Pinecone to compare past scenarios

4. **Project Status Tracking Agent**
   - Monitors internal risks (delays, resignations, payments)

5. **Reporting Agent**
   - Generates human-readable reports
   - Provides mitigation suggestions and risk alerts

---

## 🧰 Custom Tools

Each agent is equipped with **custom-built tools** tailored to their specific task. These tools interface with APIs, databases, or data processors to ensure high-quality output and efficient task handling.


