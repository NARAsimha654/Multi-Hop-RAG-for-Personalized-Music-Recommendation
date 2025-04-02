# 🎵 RAG-Based Music Recommendation System

**Retrieval-Augmented Generation (RAG) for Personalized Music Recommendations**

![Music Recommendation Banner](https://your-image-url.com) _(Optional: Add an image/banner here)_

---

## 📌 Project Overview

This project implements a **Retrieval-Augmented Generation (RAG)** model to provide personalized music recommendations. By leveraging **multi-hop reasoning**, **dense and sparse retrieval**, and **transformer-based text generation**, the system enhances music recommendations based on user queries and past listening history.

The system retrieves relevant song metadata, lyrics, and user preferences before generating recommendations using **LLMs (Large Language Models)**.

---

## 🚀 Features

✅ **Personalized Music Recommendations** based on user profiles and past interactions  
✅ **Multi-Hop Reasoning** for complex queries (e.g., "Find me songs like Artist X but with upbeat jazz elements")  
✅ **Hybrid Retrieval** combining **Dense (FAISS)** and **Sparse (BM25)** search for relevant music data  
✅ **Transformer-based LLMs** for natural language response generation  
✅ **Scalable API** using FastAPI for real-time recommendations

---

## 🛠️ Tech Stack

| Category             | Technology Used                    |
| -------------------- | ---------------------------------- |
| Programming Language | Python 3.x                         |
| Model Frameworks     | PyTorch, Hugging Face Transformers |
| Retrieval Methods    | FAISS (Dense), BM25 (Sparse)       |
| Backend              | FastAPI                            |
| Deployment           | Docker (Optional)                  |
| Data Processing      | Pandas, NumPy                      |
| Visualization        | Matplotlib, Jupyter Notebooks      |

---

## 🎯 Novel Enhancements

### 🔹 1. Multi-Hop Graph-Based Retrieval (Graph + RAG Hybrid)

- Instead of relying only on **FAISS (dense retrieval) + BM25 (sparse retrieval)**, this system integrates a **Graph Neural Network (GNN) for multi-hop retrieval**.
- Builds a **Knowledge Graph (KG)** where **nodes** represent songs, artists, and genres, and **edges** capture relationships (e.g., collaborations, similar moods, genre transitions).
- The **retrieval process uses Graph Traversal + RAG**, meaning recommendations come from graph-based reasoning rather than just simple text matching.

**✅ Tech Used:**

- **Neo4j / NetworkX** for Graph Retrieval
- **Graph Neural Networks (GNNs)** for Multi-Hop Reasoning
- **FAISS + BM25** for Hybrid Search

### 🔹 2. Personalized Conversational AI for Music (Context-Aware Chatbot)

- Instead of returning **static song lists**, this system introduces a **Conversational AI** that refines recommendations dynamically.
- The chatbot interacts with the user in real-time and adjusts recommendations **based on feedback and past history**.
- Uses **memory-based reasoning** to store context (e.g., previous searches, favorite genres, disliked songs).

**✅ Tech Used:**

- **LangChain / OpenAI GPT + Memory Storage**
- **FastAPI / Spring Boot** Backend
- **Spotify API Integration for Playback**

### 🔹 3. Emotional & Mood-Based Retrieval (Affect-Aware Recommendations)

- Uses **Sentiment Analysis** to detect **mood & intent** in user queries.
- Retrieves music based on **emotional tone**.

**✅ Tech Used:**

- **OpenAI GPT / Hugging Face Transformers**
- **Emotion-Based Embeddings (AffectVec, VADER, etc.)**

## 🛠 Tech Stack

| **Category**                | **Technology Used**                              |
| --------------------------- | ------------------------------------------------ |
| **Backend**                 | FastAPI (Python) / Spring Boot (Java)            |
| **Graph Retrieval**         | Neo4j, NetworkX, Graph Neural Networks (GNNs)    |
| **Embedding Search**        | FAISS (Dense Retrieval), BM25 (Sparse Retrieval) |
| **RAG Model**               | Hugging Face Transformers, OpenAI API            |
| **Conversational AI**       | LangChain + GPT-based Memory                     |
| **Emotion Analysis**        | Sentiment Analysis (AffectVec, VADER)            |
| **Spotify API Integration** | Spotipy                                          |

## 📌 Final Novel Pipeline

### 1️⃣ Query Input (Spotify UI + Chatbot)

- User enters a **natural language query**.
- Sentiment analysis detects **mood & intent**.

### 2️⃣ Multi-Hop Graph Retrieval

- A **Graph Neural Network (GNN)** explores artist relations, genre transitions.
- FAISS + BM25 retrieve relevant tracks.

### 3️⃣ Retrieval-Augmented Generation (RAG)

- RAG fine-tuned for **music metadata, lyrics, and emotions**.
- Generates a **natural language response** explaining the recommendations.

### 4️⃣ Interactive Conversational AI

- AI interacts with the user for **real-time refinement**.
- Memory-based **context tracking** improves personalization.

### 5️⃣ Final Playlist & Spotify API Integration

- Recommended songs/playlists are sent to the **Spotify API for playback**.
- User can **modify preferences dynamically**.

## ⏳ 1.5-Month Implementation Plan

| **Week**   | **Task**                                            |
| ---------- | --------------------------------------------------- |
| **Week 1** | Set up Spotify API, retrieve user listening history |
| **Week 2** | Implement **Graph-Based Retrieval (GNN + Neo4j)**   |
| **Week 3** | Integrate **FAISS + BM25 Hybrid Retrieval**         |
| **Week 4** | Fine-tune **RAG for Music**                         |
| **Week 5** | Build **Conversational AI (LangChain + GPT)**       |
| **Week 6** | Test & Deploy via FastAPI/Spring Boot               |

## 🔗 Related GitHub Projects for Reference

1. **RAG-Based Retrieval** - [https://github.com/facebookresearch/DPR](https://github.com/facebookresearch/DPR)
2. **Neo4j + Graph-Based Retrieval** - [https://github.com/neo4j-examples](https://github.com/neo4j-examples)
3. **FAISS Dense Search** - [https://github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)
4. **BM25 Sparse Search** - [https://github.com/dorianbrown/rank_bm25](https://github.com/dorianbrown/rank_bm25)
5. **Spotify API Example** - [https://github.com/spotipy-dev](https://github.com/spotipy-dev)

## ✅ Summary of Novel Contributions

| **Feature**                                  | **Existing Systems (Spotify, Apple, etc.)**               | **Our Novel System**                               |
| -------------------------------------------- | --------------------------------------------------------- | -------------------------------------------------- |
| **Multi-Hop Graph Retrieval**                | ❌ Not used for music recommendations                     | ✅ Uses **GNN + RAG for reasoning**                |
| **Hybrid Retrieval (FAISS + BM25)**          | ⚠️ Some systems use embeddings (FAISS), but not multi-hop | ✅ Uses **FAISS + BM25 + Graph Retrieval**         |
| **Conversational AI for Dynamic Refinement** | ❌ No dynamic chat-based music recommendations            | ✅ **Refines recommendations via AI chatbot**      |
| **Emotion-Based Music Retrieval**            | ❌ No explicit emotion detection in text queries          | ✅ **Uses sentiment analysis for recommendations** |
