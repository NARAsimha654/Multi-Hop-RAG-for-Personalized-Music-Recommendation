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

# 📁 Project Directory Structure: Music RAG Recommender

music-rag-recommender/<br>
&emsp;├── 📁 data/ &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;<!-- Local/processed data --> Local/processed data (can use .gitignore)<br>
&emsp;│&emsp;&emsp;├── raw/ &emsp;&emsp;&emsp;&emsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Raw input datasets (MSD, MPD, GTZAN, etc.)<br>
&emsp;│&emsp;&emsp;├── processed/ &emsp;&emsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Cleaned/structured CSVs/DataFrames<br>
&emsp;│&emsp;&emsp;└── external/ &emsp;&emsp;&emsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Outputs from APIs (e.g., Genius, Last.fm)<br>
<br>
&emsp;├── 📁 notebooks/ &emsp;&emsp;&emsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Jupyter notebooks for exploration<br>
&emsp;│&emsp;&emsp;├── 01_preprocessing.ipynb<br>
&emsp;│&emsp;&emsp;├── 02_graph_construction.ipynb<br>
&emsp;│&emsp;&emsp;├── 03_gnn_training.ipynb<br>
&emsp;│&emsp;&emsp;└── 04_rag_pipeline.ipynb<br>
<br>
&emsp;├── 📁 src/ &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Python modules for reusable code<br>
&emsp;│&emsp;&emsp;├── data_loader.py &emsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Load & preprocess datasets<br>
&emsp;│&emsp;&emsp;├── feature_engineering.py &nbsp;&nbsp;&nbsp; Create embeddings, extract features<br>
&emsp;│&emsp;&emsp;├── graph_builder.py &emsp;&emsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Knowledge graph construction<br>
&emsp;│&emsp;&emsp;├── retrieval.py &emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Dense + Sparse retrieval (FAISS, BM25)<br>
&emsp;│&emsp;&emsp;├── sentiment_analysis.py &emsp;&nbsp;&nbsp;&nbsp;&nbsp; Emotion-aware processing (VADER, NRC, etc.)<br>
&emsp;│&emsp;&emsp;└── rag_pipeline.py &emsp;&emsp;&emsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Final RAG-based pipeline<br>
<br>
&emsp;├── 📁 models/ &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Saved models (GNN, RAG, etc.)<br>
&emsp;│&emsp;&emsp;└── gnn_model.pt<br>
<br>
&emsp;├── 📁 evaluation/ &emsp;&emsp;&emsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Scripts to evaluate recommendation performance<br>
&emsp;│&emsp;&emsp;└── metrics.py<br>
<br>
&emsp;├── 📁 api/ &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;&nbsp;&nbsp;&nbsp; Backend API files (FastAPI, etc.)<br>
&emsp;│&emsp;&emsp;└── main.py<br>
<br>
&emsp;├── 📁 config/ &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Config files for paths, hyperparams, etc.<br>
&emsp;│&emsp;&emsp;└── config.yaml<br>
<br>
&emsp;├── 📄 workflow.md &emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;&nbsp;&nbsp; Full end-to-end workflow steps ✅<br>
&emsp;├── 📄 datasets_and_tools.md &emsp;&nbsp; What we made earlier 📦<br>
&emsp;├── 📄 README.md &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;&nbsp; Overview of the project<br>
&emsp;├── 📄 requirements.txt &emsp;&emsp;&emsp;&nbsp;&nbsp;&nbsp; All required pip packages<br>
&emsp;├── 📄 .gitignore &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;&nbsp; Ignore large data/models<br>
&emsp;└── 📄 LICENSE<br>
