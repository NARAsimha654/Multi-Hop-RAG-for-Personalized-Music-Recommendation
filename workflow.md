# 🚀 Multi-Hop RAG for Personalized Music Recommendation

## **📌 Project Overview**

This project implements a Retrieval-Augmented Generation (RAG) model to provide personalized music recommendations. By leveraging **multi-hop reasoning, dense & sparse retrieval, graph-based knowledge, and transformer-based text generation**, this system enhances music recommendations based on user queries, sentiment, and past listening history.

---

## **📂 Phase 1: Data Collection, Preprocessing & Feature Engineering**

### 🔹 **Task 1: Dataset Aggregation & APIs Setup**

- ✅ **Datasets & APIs**:
  - **Million Song Dataset (MSD)** + **Last.fm Dataset** (user preferences)
  - **Spotify Million Playlist Dataset (MPD)** (playlist co-occurrence)
  - **GTZAN Genre Dataset** via **DeepLake** (genre classification benchmark)
  - **Genius API** (lyrics data)
  - **MusicBrainz API** (metadata, artist/song relationships)
  - **Last.fm API** (user tagging behavior)
  - **Librosa**, **PyAudioAnalysis** (audio feature extraction)
  - **VADER**, **NRCLex**, **NRC-VAD** (emotion & sentiment analysis)

📌 **Actionable Steps**:

- [x] Download datasets onto a **local machine**.
- [x] Configure and validate all external APIs.
- [ ] Move to **DGX A100** once preprocessing is completed.
- [ ] **Data versioning**: Track changes using **DVC or MLflow**.

### 🔹 **Task 2: Preprocessing & Feature Engineering**

📌 **Actionable Steps**:

- [ ] **Data Cleaning**: Standardize fields, handle nulls & duplicates.
- [ ] **Feature Engineering**:
  - Node features: embeddings from audio/lyrics.
  - Edge weights: similarity, co-occurrence.
- [ ] **Multimodal Embedding Generation**:
  - Lyrics embeddings via **SBERT** or **Word2Vec**.
  - Audio embeddings via **Librosa** & **PyAudioAnalysis**.
  - Emotion embeddings using **VADER**, **NRCLex**, and **NRC-VAD**.

---

## **📂 Phase 2: Knowledge Graph Construction (Multi-Hop Retrieval)**

### 🔹 **Task 3: Build a Music Knowledge Graph**

📌 **Actionable Steps**:

- [ ] Define schema: Nodes = Songs, Artists, Genres, Moods, Users.
- [ ] Edges = Collaborations, Similar moods, Playlist co-occurrence.
- [ ] Construct graph using **Neo4j** with `py2neo` or `neo4j-driver`.
- [ ] Validate structure with **NetworkX**.
- [ ] Enable dynamic updates from APIs (Genius, MusicBrainz, Last.fm).

### 🔹 **Task 4: Train GNN for Multi-Hop Retrieval**

📌 **Actionable Steps**:

- [ ] Choose model: **GraphSAGE**, **GAT**, or **GCN**.
- [ ] Train on Neo4j-extracted embeddings.
- [ ] Evaluate using **Path Recall**, **Hit@K**, and **MAP**.

---

## **📂 Phase 3: Hybrid Retrieval System**

### 🔹 **Task 5: Dense Retrieval via FAISS**

📌 **Actionable Steps**:

- [ ] Generate dense audio/lyrics embeddings.
- [ ] Index them using **FAISS** for fast similarity search.

### 🔹 **Task 6: Sparse Retrieval via BM25**

📌 **Actionable Steps**:

- [ ] Implement **BM25** using `rank_bm25` for lyrics search.
- [ ] Use **reranking** via hybrid scoring: semantic + emotion similarity.

📌 **Baselines**:

- [ ] Collaborative Filtering (user-item matrix).
- [ ] Compare with Spotify Recommendation Engine.

---

## **📂 Phase 4: RAG-Based Music Recommendation**

### 🔹 **Task 7: Fine-Tune RAG Pipeline**

📌 **Actionable Steps**:

- [ ] Implement **RAG** with Hugging Face (e.g., T5, BART).
- [ ] Finetune on playlist and query-based recommendation tasks.

### 🔹 **Task 8: Emotion-Aware Retrieval Layer**

📌 **Actionable Steps**:

- [ ] Analyze lyrics using **VADER**, **NRCLex**, **NRC-VAD**.
- [ ] Weight retrieval based on mood similarity.
- [ ] Enhance queries with emotion context.

---

## **📂 Phase 5: Conversational Interface (Chatbot)**

### 🔹 **Task 9: GPT-Based Music Chatbot**

📌 **Actionable Steps**:

- [ ] Use **LangChain** with **GPT-3.5/4**.
- [ ] Personalize conversations using user preferences.
- [ ] Enable fallback logic and privacy compliance.

---

## **📂 Phase 6: Backend API & Spotify Integration**

### 🔹 **Task 10: Backend Architecture**

📌 **Actionable Steps**:

- [ ] Develop REST API with **FastAPI**.
- [ ] Implement fallback for failed Spotify API calls.

### 🔹 **Task 11: Playback via Spotify**

📌 **Actionable Steps**:

- [ ] Use **Spotipy** for playback and playlist management.
- [ ] Authenticate using Spotify OAuth.

---

## **📂 Phase 7: Training & Deployment on DGX A100**

### 🔹 **Task 12: High-Performance Training**

📌 **Actionable Steps**:

- [ ] Migrate to **DGX A100 server**.
- [ ] Train dense retrievers, GNN, and RAG model.

### 🔹 **Task 13: System Load & Inference Optimization**

📌 **Actionable Steps**:

- [ ] Use **Locust** for load testing.
- [ ] Optimize memory and GPU utilization.

---

## **📂 Phase 8: Evaluation & Research Paper**

### 🔹 **Task 14: Evaluate System Performance**

📌 **Actionable Steps**:

- [ ] Compare RAG vs traditional hybrid retrieval models.
- [ ] Conduct ablation studies for each modality.

### 🔹 **Task 15: Write & Submit Research Paper**

📌 **Actionable Steps**:

- [ ] Complete detailed literature review.
- [ ] Highlight novelty in multimodal & emotion-aware retrieval.
- [ ] Target submission to **NeurIPS, ICASSP, or ACM RecSys**.

---
