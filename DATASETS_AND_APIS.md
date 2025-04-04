# 📚 Datasets & APIs Used in the Project

This project utilizes a rich collection of music-related datasets and APIs to support the development of a **Multi-Hop Retrieval-Augmented Generation (RAG) System for Personalized Music Recommendation**.

---

## 🎶 Datasets

### 1. **Million Song Dataset (MSD)**

- **Description**: A comprehensive dataset of audio features and metadata for one million contemporary popular music tracks.
- **Source**: [http://millionsongdataset.com](http://millionsongdataset.com)
- **Used For**: Metadata, artist information, song titles, release years.

---

### 2. **Spotify Million Playlist Dataset (MPD)**

- **Description**: A large-scale collection of user-generated Spotify playlists.
- **Source**: [https://github.com/spotify/annoy/tree/main/examples/mpd](https://github.com/spotify/annoy/tree/main/examples/mpd)
- **Used For**: User playlist context, co-occurrence patterns, personalization signals.

---

### 3. **GTZAN Genre Classification Dataset** (via DeepLake)

- **Description**: 1,000 audio tracks categorized into 10 genres for music genre classification benchmarking.
- **Accessed via**: [DeepLake](https://activeloop.ai) using `deeplake.load("hub://activeloop/gtzan-genre")`
- **Used For**: Genre classification and evaluation.

---

## 🎤 Lyrics & Sentiment Analysis

### 4. **Genius API**

- **Description**: API access to song lyrics and artist metadata.
- **Source**: [https://docs.genius.com/](https://docs.genius.com/)
- **Used For**: Lyrics retrieval for sentiment and emotion analysis.

### 5. **VADER (Valence Aware Dictionary and sEntiment Reasoner)**

- **Description**: A rule-based sentiment analysis tool specifically tuned for social media and lyrics.
- **Source**: `pip install vaderSentiment`
- **Used For**: Sentiment analysis on song lyrics.

### 6. **NRC-VAD Lexicon**

- **Description**: A lexicon mapping words to Valence, Arousal, and Dominance (VAD) scores.
- **Source**: [https://saifmohammad.com/WebPages/nrc-vad.html](https://saifmohammad.com/WebPages/nrc-vad.html)
- **Used For**: Emotion/mood profiling in lyrics.

### 7. **NRCLex**

- **Description**: Python wrapper for NRC Emotion Lexicon.
- **Source**: [https://pypi.org/project/nrclex/](https://pypi.org/project/nrclex/)
- **Used For**: Emotion detection like joy, sadness, fear, anger.

---

## 🎧 Audio Feature Extraction

### 8. **Librosa**

- **Description**: A Python library for music and audio analysis.
- **Source**: [https://librosa.org/](https://librosa.org/)
- **Used For**: Extracting MFCCs, chroma features, tempo, spectral features, etc.

### 9. **PyAudioAnalysis**

- **Description**: A library for audio feature extraction and classification.
- **Source**: [https://github.com/tyiannak/pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis)
- **Used For**: Additional feature extraction and audio classification support.

---

## 🎼 Metadata & Tagging APIs

### 10. **MusicBrainz API**

- **Description**: Open music encyclopedia that collects music metadata.
- **Source**: [https://musicbrainz.org/doc/MusicBrainz_API](https://musicbrainz.org/doc/MusicBrainz_API)
- **Used For**: Artist networks, album info, genre hierarchy.

### 11. **Last.fm API**

- **Description**: Music tagging and user behavior API.
- **Source**: [https://www.last.fm/api](https://www.last.fm/api)
- **Used For**: User-driven genre tags, song tags, and artist trends.

---

## 📌 Summary Table

| Dataset/API          | Purpose                           | Source / Access              |
| -------------------- | --------------------------------- | ---------------------------- |
| MSD                  | Metadata                          | millionsongdataset.com       |
| MPD                  | User playlist context             | Spotify GitHub               |
| GTZAN (via DeepLake) | Genre classification              | hub://activeloop/gtzan-genre |
| Genius API           | Lyrics                            | genius.com API               |
| VADER                | Sentiment analysis                | vaderSentiment via PyPI      |
| NRC-VAD              | Valence-Arousal-Dominance lexicon | saifmohammad.com             |
| NRCLex               | Emotion lexicon                   | PyPI                         |
| Librosa              | Audio feature extraction          | librosa.org                  |
| PyAudioAnalysis      | Audio analysis                    | GitHub                       |
| MusicBrainz API      | Metadata + graph retrieval        | musicbrainz.org              |
| Last.fm API          | User tags, trends                 | last.fm/api                  |

---

> 💡 _This combination of datasets and APIs enables a rich multi-modal, user-centric, emotion-aware music recommendation system._
