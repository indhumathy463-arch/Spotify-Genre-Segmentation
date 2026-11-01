# 🎵 Spotify Songs' Genre Segmentation & AI Recommendation System

[![Python 3.11](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?logo=plotly&logoColor=white)](https://plotly.com)

An Artificial Intelligence and Machine Learning system that analyzes musical audio features, groups songs into distinct acoustic personas using **unsupervised K-Means clustering**, and recommends similar tracks via **Cosine Similarity**.

Developed for B.Sc. Computer Science AI Coursework.

---

## 📑 Table of Contents
1. [Project Overview](#-project-overview)
2. [Project File Structure](#-project-file-structure)
3. [Dataset Description](#-dataset-description)
4. [Machine Learning Methodology](#-machine-learning-methodology)
   - [Data Preprocessing & Deduplication](#1-data-preprocessing--deduplication)
   - [Exploratory Data Analysis (EDA)](#2-exploratory-data-analysis-eda)
   - [Correlation Analysis](#3-correlation-analysis)
   - [Unsupervised Clustering (K-Means)](#4-unsupervised-clustering-k-means)
   - [Content-Based Recommendation Engine](#5-content-based-recommendation-engine)
5. [Installation & Setup](#-installation--setup)
6. [How to Run the Jupyter Notebook](#-how-to-run-the-jupyter-notebook)
7. [How to Run the Streamlit Dashboard](#-how-to-run-the-streamlit-dashboard)
8. [Cluster Personas & Results](#-cluster-personas--results)
9. [Limitations & Future Improvements](#-limitations--future-improvements)

---

## 🌟 Project Overview
Streaming platforms like Spotify rely on acoustic analysis rather than manual tags to discover similar songs and curate personalized playlists. This project implements a complete data science lifecycle:
- Cleans and deduplicates **32,833 tracks** across **23 audio and metadata features**.
- Standardizes numerical audio features using `StandardScaler`.
- Performs detailed **Exploratory Data Analysis (EDA)** across genres, subgenres, and top playlists.
- Discovers optimal acoustic groupings using the **Elbow Method (WCSS)** and **Silhouette Scores**.
- Assigns tracks to **6 distinct musical personas**.
- Reduces dimensions using **Principal Component Analysis (PCA)** for 2D and 3D visual exploration.
- Builds a fast **content-based recommendation system** powered by **Cosine Similarity**.
- Presents an interactive, Spotify-themed web dashboard built with **Streamlit**.

---

## 📁 Project File Structure
```
Spotify_Genre_Segmentation/
│
├── dataset/
│   └── spotify dataset.csv              # Official dataset (32,833 tracks)
│
├── notebooks/
│   └── spotify_analysis.ipynb          # Complete runnable Jupyter notebook with plots
│
├── models/
│   └── saved_model.pkl                 # Serialized K-Means model, scaler, and PCA pipeline
│
├── app.py                              # Streamlit interactive web dashboard
├── model.py                            # Core ML pipeline & recommendation functions
├── generate_notebook.py                # Notebook builder script
├── requirements.txt                    # Project dependencies
├── saved_model.pkl                     # Root copy of trained pipeline
├── Project 2 - Spotify Genre Grouping.docx # Official project specification
└── README.md                           # Documentation and user manual
```

---

## 📊 Dataset Description
The dataset contains **32,833 rows** and **23 columns** representing Spotify playlist tracks:

| Column Name | Type | Description | Role in ML |
| :--- | :---: | :--- | :--- |
| `track_id` | String | Unique Spotify track alphanumeric ID | Deduplication key |
| `track_name` | String | Song title | Display & search |
| `track_artist` | String | Artist / Performer name | Display & search |
| `track_popularity` | Integer | Popularity index (0–100) | Metadata / Sorting |
| `playlist_genre` | String | Broad genre (`edm`, `rap`, `pop`, `r&b`, `latin`, `rock`) | Validation & EDA |
| `playlist_subgenre` | String | 24 specific musical styles | Detailed analysis |
| `danceability` | Float | Suitability for dancing (0.0 to 1.0) | Core ML Feature |
| `energy` | Float | Perceptual intensity & activity (0.0 to 1.0) | Core ML Feature |
| `key` | Integer | Pitch key notation (0 to 11) | Core ML Feature |
| `loudness` | Float | Overall loudness in decibels (-46 to 1.3 dB) | Core ML Feature |
| `mode` | Integer | Modality (Major = 1, Minor = 0) | Core ML Feature |
| `speechiness` | Float | Spoken word ratio (0.0 to 1.0) | Core ML Feature |
| `acousticness` | Float | Acoustic / unplugged confidence (0.0 to 1.0) | Core ML Feature |
| `instrumentalness` | Float | Absence of vocal content (0.0 to 1.0) | Core ML Feature |
| `liveness` | Float | Presence of audience / live venue (0.0 to 1.0) | Core ML Feature |
| `valence` | Float | Musical positiveness / cheerfulness (0.0 to 1.0) | Core ML Feature |
| `tempo` | Float | Estimated tempo in beats per minute (BPM) | Core ML Feature |
| `duration_ms` | Integer | Track length in milliseconds | Metadata |

---

## 🧠 Machine Learning Methodology

### 1. Data Preprocessing & Deduplication
- **Missing Values**: 5 records out of 32,833 lacked song titles and artist names (0.015%). These 5 records were dropped.
- **Duplicate Records**: In streaming services, a single popular track frequently appears in multiple curated playlists. Deduplication on `track_id` yielded **28,352 unique tracks**, while aggregating all playlist occurrences for analysis.
- **Feature Scaling**: Features like `loudness` (dB) and `tempo` (BPM) have different numeric scales compared to unit-bounded features (0.0–1.0). We apply `StandardScaler` to ensure Euclidean distance calculations in K-Means are unbiased:
  $$z = \frac{x - \mu}{\sigma}$$

### 2. Exploratory Data Analysis (EDA)
- **Genre Balance**: The dataset has roughly 5,000 to 6,000 tracks per genre (`edm`: 6,043, `rap`: 5,746, `pop`: 5,507, `r&b`: 5,431, `latin`: 5,155, `rock`: 4,951).
- **Audio Distributions**:
  - `danceability`: Bell curve centered at 0.65.
  - `energy` & `loudness`: Left-skewed toward high energy (0.70) and loudness (-6.7 dB).
  - `tempo`: Peaks at 120–130 BPM (dance/house) and 90–100 BPM (hip-hop).
  - `acousticness`: Right-skewed (< 0.20 for most commercial tracks).

### 3. Correlation Analysis
- **Loudness & Energy ($r = +0.68$)**: Strongest positive correlation; louder tracks convey higher energy.
- **Acousticness & Energy ($r = -0.54$)**: Strongest negative correlation; acoustic instruments contrast with electric/electronic soundscapes.
- **Danceability & Valence ($r = +0.33$)**: Tracks designed for dancing tend to sound happier.

### 4. Unsupervised Clustering (K-Means)
- Evaluated $k = 2$ to $8$ using the **Elbow Method (WCSS/Inertia)** and **Silhouette Coefficient**.
- Selected **$k = 6$** as the optimal cluster count, matching the 6 curated Spotify genres and producing interpretable personas.
- Visualized in 2D and 3D using **Principal Component Analysis (PCA)**.

### 5. Content-Based Recommendation Engine
- Uses **Cosine Similarity** over the 11 normalized audio features:
  $$\text{Cosine Similarity}(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|}$$
- Returns the top $N$ closest musical tracks, their acoustic match percentage, and an automated explanation of which audio attributes match most closely.

---

## 🎯 Cluster Personas & Results

| Cluster | Persona Name | Key Audio Characteristics | Dominant Genres |
| :---: | :--- | :--- | :--- |
| **0** | **Electronic & Deep Bass (EDM Pulse)** | Low acousticness (0.08), high electronic energy, fast tempos | EDM (57%), Pop |
| **1** | **Live Stage & High Vibrance** | Extremely high liveness (0.62), stadium atmosphere | EDM, Rock, Rap |
| **2** | **Hard Rock & Driving Anthems** | High loudness (-5.3 dB), intense energy (0.79), rock tempo | Rock (26%), EDM, Pop |
| **3** | **Chill Acoustic & Mellow Soul** | Low loudness (-10.6 dB), soft energy (0.42), acoustic warm tone | R&B (32%), Pop, Rock |
| **4** | **Upbeat Latin & Dance Pop** | High danceability (0.74), cheerful positive valence (0.68) | Latin (25%), Pop (21%), R&B |
| **5** | **Rhythmic Rap & Urban Hip-Hop** | High speechiness (>0.20), punchy drums, rhythmic flow | Rap (57%), R&B |

> **Validation Note**: The unsupervised algorithm successfully grouped **Rap tracks into Cluster 5 with >60% accuracy** and **EDM into Cluster 0** based purely on acoustic measurements without access to genre tags during training.

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12 installed on your laptop.
- Windows, macOS, or Linux.

### Step 1: Open Terminal / PowerShell
Navigate to the project root folder:
```bash
cd "path/to/Spotify_Genre_Segmentation"
```

### Step 2: Create and Activate a Virtual Environment (Optional but Recommended)
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 📓 How to Run the Jupyter Notebook
You can open and run `notebooks/spotify_analysis.ipynb` in **VS Code**, **Jupyter Notebook**, or **JupyterLab**:

```bash
jupyter notebook notebooks/spotify_analysis.ipynb
```
Or open the folder directly in VS Code and select the Python kernel.  
All cells have been pre-executed, so all graphs, metrics, and tables are immediately visible upon opening.

---

## 🚀 How to Run the Streamlit Dashboard
Launch the interactive web application with:

```bash
streamlit run app.py
```

The application will automatically open in your default browser at:
```
http://localhost:8501
```

### Website Structure & Navigation Flow:
1. **🏠 Home (Welcome Landing Page)**: A clean Spotify-inspired hero section featuring the project title, auditory AI introduction, key highlights, and a prominent **"Get Started"** call-to-action button.
2. **🗂️ Module Selection Page**: Opens when the user clicks "Get Started", presenting the 7 core project modules as clickable cards with **"View Details"** buttons.
3. **🔍 Overview & Dataset Explorer**: Filter 32,833 tracks by genre, search songs or artists, and view descriptive statistics.
4. **📊 Exploratory Data Analysis**: Interactive Plotly histograms, genre bar charts, subgenre rankings, top playlists, and bivariate comparisons.
5. **🔗 Feature Correlations**: Interactive Pearson correlation heatmap and positive/negative relationship takeaway cards.
6. **🎯 K-Means Segmentation**: Persona profile cards, 5D audio radar chart, 2D/3D PCA cluster scatter plots, and cross-genre distributions.
7. **🎧 Song Recommender**: Search any track from 28k unique songs, view its audio signature radar, and get top recommendations with match percentages and explanations.
8. **🧠 Project Methodology & Architecture**: In-depth academic documentation of data preprocessing, Z-score standardization, Elbow method, and Cosine Similarity math.
9. **📚 Audio Feature Dictionary**: Comprehensive table explaining all 11 Spotify audio dimensions, ranges, and musical significance.
*Complete Navigation System: Clean sidebar branding without unnecessary statistics, instant "Back to Modules" header, and "Previous Module / Back to Modules / Back to Home / Next Module" footers on every content page.*

---

## ⚠️ Limitations & Future Improvements
1. **Tabular Features vs. Raw Waveforms**: The Spotify Web API provides pre-calculated aggregate audio features. In future work, deep learning models (e.g., CNNs on Mel-spectrograms) could extract subtle instrumental timbres directly from raw audio.
2. **Hybrid Recommendation**: Combining content-based audio similarity with collaborative filtering (user listening history, skip rate, thumbs up/down) would create a commercial-grade hybrid recommendation system.
3. **Cluster Boundary Overlap**: Genres like Pop and Latin frequently share similar danceability and valence, leading to smooth cluster boundaries rather than hard partitions.

