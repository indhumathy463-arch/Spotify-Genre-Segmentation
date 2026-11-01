"""
Spotify Songs' Genre Segmentation & Recommendation System
Interactive Multi-Page Web Application
---------------------------------------------------------
Author: AI Machine Learning Engineer
Theme: Modern Spotify Dark Theme (High Contrast & Clean Architecture)
"""

import os
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity

# Import core machine learning and preprocessing pipeline
from model import (
    AUDIO_FEATURES,
    CLUSTER_PERSONAS,
    load_pipeline,
    load_raw_data,
    preprocess_data,
    train_clustering_pipeline,
    search_songs,
    recommend_similar_songs
)

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Spotify Genre Segmentation & AI Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. SPOTIFY DARK THEME CSS (HIGH CONTRAST & READABILITY)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global Background and Fonts */
    .stApp {
        background-color: #121212 !important;
        color: #FFFFFF !important;
        font-family: 'CircularStd', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Headings & Subheadings - Crisp Pure White */
    h1, h2, h3, h4, h5, h6, 
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    
    /* Body & Paragraph Text - Light Readable Grey */
    .stMarkdown p, .stMarkdown li, .stMarkdown span, p, span, li {
        color: #E5E7EB !important;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    /* Bold text inside markdown */
    .stMarkdown b, .stMarkdown strong, strong, b {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    
    /* Captions & Small descriptions - Bright Silver */
    .stCaption, [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p, small {
        color: #D1D5DB !important;
        font-size: 0.92rem !important;
        font-weight: 500 !important;
    }
    
    /* KPI Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #181818 !important;
        border: 1px solid #2e2e2e !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
    }
    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] *,
    div[data-testid="stMetricLabel"] p {
        color: #F3F4F6 !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em;
    }
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] * {
        color: #1DB954 !important;
        font-weight: 800 !important;
        font-size: 2rem !important;
        text-shadow: 0 0 12px rgba(29, 185, 84, 0.25);
    }
    
    /* Form Labels, Search Labels & Filter Labels */
    label[data-testid="stWidgetLabel"],
    label[data-testid="stWidgetLabel"] p,
    label[data-testid="stWidgetLabel"] span,
    div[data-testid="stWidgetLabel"] label,
    label {
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 0.98rem !important;
        margin-bottom: 6px !important;
    }
    
    /* Text Inputs, Select Boxes, Sliders */
    div[data-baseweb="input"] {
        background-color: #242424 !important;
        border: 1px solid #3e3e3e !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="input"] input {
        color: #FFFFFF !important;
        background-color: #242424 !important;
        font-size: 0.95rem !important;
    }
    div[data-baseweb="input"] input::placeholder {
        color: #9CA3AF !important;
    }
    div[data-baseweb="select"] > div {
        background-color: #242424 !important;
        border: 1px solid #3e3e3e !important;
        border-radius: 8px !important;
        color: #FFFFFF !important;
    }
    div[data-baseweb="select"] span {
        color: #FFFFFF !important;
        font-size: 0.95rem !important;
    }
    div[data-testid="stCheckbox"] label span {
        color: #E5E7EB !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0b0b0b !important;
        border-right: 1px solid #282828 !important;
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] h4 {
        color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        color: #E5E7EB !important;
    }
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: #D1D5DB !important;
        font-size: 0.88rem !important;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label span {
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    
    /* Navigation Tabs */
    button[data-baseweb="tab"] {
        color: #D1D5DB !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 10px 18px !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #1DB954 !important;
        font-weight: 700 !important;
        border-bottom: 3px solid #1DB954 !important;
    }
    button[data-baseweb="tab"]:hover {
        color: #FFFFFF !important;
    }
    
    /* Spotify Custom Cards */
    .spotify-card {
        background-color: #181818;
        border: 1px solid #2e2e2e;
        border-radius: 12px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .spotify-card:hover {
        border-color: #1DB954;
        box-shadow: 0 6px 24px rgba(29, 185, 84, 0.2);
    }
    .spotify-card h3, .spotify-card h4 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    .spotify-card p, .spotify-card li {
        color: #E5E7EB !important;
        font-size: 0.95rem !important;
    }
    .spotify-card b, .spotify-card strong {
        color: #FFFFFF !important;
    }

    /* Landing Page Hero Container */
    .hero-container {
        background: linear-gradient(145deg, rgba(29, 185, 84, 0.12) 0%, rgba(24, 24, 24, 0.92) 40%, rgba(18, 18, 18, 1) 100%);
        border: 1px solid rgba(29, 185, 84, 0.35);
        border-radius: 20px;
        padding: 48px 36px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6);
    }
    .hero-badge {
        display: inline-block;
        background-color: rgba(29, 185, 84, 0.15);
        color: #1ed760;
        border: 1px solid #1DB954;
        padding: 6px 18px;
        border-radius: 30px;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 18px;
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #FFFFFF;
        line-height: 1.2;
        margin-bottom: 16px;
        letter-spacing: -0.03em;
    }
    .hero-subtitle {
        font-size: 1.15rem;
        color: #D1D5DB;
        max-width: 820px;
        margin: 0 auto 28px auto;
        line-height: 1.6;
    }
    .hero-feature-pill {
        display: inline-block;
        background-color: #1f1f1f;
        border: 1px solid #333333;
        border-radius: 24px;
        padding: 8px 18px;
        margin: 6px;
        font-size: 0.92rem;
        color: #F3F4F6;
        font-weight: 500;
    }
    .hero-feature-pill b {
        color: #1DB954;
    }

    /* Module Selection Card */
    .module-card {
        background-color: #181818;
        border: 1px solid #2e2e2e;
        border-radius: 14px;
        padding: 24px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .module-card:hover {
        border-color: #1DB954;
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(29, 185, 84, 0.22);
    }
    .module-icon {
        font-size: 2.2rem;
        margin-bottom: 12px;
    }
    .module-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 8px;
    }
    .module-desc {
        color: #D1D5DB;
        font-size: 0.93rem;
        line-height: 1.5;
        margin-bottom: 16px;
        flex-grow: 1;
    }
    
    /* Primary Spotify Action Buttons */
    .stButton > button {
        background-color: #1DB954 !important;
        color: #000000 !important;
        font-weight: 700 !important;
        border-radius: 500px !important;
        padding: 10px 24px !important;
        border: none !important;
        transition: all 0.2s ease !important;
        font-size: 0.95rem !important;
    }
    .stButton > button:hover {
        background-color: #1ed760 !important;
        transform: scale(1.03) !important;
        color: #000000 !important;
    }

    /* Secondary / Back Buttons */
    .secondary-btn > button {
        background-color: #242424 !important;
        color: #FFFFFF !important;
        border: 1px solid #3e3e3e !important;
    }
    .secondary-btn > button:hover {
        background-color: #333333 !important;
        border-color: #1DB954 !important;
        color: #1ed760 !important;
    }
    
    /* Genre & Cluster Tag Pills */
    .genre-pill {
        display: inline-block;
        background-color: #222222;
        color: #1ed760 !important;
        border: 1px solid #1DB954;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.88rem;
        font-weight: 700;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    
    /* Expanders */
    div[data-testid="stExpander"] {
        background-color: #181818 !important;
        border: 1px solid #2e2e2e !important;
        border-radius: 10px !important;
    }
    div[data-testid="stExpander"] details summary span p {
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    /* Alert / Information / Callout Boxes */
    div[data-testid="stAlert"] {
        background-color: #16241a !important;
        border: 1px solid #1DB954 !important;
        border-radius: 10px !important;
    }
    div[data-testid="stAlert"] p {
        color: #FFFFFF !important;
        font-weight: 500 !important;
    }
    
    /* Tables */
    [data-testid="stTable"] th {
        color: #1DB954 !important;
        font-weight: 700 !important;
        background-color: #202020 !important;
        font-size: 0.95rem !important;
    }
    [data-testid="stTable"] td {
        color: #F3F4F6 !important;
        background-color: #161616 !important;
        font-size: 0.92rem !important;
    }

    /* Code blocks */
    code {
        color: #1ed760 !important;
        background-color: #242424 !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
    }
    
    /* Divider */
    hr {
        border-color: #2e2e2e !important;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 3. DATA & MODEL CACHING
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Spotify AI Pipeline...")
def get_pipeline():
    """Load cached pipeline or train it if not found."""
    try:
        pipeline = load_pipeline('saved_model.pkl')
    except Exception:
        df_raw = load_raw_data()
        df_unique, X_scaled, scaler = preprocess_data(df_raw, deduplicate=True)
        pipeline = train_clustering_pipeline(df_unique, X_scaled, n_clusters=6)
    return pipeline

@st.cache_data(show_spinner="Loading raw dataset for EDA...")
def get_raw_data():
    """Load full dataset for playlist and genre level analysis."""
    return load_raw_data()


# -----------------------------------------------------------------------------
# 4. LOAD RESOURCES & DEFINE PAGES
# -----------------------------------------------------------------------------
pipeline = get_pipeline()
df_unique = pipeline['df_unique']
X_scaled = pipeline['X_scaled']
df_raw = get_raw_data()

# Canonical pages list
PAGES = [
    "🏠 Home",
    "🗂️ Module Selection",
    "🔍 Overview & Dataset Explorer",
    "📊 Exploratory Data Analysis",
    "🔗 Feature Correlations",
    "🎯 K-Means Segmentation",
    "🎧 Song Recommender",
    "🧠 Project Methodology & Architecture",
    "📚 Audio Feature Dictionary"
]

# The 7 core content modules in sequence for Previous/Next navigation
CONTENT_MODULES = [
    "🔍 Overview & Dataset Explorer",
    "📊 Exploratory Data Analysis",
    "🔗 Feature Correlations",
    "🎯 K-Means Segmentation",
    "🎧 Song Recommender",
    "🧠 Project Methodology & Architecture",
    "📚 Audio Feature Dictionary"
]

# -----------------------------------------------------------------------------
# 5. ROBUST NAVIGATION STATE MANAGEMENT
# -----------------------------------------------------------------------------

if 'current_page' not in st.session_state:
    st.session_state['current_page'] = "🏠 Home"

def navigate_to(page_name):
    """Safely updates page state and reruns the app."""
    st.session_state['current_page'] = page_name
    st.rerun()

def on_sidebar_change():
    """Triggered when user clicks sidebar radio."""
    st.session_state['current_page'] = st.session_state['sidebar_nav']
# -----------------------------------------------------------------------------
# 6. CLEAN SIDEBAR (BRANDING & NAVIGATION ONLY)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://storage.googleapis.com/pr-newsroom-wp/1/2018/11/Spotify_Logo_RGB_Green.png", width=165)
    st.markdown("### **Spotify AI**")
    st.caption("Genre Segmentation & Recommendation")
    st.markdown("---")
    
    st.radio(
        "Website Navigation",
        PAGES,
        key="sidebar_nav",
        on_change=on_sidebar_change
    )
    st.markdown("---")
    st.caption("Machine Learning Web Application")


# -----------------------------------------------------------------------------
# 7. NAVIGATION COMPONENT HELPERS
# -----------------------------------------------------------------------------
def render_module_header():
    """Top bar on every module page allowing instant return to Module Selection."""
    top_col1, top_col2 = st.columns([1, 5])
    with top_col1:
        if st.button("⬅️ Back to Modules", key=f"top_back_{st.session_state['current_page']}"):
            navigate_to("🗂️ Module Selection")
    st.markdown("---")

def render_module_footer(current_module):
    """
    Standardized footer for the 7 content modules.
    Provides: Previous Module | Back to Modules | Back to Home | Next Module
    """
    st.markdown("---")
    idx = CONTENT_MODULES.index(current_module)
    prev_mod = CONTENT_MODULES[idx - 1] if idx > 0 else None
    next_mod = CONTENT_MODULES[idx + 1] if idx < len(CONTENT_MODULES) - 1 else None
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if prev_mod:
            prev_label = prev_mod.split(" ", 1)[1] if " " in prev_mod else prev_mod
            if st.button(f"⬅️ Previous: {prev_label[:14]}..", key=f"btn_prev_{current_module}", use_container_width=True):
                navigate_to(prev_mod)
    with col2:
        if st.button("🗂️ Back to Modules", key=f"btn_mods_{current_module}", use_container_width=True):
            navigate_to("🗂️ Module Selection")
    with col3:
        if st.button("🏠 Back to Home", key=f"btn_home_{current_module}", use_container_width=True):
            navigate_to("🏠 Home")
    with col4:
        if next_mod:
            next_label = next_mod.split(" ", 1)[1] if " " in next_mod else next_mod
            if st.button(f"Next: {next_label[:14]}.. ➡️", key=f"btn_next_{current_module}", use_container_width=True):
                navigate_to(next_mod)


# =============================================================================
# PAGE 1: 🏠 HOME / LANDING PAGE
# =============================================================================
if st.session_state['current_page'] == "🏠 Home":
    # Hero Welcome Section
    st.markdown("""
    <div class="hero-container">
        <div class="hero-badge">🎵 AI Audio Intelligence & Segmentation</div>
        <div class="hero-title">Spotify Songs' Genre Segmentation & Recommender</div>
        <div class="hero-subtitle">
            An end-to-end Machine Learning platform that decodes musical properties through 
            <b>auditory feature extraction</b>, <b>unsupervised K-Means clustering</b>, 
            and <b>high-dimensional vector similarity</b>.
        </div>
        <div style="margin-bottom: 25px;">
            <span class="hero-feature-pill">📂 <b>32,833</b> Playlist Records</span>
            <span class="hero-feature-pill">🎵 <b>28,352</b> Unique Songs</span>
            <span class="hero-feature-pill">🎚️ <b>11</b> Scaled Audio Dimensions</span>
            <span class="hero-feature-pill">🎯 <b>6</b> Acoustic Personas</span>
            <span class="hero-feature-pill">⚡ <b>Cosine</b> Recommendation Engine</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Prominent Call to Action Button
    cta_col1, cta_col2, cta_col3 = st.columns([1, 1.2, 1])
    with cta_col2:
        if st.button("🚀 Get Started / Explore Project →", key="btn_hero_get_started", use_container_width=True):
            navigate_to("🗂️ Module Selection")
            
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("💡 **Welcome**: Click **Get Started** above to view the 7 project modules, or use the sidebar navigation to jump into any analysis.")


# =============================================================================
# PAGE 2: 🗂️ MODULE SELECTION PAGE
# =============================================================================
elif st.session_state['current_page'] == "🗂️ Module Selection":
    # Header & Back to Home
    top_c1, top_c2 = st.columns([1, 5])
    with top_c1:
        if st.button("🏠 Back to Home", key="btn_mod_sel_home"):
            navigate_to("🏠 Home")
            
    st.title("🗂️ Project Modules")
    st.markdown("Select any module below to explore its data analysis, machine learning models, or recommendation tools:")
    st.markdown("---")
    
    # Row 1: 3 modules
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="module-card">
            <div>
                <div class="module-icon">🔍</div>
                <div class="module-title">Overview & Dataset</div>
                <div class="module-desc">
                    Inspect the Spotify catalog, filter tracks by genre, search track titles and artists, and view descriptive statistics.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("View Details →", key="card_btn_dataset", use_container_width=True):
            navigate_to("🔍 Overview & Dataset Explorer")
            
    with col2:
        st.markdown("""
        <div class="module-card">
            <div>
                <div class="module-icon">📊</div>
                <div class="module-title">Exploratory Data Analysis</div>
                <div class="module-desc">
                    Interactive visualizations covering genre distributions, 24 subgenres, top playlists, and 8 core audio feature distributions.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("View Details →", key="card_btn_eda", use_container_width=True):
            navigate_to("📊 Exploratory Data Analysis")

    with col3:
        st.markdown("""
        <div class="module-card">
            <div>
                <div class="module-icon">🔗</div>
                <div class="module-title">Feature Correlations</div>
                <div class="module-desc">
                    Interactive Pearson correlation heatmap revealing relationships between loudness, dynamic energy, acousticness, and mood.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("View Details →", key="card_btn_corr", use_container_width=True):
            navigate_to("🔗 Feature Correlations")

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2: 2 modules
    col4, col5 = st.columns(2)
    
    with col4:
        st.markdown("""
        <div class="module-card">
            <div>
                <div class="module-icon">🎯</div>
                <div class="module-title">K-Means Segmentation</div>
                <div class="module-desc">
                    Explore the 6 unsupervised acoustic personas, radar profile charts, and interactive 2D and 3D PCA cluster scatter plots.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("View Details →", key="card_btn_kmeans", use_container_width=True):
            navigate_to("🎯 K-Means Segmentation")

    with col5:
        st.markdown("""
        <div class="module-card">
            <div>
                <div class="module-icon">🎧</div>
                <div class="module-title">Song Recommender</div>
                <div class="module-desc">
                    Search any song in the 28k dataset and discover acoustically similar tracks using Cosine Similarity with transparent explanations.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("View Details →", key="card_btn_rec", use_container_width=True):
            navigate_to("🎧 Song Recommender")

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 3: 2 modules
    col6, col7 = st.columns(2)
    
    with col6:
        st.markdown("""
        <div class="module-card">
            <div>
                <div class="module-icon">🧠</div>
                <div class="module-title">Methodology & Architecture</div>
                <div class="module-desc">
                    In-depth documentation of Z-score normalization, WCSS/Elbow discovery, PCA projection, and Cosine Similarity equations.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("View Details →", key="card_btn_method", use_container_width=True):
            navigate_to("🧠 Project Methodology & Architecture")

    with col7:
        st.markdown("""
        <div class="module-card">
            <div>
                <div class="module-icon">📚</div>
                <div class="module-title">Audio Feature Dictionary</div>
                <div class="module-desc">
                    Comprehensive reference table explaining all 11 Spotify audio dimensions, numerical ranges, and psychoacoustic significance.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("View Details →", key="card_btn_dict", use_container_width=True):
            navigate_to("📚 Audio Feature Dictionary")


# =============================================================================
# PAGE 3: 🔍 OVERVIEW & DATASET EXPLORER
# =============================================================================
elif st.session_state['current_page'] == "🔍 Overview & Dataset Explorer":
    render_module_header()
    
    st.title("🔍 Overview & Dataset Explorer")
    st.markdown("""
    Explore the curated Spotify dataset containing **32,833 playlist tracks**.
    Filter by genre, search track names or artists, and inspect descriptive audio statistics.
    """)
    
    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Playlist Records", f"{len(df_raw):,}")
    with col2:
        st.metric("Unique Catalog Songs", f"{len(df_unique):,}")
    with col3:
        st.metric("Curated Genres", f"{df_raw['playlist_genre'].nunique()}")
    with col4:
        st.metric("Audio Features Scaled", f"{len(AUDIO_FEATURES)}")
    
    st.markdown("---")
    st.subheader("Filter and Search Tracks")
    
    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        selected_genre = st.selectbox(
            "Filter by Playlist Genre:",
            ["All"] + sorted(df_raw['playlist_genre'].unique().tolist())
        )
    with col_filter2:
        search_kw = st.text_input("Search Track Name or Artist:", placeholder="e.g. Ed Sheeran, Queen, Drake")
    
    filtered_df = df_raw.copy()
    if selected_genre != "All":
        filtered_df = filtered_df[filtered_df['playlist_genre'] == selected_genre]
    if search_kw.strip():
        mask = (
            filtered_df['track_name'].str.lower().str.contains(search_kw.lower(), na=False) |
            filtered_df['track_artist'].str.lower().str.contains(search_kw.lower(), na=False)
        )
        filtered_df = filtered_df[mask]
        
    st.caption(f"Showing {min(len(filtered_df), 15)} of {len(filtered_df):,} matching tracks:")
    display_cols = ['track_name', 'track_artist', 'playlist_genre', 'playlist_subgenre', 'danceability', 'energy', 'loudness', 'tempo', 'track_popularity']
    st.dataframe(
        filtered_df[display_cols].head(15).reset_index(drop=True),
        use_container_width=True
    )
    
    with st.expander("📈 View Descriptive Statistics of Audio Features"):
        st.dataframe(df_unique[AUDIO_FEATURES].describe().T[['mean', 'std', 'min', '50%', 'max']].round(3), use_container_width=True)

    render_module_footer("🔍 Overview & Dataset Explorer")


# =============================================================================
# PAGE 4: 📊 EXPLORATORY DATA ANALYSIS
# =============================================================================
elif st.session_state['current_page'] == "📊 Exploratory Data Analysis":
    render_module_header()
    
    st.title("📊 Exploratory Data Analysis (EDA)")
    st.markdown("Analyze how musical properties and genres distribute across Spotify's curated catalog.")
    
    tab1, tab2, tab3 = st.tabs(["🎼 Genre & Playlist Distributions", "🔊 Audio Feature Distributions", "⚖️ Feature Comparisons Across Genres"])
    
    # Tab 1: Genres & Playlists
    with tab1:
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            genre_counts = df_raw['playlist_genre'].value_counts().reset_index()
            genre_counts.columns = ['Genre', 'Track Count']
            fig_genre = px.bar(
                genre_counts, x='Genre', y='Track Count',
                color='Genre',
                color_discrete_sequence=px.colors.sequential.Viridis,
                title="<b>Songs per Playlist Genre</b>"
            )
            fig_genre.update_layout(paper_bgcolor='#181818', plot_bgcolor='#181818', font_color='#FFFFFF')
            st.plotly_chart(fig_genre, use_container_width=True)
            st.info("💡 **Insight**: All 6 genres have roughly ~5,000 to 6,000 tracks, providing an exceptionally balanced dataset for clustering.")
            
        with col_g2:
            top_sub = df_raw['playlist_subgenre'].value_counts().head(10).reset_index()
            top_sub.columns = ['Subgenre', 'Track Count']
            fig_sub = px.bar(
                top_sub, x='Track Count', y='Subgenre', orientation='h',
                color='Track Count',
                color_continuous_scale='Mint',
                title="<b>Top 10 Playlist Subgenres</b>"
            )
            fig_sub.update_layout(paper_bgcolor='#181818', plot_bgcolor='#181818', font_color='#FFFFFF', yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_sub, use_container_width=True)
            st.info("💡 **Insight**: Progressive electro house, southern hip hop, and indie poptimism are the most represented subgenres.")

        st.subheader("Top Playlists with Most Tracks")
        top_play = df_raw['playlist_name'].value_counts().head(10).reset_index()
        top_play.columns = ['Playlist Name', 'Songs in Dataset']
        fig_play = px.bar(
            top_play, x='Songs in Dataset', y='Playlist Name', orientation='h',
            color='Songs in Dataset',
            color_continuous_scale='Teal',
            title="<b>Top Curated Playlists</b>"
        )
        fig_play.update_layout(paper_bgcolor='#181818', plot_bgcolor='#181818', font_color='#FFFFFF', yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_play, use_container_width=True)

    # Tab 2: Feature Distributions
    with tab2:
        feat_choice = st.selectbox(
            "Select Audio Feature to Inspect Distribution:",
            ['danceability', 'energy', 'loudness', 'tempo', 'acousticness', 'speechiness', 'valence', 'track_popularity']
        )
        fig_hist = px.histogram(
            df_unique, x=feat_choice, nbins=40, marginal="box",
            color_discrete_sequence=['#1DB954'],
            title=f"<b>Distribution of {feat_choice.capitalize()}</b>"
        )
        fig_hist.update_layout(paper_bgcolor='#181818', plot_bgcolor='#181818', font_color='#FFFFFF')
        st.plotly_chart(fig_hist, use_container_width=True)
        
        explanations = {
            'danceability': "Measures how suitable a track is for dancing based on tempo, rhythm stability, beat strength, and regularity. Centered around 0.65.",
            'energy': "A perceptual measure of intensity and activity (0.0 to 1.0). High energy tracks feel fast, loud, and noisy. Left-skewed toward high values (~0.70).",
            'loudness': "Overall loudness in decibels (dB), averaged across the entire track. Ranges between -46 dB and 1.3 dB, peaking between -8 dB and -5 dB.",
            'tempo': "The overall estimated pace of a track in beats per minute (BPM). Major peaks occur at 120-130 BPM (dance/house) and 90-100 BPM (hip-hop).",
            'acousticness': "Confidence measure from 0.0 to 1.0 of whether the track is acoustic. Strongly right-skewed; contemporary music is predominantly electronic/amplified.",
            'speechiness': "Detects the presence of spoken words in a track. Most music scores below 0.15, with rap and hip-hop scoring above 0.20.",
            'valence': "Musical positiveness (0.0 to 1.0). High valence sounds happy/cheerful, whereas low valence sounds sad/depressive. Centered symmetrically around 0.51.",
            'track_popularity': "Spotify calculated score (0-100) based on total play counts and recency of streams."
        }
        st.info(f"💡 **Feature Description**: {explanations.get(feat_choice, '')}")

    # Tab 3: Feature Comparisons
    with tab3:
        st.subheader("Bivariate Feature Comparison")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            x_axis = st.selectbox("X-Axis Feature:", AUDIO_FEATURES, index=0)
        with col_c2:
            y_axis = st.selectbox("Y-Axis Feature:", AUDIO_FEATURES, index=1)
            
        sample_eda = df_unique.sample(n=min(len(df_unique), 2500), random_state=42)
        fig_scatter = px.scatter(
            sample_eda, x=x_axis, y=y_axis, color='playlist_genre',
            hover_data=['track_name', 'track_artist'],
            color_discrete_sequence=px.colors.qualitative.Plotly,
            title=f"<b>{x_axis.capitalize()} vs. {y_axis.capitalize()} by Genre</b>"
        )
        fig_scatter.update_layout(paper_bgcolor='#181818', plot_bgcolor='#181818', font_color='#FFFFFF')
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.subheader("Audio Feature Boxplots by Genre")
        box_feat = st.selectbox("Select Feature for Boxplot Comparison:", ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'valence'])
        fig_box = px.box(
            df_raw, x='playlist_genre', y=box_feat, color='playlist_genre',
            title=f"<b>{box_feat.capitalize()} Across Playlist Genres</b>"
        )
        fig_box.update_layout(paper_bgcolor='#181818', plot_bgcolor='#181818', font_color='#FFFFFF')
        st.plotly_chart(fig_box, use_container_width=True)

    render_module_footer("📊 Exploratory Data Analysis")


# =============================================================================
# PAGE 5: 🔗 FEATURE CORRELATIONS
# =============================================================================
elif st.session_state['current_page'] == "🔗 Feature Correlations":
    render_module_header()
    
    st.title("🔗 Correlation Matrix Analysis")
    st.markdown("""
    The correlation matrix calculates **Pearson's Correlation Coefficient ($r$)** across audio features.
    Values range from **-1.0 (perfect inverse relationship)** to **+1.0 (perfect positive relationship)**.
    """)
    
    corr = df_unique[AUDIO_FEATURES].corr().round(2)
    
    fig_corr = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="<b>Interactive Correlation Heatmap of Spotify Audio Features</b>"
    )
    fig_corr.update_layout(paper_bgcolor='#181818', plot_bgcolor='#181818', font_color='#FFFFFF')
    st.plotly_chart(fig_corr, use_container_width=True)
    
    col_cor1, col_cor2 = st.columns(2)
    with col_cor1:
        st.markdown("""
        <div class="spotify-card">
            <h4 style="color: #1DB954;">🔥 Strong Positive Correlations</h4>
            <ul>
                <li><b>Loudness & Energy (+0.68)</b>: Tracks mastered at louder decibel levels naturally convey greater psychological intensity and energetic drive.</li>
                <li><b>Danceability & Valence (+0.33)</b>: Music that encourages dancing tends to carry brighter, happier emotional valence.</li>
                <li><b>Energy & Valence (+0.30)</b>: High-energy songs often project positive mood.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col_cor2:
        st.markdown("""
        <div class="spotify-card">
            <h4 style="color: #e74c3c;">❄️ Strong Negative Correlations</h4>
            <ul>
                <li><b>Acousticness & Energy (-0.54)</b>: Acoustic recordings (acoustic guitar, piano) contrast directly with intense, synthetic dance and rock music.</li>
                <li><b>Acousticness & Loudness (-0.36)</b>: Acoustic instruments retain dynamic range, avoiding aggressive modern loudness compression.</li>
                <li><b>Danceability & Tempo (-0.18)</b>: Very fast tempos (150+ BPM) make body movement harder than moderate dance grooves (100-125 BPM).</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    render_module_footer("🔗 Feature Correlations")


# =============================================================================
# PAGE 6: 🎯 K-MEANS SEGMENTATION
# =============================================================================
elif st.session_state['current_page'] == "🎯 K-Means Segmentation":
    render_module_header()
    
    st.title("🎯 K-Means Clustering & Song Segmentation")
    st.markdown("""
    We use **unsupervised K-Means clustering** to partition 28,352 unique Spotify tracks into **6 distinct acoustic personas**
    without using human genre labels during training.
    """)
    
    tab_m1, tab_m2, tab_m3 = st.tabs(["🌟 Cluster Personas & Radar Profiles", "📍 2D & 3D PCA Visualization", "📊 Clusters vs. Playlist Genres"])
    
    # Tab 1: Personas
    with tab_m1:
        st.subheader("Acoustic Personas Overview")
        cluster_means = df_unique.groupby('cluster')[AUDIO_FEATURES].mean().round(3)
        
        cols = st.columns(3)
        for c in range(6):
            persona = CLUSTER_PERSONAS[c]
            count = (df_unique['cluster'] == c).sum()
            with cols[c % 3]:
                st.markdown(f"""
                <div class="spotify-card">
                    <h3>{persona['icon']} Cluster {c}</h3>
                    <h4 style="color: #1DB954; margin-top:-10px;">{persona['name']}</h4>
                    <p style="color: #E5E7EB; font-size: 0.92rem; line-height: 1.5;">{persona['desc']}</p>
                    <p style="color: #E5E7EB; margin-top: 8px;"><b>Tracks</b>: <span style="color:#1DB954; font-weight:bold;">{count:,}</span> ({count/len(df_unique)*100:.1f}%)</p>
                </div>
                """, unsafe_allow_html=True)
                
        st.subheader("Radar Chart: Audio Feature Profile per Cluster")
        radar_feats = ['danceability', 'energy', 'speechiness', 'acousticness', 'valence']
        fig_radar = go.Figure()
        
        for c in range(6):
            vals = cluster_means.loc[c, radar_feats].tolist()
            vals += [vals[0]]  # close loop
            fig_radar.add_trace(go.Scatterpolar(
                r=vals,
                theta=[f.capitalize() for f in radar_feats] + [radar_feats[0].capitalize()],
                name=f"Cluster {c}: {CLUSTER_PERSONAS[c]['name'].split('(')[0]}"
            ))
            
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            paper_bgcolor='#181818',
            font_color='#FFFFFF',
            title="<b>Centroid Comparison Across Audio Dimensions</b>"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Tab 2: PCA
    with tab_m2:
        st.subheader("Principal Component Analysis (PCA) Projection")
        st.markdown("Reduces 11-dimensional audio space to 2D and 3D visual coordinates.")
        
        view_mode = st.radio("Select View Mode:", ["2D PCA Projection", "3D Interactive PCA Projection"], horizontal=True)
        sample_pca = df_unique.sample(n=min(len(df_unique), 3500), random_state=42)
        
        if view_mode == "2D PCA Projection":
            fig_pca2 = px.scatter(
                sample_pca, x='pca_x', y='pca_y',
                color='cluster_name',
                hover_data=['track_name', 'track_artist', 'playlist_genre'],
                color_discrete_sequence=px.colors.qualitative.Dark24,
                title="<b>2D PCA Projection of Song Clusters</b>"
            )
            fig_pca2.update_layout(paper_bgcolor='#181818', plot_bgcolor='#181818', font_color='#FFFFFF')
            st.plotly_chart(fig_pca2, use_container_width=True)
        else:
            fig_pca3 = px.scatter_3d(
                sample_pca, x='pca_x', y='pca_y', z='pca_z',
                color='cluster_name',
                hover_data=['track_name', 'track_artist'],
                color_discrete_sequence=px.colors.qualitative.Dark24,
                title="<b>3D Interactive PCA Cluster Space</b>"
            )
            fig_pca3.update_layout(
                paper_bgcolor='#181818',
                scene=dict(
                    xaxis=dict(backgroundcolor='#181818', color='#FFFFFF'),
                    yaxis=dict(backgroundcolor='#181818', color='#FFFFFF'),
                    zaxis=dict(backgroundcolor='#181818', color='#FFFFFF')
                ),
                font_color='#FFFFFF'
            )
            st.plotly_chart(fig_pca3, use_container_width=True)

    # Tab 3: Clusters vs Genres
    with tab_m3:
        st.subheader("Cluster vs. Playlist Genre Crosstabulation")
        st.markdown("Shows how unsupervised acoustic clusters correspond with Spotify's genre classifications.")
        
        ct = pd.crosstab(df_unique['cluster_name'], df_unique['playlist_genre'], normalize='index') * 100
        fig_ct = px.bar(
            ct.reset_index(),
            x=ct.columns.tolist(),
            y='cluster_name',
            orientation='h',
            title="<b>Playlist Genre Distribution Within Each Audio Cluster (%)</b>",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_ct.update_layout(paper_bgcolor='#181818', plot_bgcolor='#181818', font_color='#FFFFFF')
        st.plotly_chart(fig_ct, use_container_width=True)
        st.success("✅ **Key Finding**: Without ever seeing genre labels during training, the unsupervised model clustered Rap songs (Cluster 5) with >60% precision based entirely on audio properties like speechiness and rhythm!")

    render_module_footer("🎯 K-Means Segmentation")


# =============================================================================
# PAGE 7: 🎧 SONG RECOMMENDER
# =============================================================================
elif st.session_state['current_page'] == "🎧 Song Recommender":
    render_module_header()
    
    st.title("🎧 Content-Based Song Recommendation Engine")
    st.markdown("""
    Search for any song in the dataset and discover **acoustically similar tracks** using **Cosine Similarity**
    calculated over normalized Spotify audio features.
    """)
    
    col_s1, col_s2 = st.columns([3, 1])
    with col_s1:
        query = st.text_input("Search Song Title or Artist:", value="Ed Sheeran", placeholder="e.g. Blinding Lights, Eminem, Queen, Drake")
    with col_s2:
        top_n = st.slider("Recommendations Count:", min_value=3, max_value=12, value=5)
        
    same_cluster_filter = st.checkbox("Restrict recommendations strictly to the same acoustic cluster", value=False)
    
    matches = search_songs(query, df_unique, limit=25)
    
    if matches.empty:
        st.warning(f"No songs found matching '{query}'. Try searching for popular artists like 'Eminem', 'Queen', 'Coldplay', 'Drake', or 'Maroon 5'.")
    else:
        # Selection dropdown from matches
        song_options = [
            f"{row['track_name']} — {row['track_artist']} ({row['playlist_genre'].upper()})"
            for _, row in matches.iterrows()
        ]
        selected_option = st.selectbox("Select Target Song from Search Matches:", song_options)
        selected_idx = song_options.index(selected_option)
        chosen_row = matches.iloc[selected_idx]
        
        # Display Target Song Details Card
        st.markdown("---")
        st.subheader("🎯 Selected Target Song")
        
        col_t1, col_t2 = st.columns([2, 3])
        with col_t1:
            st.markdown(f"""
            <div class="spotify-card">
                <h3 style="margin-bottom: 5px;">🎵 {chosen_row['track_name']}</h3>
                <h4 style="color: #1DB954; margin-top: 0;">👤 {chosen_row['track_artist']}</h4>
                <p style="color: #E5E7EB; margin: 6px 0;"><b>Album</b>: {chosen_row.get('track_album_name', 'N/A')}</p>
                <p style="color: #E5E7EB; margin: 6px 0;"><b>Curated Genre</b>: <span class="genre-pill">{chosen_row['playlist_genre'].upper()}</span></p>
                <p style="color: #E5E7EB; margin: 6px 0;"><b>Cluster</b>: <span style="color:#1DB954; font-weight:bold;">{chosen_row['cluster_name']}</span></p>
                <p style="color: #E5E7EB; margin: 6px 0;"><b>Popularity Score</b>: ⭐ {chosen_row['track_popularity']}/100</p>
                <p style="color: #E5E7EB; margin: 6px 0;"><b>Tempo</b>: ⏱️ {chosen_row['tempo']:.1f} BPM</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col_t2:
            radar_cols = ['danceability', 'energy', 'speechiness', 'acousticness', 'valence']
            t_vals = [chosen_row[f] for f in radar_cols]
            t_vals += [t_vals[0]]
            
            fig_target_radar = go.Figure()
            fig_target_radar.add_trace(go.Scatterpolar(
                r=t_vals,
                theta=[f.capitalize() for f in radar_cols] + [radar_cols[0].capitalize()],
                fill='toself',
                name=chosen_row['track_name'][:25],
                line_color='#1DB954'
            ))
            fig_target_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                paper_bgcolor='#181818',
                font_color='#FFFFFF',
                title="<b>Target Song Audio Signature</b>"
            )
            st.plotly_chart(fig_target_radar, use_container_width=True)

        # Generate Recommendations
        target_song, recommendations = recommend_similar_songs(
            chosen_row['track_id'],
            df_unique,
            X_scaled,
            top_n=top_n,
            same_cluster_only=same_cluster_filter
        )
        
        st.markdown("---")
        st.subheader(f"🎶 Top {top_n} Similar Songs Recommended by AI")
        
        for i, (_, rec) in enumerate(recommendations.iterrows(), 1):
            with st.container():
                st.markdown(f"""
                <div class="spotify-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h4 style="margin: 0; color: #FFFFFF;">#{i}. {rec['track_name']}</h4>
                            <p style="margin: 4px 0 8px 0; color: #1DB954; font-weight:700;">👤 {rec['track_artist']}</p>
                        </div>
                        <div style="text-align: right;">
                            <span style="font-size: 1.6rem; font-weight: 800; color: #1DB954;">{rec['similarity_pct']}%</span>
                            <br><span style="font-size: 0.85rem; color: #D1D5DB; font-weight: 500;">Acoustic Match</span>
                        </div>
                    </div>
                    <div style="margin-top: 12px;">
                        <span class="genre-pill">{rec['playlist_genre'].upper()}</span>
                        <span class="genre-pill">{rec['cluster_name'].split('(')[0]}</span>
                        <span style="color: #E5E7EB; font-size: 0.9rem; margin-left: 10px; font-weight: 500;">💡 <b>Why:</b> {rec['match_reason']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    render_module_footer("🎧 Song Recommender")


# =============================================================================
# PAGE 8: 🧠 PROJECT METHODOLOGY & ARCHITECTURE
# =============================================================================
elif st.session_state['current_page'] == "🧠 Project Methodology & Architecture":
    render_module_header()
    
    st.title("🧠 Project Methodology & Machine Learning Architecture")
    st.markdown("""
    This section documents the end-to-end data science and unsupervised machine learning workflow
    implemented in this project.
    """)
    
    st.markdown("""
    <div class="spotify-card">
        <h3>🔬 Machine Learning Pipeline Architecture</h3>
        <ol>
            <li><b>Data Preprocessing & Deduplication</b>:
                <br>The raw 32,833 records contain duplicate entries because popular tracks appear across multiple Spotify playlists. We resolved 5 missing metadata rows and consolidated songs into 28,352 unique tracks.
            </li>
            <br>
            <li><b>Feature Standardization (Z-score Scaling)</b>:
                <br>Because features like <code>loudness</code> (-46 to 1.3 dB) and <code>tempo</code> (60 to 240 BPM) have far larger scales than unit-bounded attributes like <code>danceability</code> (0 to 1), applying <code>StandardScaler</code> guarantees unbiased Euclidean distance calculations:
                <br><br>
                $$z = \\frac{x - \\mu}{\\sigma}$$
            </li>
            <br>
            <li><b>Optimal Cluster Discovery ($k=6$)</b>:
                <br>Using the <b>Elbow Method (Within-Cluster Sum of Squares - WCSS)</b> and <b>Silhouette Coefficients</b>, $k=6$ was selected as the optimal hyperparameter. It achieves strong mathematical separation and mirrors Spotify's 6 curated playlist genres.
                <br><br>
                $$WCSS = \\sum_{i=1}^k \\sum_{x \\in C_i} ||x - \\mu_i||^2$$
            </li>
            <br>
            <li><b>Dimensionality Reduction (PCA)</b>:
                <br>Principal Component Analysis projects the 11-dimensional audio space into 2D and 3D visual planes while preserving the maximum variance of the acoustic features.
            </li>
            <br>
            <li><b>Content-Based Recommendation Engine</b>:
                <br>Calculates Cosine Similarity between song feature vectors to identify the nearest musical neighbors in high-dimensional space:
                <br><br>
                $$\\text{Cosine Similarity}(\\vec{u}, \\vec{v}) = \\frac{\\vec{u} \\cdot \\vec{v}}{\\|\\vec{u}\\| \\|\\vec{v}\\|}$$
            </li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🔍 Model Personas Mapping Summary")
    st.markdown("""
    | Cluster | Persona Name | Key Audio Characteristics |
    | :---: | :--- | :--- |
    | **0** | **Electronic & Deep Bass (EDM Pulse)** | Low acousticness (0.08), high electronic energy, club tempos |
    | **1** | **Live Stage & High Vibrance** | High liveness (0.62), stadium atmosphere and concert feel |
    | **2** | **Hard Rock & Driving Anthems** | High loudness (-5.3 dB), intense energy (0.79), rock tempo |
    | **3** | **Chill Acoustic & Mellow Soul** | Low loudness (-10.6 dB), gentle energy (0.42), warm tone |
    | **4** | **Upbeat Latin & Dance Pop** | High danceability (0.74), cheerful positive valence (0.68) |
    | **5** | **Rhythmic Rap & Urban Hip-Hop** | High speechiness (>0.20), punchy drums, rhythmic flow |
    """)

    render_module_footer("🧠 Project Methodology & Architecture")


# =============================================================================
# PAGE 9: 📚 AUDIO FEATURE DICTIONARY
# =============================================================================
elif st.session_state['current_page'] == "📚 Audio Feature Dictionary":
    render_module_header()
    
    st.title("📚 Spotify Audio Feature Dictionary")
    st.markdown("""
    This dictionary provides definitions, typical numerical ranges, and musical significance
    for all 11 core auditory features extracted by the Spotify Web API.
    """)
    
    feature_dict = [
        {"Feature": "danceability", "Range": "0.0 - 1.0", "Type": "Float", "Description": "How suitable a track is for dancing based on tempo, rhythm stability, beat strength, and overall regularity."},
        {"Feature": "energy", "Range": "0.0 - 1.0", "Type": "Float", "Description": "A perceptual measure of dynamic intensity and activity. Energetic tracks feel fast, loud, and noisy."},
        {"Feature": "loudness", "Range": "-60 to 0 dB", "Type": "Float", "Description": "Overall loudness of the track in decibels (dB), averaged across the entire recording."},
        {"Feature": "speechiness", "Range": "0.0 - 1.0", "Type": "Float", "Description": "Detects the presence of spoken words. Rap and hip-hop typically score above 0.20, while pure speech/podcasts score > 0.66."},
        {"Feature": "acousticness", "Range": "0.0 - 1.0", "Type": "Float", "Description": "A confidence measure of whether the track is acoustic (unplugged instruments like acoustic guitar, piano)."},
        {"Feature": "instrumentalness", "Range": "0.0 - 1.0", "Type": "Float", "Description": "Predicts whether a track contains no vocals. Rap and vocal pop score close to 0.0; ambient/electronic tracks score close to 1.0."},
        {"Feature": "liveness", "Range": "0.0 - 1.0", "Type": "Float", "Description": "Detects the presence of an audience in the recording. A score above 0.8 provides strong likelihood that the track is live."},
        {"Feature": "valence", "Range": "0.0 - 1.0", "Type": "Float", "Description": "A measure describing the musical positiveness. High valence tracks sound happy and euphoric; low valence tracks sound sad or angry."},
        {"Feature": "tempo", "Range": "0 - 250 BPM", "Type": "Float", "Description": "The overall estimated tempo of a track in beats per minute (BPM), reflecting the speed of the underlying beat."},
        {"Feature": "mode", "Range": "0 or 1", "Type": "Integer", "Description": "Modality of the track: 1 indicates Major key (typically brighter), 0 indicates Minor key (typically darker/melancholic)."},
        {"Feature": "key", "Range": "0 - 11", "Type": "Integer", "Description": "The key the track is in using standard Pitch Class notation (0 = C, 1 = C♯/D♭, 2 = D, etc.)."}
    ]
    
    st.table(pd.DataFrame(feature_dict))
    
    st.markdown("""
    <div class="spotify-card">
        <h4>💡 Practical Machine Learning Note</h4>
        <p>
        Notice that features such as <code>loudness</code> (dB) and <code>tempo</code> (BPM) have numerical values spanning from -46 to +240,
        whereas features like <code>danceability</code> and <code>valence</code> are strictly bounded between 0.0 and 1.0.
        Without <code>StandardScaler</code> normalization, distance-based algorithms like K-Means would be heavily biased toward tempo and loudness alone!
        </p>
    </div>
    """, unsafe_allow_html=True)

    render_module_footer("📚 Audio Feature Dictionary")
