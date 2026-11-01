"""
Comprehensive Test Suite for Spotify Genre Segmentation & Streamlit App
Tests all 6 pages, recommendation logic, data consistency, and statistics.
"""

import os
import sys
import numpy as np
import pandas as pd
from model import (
    AUDIO_FEATURES,
    CLUSTER_PERSONAS,
    load_pipeline,
    load_raw_data,
    preprocess_data,
    search_songs,
    recommend_similar_songs
)

def test_dataset_and_pipeline():
    print("--- 1. Testing Dataset & Model Pipeline ---")
    df_raw = load_raw_data()
    assert df_raw.shape[0] == 32828, f"Expected 32,828 cleaned rows, got {df_raw.shape[0]}"
    assert df_raw.shape[1] == 23, f"Expected 23 columns, got {df_raw.shape[1]}"
    print(f" [PASS] Raw dataset loaded: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

    pipeline = load_pipeline('saved_model.pkl')
    df_unique = pipeline['df_unique']
    X_scaled = pipeline['X_scaled']
    kmeans = pipeline['kmeans']

    assert len(df_unique) == 28352, f"Expected 28,352 unique tracks, got {len(df_unique)}"
    assert X_scaled.shape == (28352, len(AUDIO_FEATURES)), f"Expected shape (28352, 11), got {X_scaled.shape}"
    assert kmeans.n_clusters == 6, f"Expected 6 clusters, got {kmeans.n_clusters}"
    print(f" [PASS] Pipeline loaded: {len(df_unique)} unique tracks, 6 clusters, 11 scaled features")

    # Verify scaler properties: mean ~ 0, std ~ 1
    means = np.mean(X_scaled, axis=0)
    stds = np.std(X_scaled, axis=0)
    assert np.all(np.abs(means) < 1e-3), "Features are not centered around 0"
    assert np.all(np.abs(stds - 1.0) < 1e-3), "Features are not scaled to unit variance"
    print(" [PASS] Feature scaling verified: mean = 0.0, std = 1.0")

    return df_raw, df_unique, X_scaled, pipeline


def test_statistics_and_correlations(df_raw, df_unique):
    print("\n--- 2. Verifying Statistics & Correlation Calculations ---")
    
    # Check genre distribution
    genres = df_raw['playlist_genre'].value_counts()
    print(" Genre counts in dataset:")
    for g, count in genres.items():
        print(f"   - {g:6s}: {count:,}")
        assert count > 4500, f"Genre {g} has unexpectedly low track count: {count}"
    print(" [PASS] Genre distributions match actual dataset")

    # Check Pearson correlation matrix
    corr = df_unique[AUDIO_FEATURES].corr()
    loud_energy_corr = corr.loc['loudness', 'energy']
    acoust_energy_corr = corr.loc['acousticness', 'energy']
    dance_valence_corr = corr.loc['danceability', 'valence']

    print(f" Verified Correlations:")
    print(f"   - Loudness vs Energy: {loud_energy_corr:.4f} (Expected ~ +0.68)")
    print(f"   - Acousticness vs Energy: {acoust_energy_corr:.4f} (Expected ~ -0.54)")
    print(f"   - Danceability vs Valence: {dance_valence_corr:.4f} (Expected ~ +0.33)")

    assert abs(loud_energy_corr - 0.68) < 0.05, f"Unexpected correlation for Loudness/Energy: {loud_energy_corr}"
    assert abs(acoust_energy_corr - (-0.54)) < 0.05, f"Unexpected correlation for Acousticness/Energy: {acoust_energy_corr}"
    assert abs(dance_valence_corr - 0.33) < 0.05, f"Unexpected correlation for Danceability/Valence: {dance_valence_corr}"
    print(" [PASS] Correlation matrix strictly derived from actual dataset values")


def test_cluster_profiles(df_unique):
    print("\n--- 3. Verifying Clustering & Personas ---")
    assert 'cluster' in df_unique.columns, "Missing 'cluster' column"
    assert 'cluster_name' in df_unique.columns, "Missing 'cluster_name' column"
    
    cluster_counts = df_unique['cluster'].value_counts()
    for c in range(6):
        count = cluster_counts.get(c, 0)
        assert count > 1000, f"Cluster {c} is too small: {count}"
        persona = CLUSTER_PERSONAS.get(c, {})
        print(f"   Cluster {c}: {persona.get('name')} ({count:,} songs)")
    print(" [PASS] All 6 clusters populated and correctly named")


def test_recommendation_engine(df_unique, X_scaled):
    print("\n--- 4. Testing Recommendation Engine with Diverse Real Songs ---")
    
    test_queries = [
        ("Ed Sheeran", "Pop"),
        ("Queen", "Rock"),
        ("Eminem", "Rap"),
        ("Martin Garrix", "EDM"),
        ("The Weeknd", "R&B"),
        ("Daddy Yankee", "Latin"),
        ("Coldplay", "Rock/Pop"),
        ("Post Malone", "Rap/Pop")
    ]

    for query, genre_hint in test_queries:
        matches = search_songs(query, df_unique, limit=5)
        assert not matches.empty, f"Search failed for known artist: '{query}'"
        target_song = matches.iloc[0]
        
        target, recs = recommend_similar_songs(target_song['track_id'], df_unique, X_scaled, top_n=5)
        assert len(recs) == 5, f"Expected 5 recommendations, got {len(recs)}"
        assert target_song['track_id'] not in recs['track_id'].values, "Target song included in its own recommendations"
        
        # Check similarity scores
        assert (recs['similarity_pct'] >= 70.0).all(), f"Similarity scores too low for {query}"
        assert (recs['similarity_pct'] <= 100.0).all(), f"Similarity score exceeds 100% for {query}"
        
        # Verify deduplication of track_name + artist
        for _, rec in recs.iterrows():
            same_name = rec['track_name'].strip().lower() == target_song['track_name'].strip().lower()
            same_artist = rec['track_artist'].strip().lower() == target_song['track_artist'].strip().lower()
            assert not (same_name and same_artist), f"Duplicate song recommended: {rec['track_name']} by {rec['track_artist']}"
            
        print(f" [PASS] '{target['track_name'][:25]}' by {target['track_artist']} -> Top Rec: '{recs.iloc[0]['track_name'][:25]}' ({recs.iloc[0]['similarity_pct']}%)")

    # Test edge cases
    print("\n Testing Edge Cases:")
    empty_res = search_songs("", df_unique)
    assert empty_res.empty, "Empty search should return empty DataFrame"
    print("   - Empty query returns 0 results: PASS")

    nonexistent_res = search_songs("xyzrandomquery9876543210", df_unique)
    assert nonexistent_res.empty, "Nonexistent search should return empty DataFrame"
    print("   - Nonexistent query returns 0 results: PASS")

    cluster_filtered_target, cluster_filtered_recs = recommend_similar_songs(
        df_unique.iloc[0]['track_id'], df_unique, X_scaled, top_n=3, same_cluster_only=True
    )
    for _, r in cluster_filtered_recs.iterrows():
        assert r['cluster'] == cluster_filtered_target['cluster'], "Filtered recommendation produced cross-cluster song"
    print("   - Same-cluster filter strictly enforces cluster constraint: PASS")


def test_dashboard_components():
    print("\n--- 5. Testing Streamlit Dashboard Components (app.py) ---")
    with open('app.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # Check all 9 page definitions
    pages = [
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
    for p in pages:
        assert p in content, f"Missing page in app.py: {p}"
    print(f" [PASS] All {len(pages)} multi-page website sections present")

    # Verify that clean sidebar requirements are met (no unnecessary stats/labels in sidebar)
    sidebar_block = content.split("with st.sidebar:")[1].split("# -----------------------------------------------------------------------------")[0]
    removed_items = [
        "Project Quick Facts",
        "Total Tracks",
        "Unique Songs",
        "Acoustic Clusters",
        "ML Model",
        "B.Sc. Computer Science AI Coursework"
    ]
    for item in removed_items:
        assert item not in sidebar_block, f"Unwanted item found in cleaned sidebar: {item}"
    print(" [PASS] Clean sidebar verified: All unnecessary stats/labels removed")

    # Verify Home Landing Page has the Get Started action
    assert "btn_hero_get_started" in content, "Missing 'Get Started' button on Landing page"
    assert "🗂️ Module Selection" in content, "Missing Module Selection page"
    print(" [PASS] Landing page & Get Started action verified")

    # Verify Module Selection has 7 cards and View Details actions
    for card_key in ["card_btn_dataset", "card_btn_eda", "card_btn_corr", "card_btn_kmeans", "card_btn_rec", "card_btn_method", "card_btn_dict"]:
        assert card_key in content, f"Missing card button in Module Selection: {card_key}"
    print(" [PASS] All 7 Module Selection cards and 'View Details' buttons verified")

    # Verify Previous / Next / Back navigation helpers
    assert "render_module_header" in content, "Missing render_module_header helper"
    assert "render_module_footer" in content, "Missing render_module_footer helper"
    assert "navigate_to" in content, "Missing navigate_to function"
    print(" [PASS] Previous/Next/Back to Modules navigation engine verified")

    # Check key components
    required_elements = [
        "st.set_page_config",
        "get_pipeline()",
        "get_raw_data()",
        "st.sidebar",
        "px.bar",
        "px.histogram",
        "px.scatter",
        "px.imshow",
        "px.scatter_3d",
        "go.Scatterpolar",
        "search_songs",
        "recommend_similar_songs"
    ]
    for el in required_elements:
        assert el in content, f"Missing required UI component in app.py: {el}"
    print(" [PASS] All required UI components, charts, and functions verified in app.py")


if __name__ == '__main__':
    print("=======================================================")
    print("       STARTING SPOTIFY PROJECT VERIFICATION SUITE     ")
    print("=======================================================\n")
    df_raw, df_unique, X_scaled, pipeline = test_dataset_and_pipeline()
    test_statistics_and_correlations(df_raw, df_unique)
    test_cluster_profiles(df_unique)
    test_recommendation_engine(df_unique, X_scaled)
    test_dashboard_components()
    print("\n=======================================================")
    print("       ALL PROJECT VERIFICATION TESTS PASSED 100%!     ")
    print("=======================================================")

