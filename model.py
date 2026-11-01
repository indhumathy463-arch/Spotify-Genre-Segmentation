"""
Spotify Songs' Genre Segmentation - Model & Recommendation Engine
------------------------------------------------------------------
This module handles data loading, preprocessing, K-Means clustering,
PCA dimensionality reduction, and content-based music recommendation.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score

# Canonical audio features used for music segmentation
AUDIO_FEATURES = [
    'danceability',
    'energy',
    'key',
    'loudness',
    'mode',
    'speechiness',
    'acousticness',
    'instrumentalness',
    'liveness',
    'valence',
    'tempo'
]

# Descriptive cluster persona names mapped to musical qualities
CLUSTER_PERSONAS = {
    0: {
        "name": "Electronic & Deep Bass (EDM Pulse)",
        "desc": "High electronic energy, heavy synthesizer layers, and club-ready tempos.",
        "icon": "⚡"
    },
    1: {
        "name": "Live Stage & High Vibrance",
        "desc": "High liveness and raw performance feel, capturing stadium and live session atmosphere.",
        "icon": "🎤"
    },
    2: {
        "name": "Hard Rock & Driving Anthems",
        "desc": "Intense loudness, rock guitar riffs, high energy, and driving rhythm.",
        "icon": "🎸"
    },
    3: {
        "name": "Chill Acoustic & Mellow Soul",
        "desc": "Warm acoustic instrumentation, gentle dynamics, and laid-back soothing vibes.",
        "icon": "☕"
    },
    4: {
        "name": "Upbeat Latin & Dance Pop",
        "desc": "High danceability, infectious positivity (valence), and rhythmic groove.",
        "icon": "💃"
    },
    5: {
        "name": "Rhythmic Rap & Urban Hip-Hop",
        "desc": "High lyrical speechiness, distinct percussion beats, and confident flow.",
        "icon": "🎧"
    }
}


def find_dataset_path():
    """Locate the dataset in common project directories."""
    candidates = [
        'spotify dataset.csv',
        os.path.join('dataset', 'spotify dataset.csv'),
        os.path.join('..', 'spotify dataset.csv'),
        os.path.join('..', 'dataset', 'spotify dataset.csv')
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("Could not locate 'spotify dataset.csv' in workspace.")


def load_raw_data(filepath=None):
    """
    Load raw Spotify dataset and perform initial clean-up.
    - Removes 5 missing rows where track_name/artist are null.
    """
    if filepath is None:
        filepath = find_dataset_path()
    
    df = pd.read_csv(filepath)
    # Remove records missing crucial textual metadata (only 5 out of 32,833)
    df_clean = df.dropna(subset=['track_name', 'track_artist']).copy()
    return df_clean


def preprocess_data(df, deduplicate=True):
    """
    Preprocess data for machine learning:
    - Deduplicates on track_id for song-level modeling.
    - Aggregates playlist occurrences so each song remembers its genres.
    - Scales numerical audio features with StandardScaler.
    """
    if deduplicate:
        # Group playlist genres for each unique track
        genre_agg = df.groupby('track_id')['playlist_genre'].apply(
            lambda x: ', '.join(sorted(set(x)))
        ).reset_index().rename(columns={'playlist_genre': 'all_genres'})

        # Keep primary metadata row per unique track_id
        df_unique = df.drop_duplicates(subset=['track_id']).copy()
        df_unique = df_unique.merge(genre_agg, on='track_id', how='left')
    else:
        df_unique = df.copy()
        df_unique['all_genres'] = df_unique['playlist_genre']

    # Audio feature matrix
    X_audio = df_unique[AUDIO_FEATURES].values

    # Standard Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_audio)

    return df_unique, X_scaled, scaler


def train_clustering_pipeline(df_unique, X_scaled, n_clusters=6, random_state=42):
    """
    Trains K-Means clustering and PCA for 2D & 3D visualizations.
    """
    # 1. Train K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    df_unique['cluster'] = cluster_labels
    df_unique['cluster_name'] = [
        CLUSTER_PERSONAS.get(c, {}).get('name', f"Cluster {c}") for c in cluster_labels
    ]

    # 2. Fit PCA for 2D and 3D visualization
    pca_2d = PCA(n_components=2, random_state=random_state)
    pca_3d = PCA(n_components=3, random_state=random_state)

    X_pca2 = pca_2d.fit_transform(X_scaled)
    X_pca3 = pca_3d.fit_transform(X_scaled)

    df_unique['pca_x'] = X_pca2[:, 0]
    df_unique['pca_y'] = X_pca2[:, 1]
    df_unique['pca_z'] = X_pca3[:, 2]

    pipeline = {
        'kmeans': kmeans,
        'scaler': StandardScaler().fit(df_unique[AUDIO_FEATURES].values),
        'pca_2d': pca_2d,
        'pca_3d': pca_3d,
        'features': AUDIO_FEATURES,
        'cluster_personas': CLUSTER_PERSONAS,
        'n_clusters': n_clusters,
        'df_unique': df_unique,
        'X_scaled': X_scaled
    }

    return pipeline


def save_pipeline(pipeline, filepath='saved_model.pkl'):
    """Save pipeline dictionary to pickle file."""
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"Pipeline saved to: {filepath}")


def load_pipeline(filepath='saved_model.pkl'):
    """Load pipeline dictionary from pickle file."""
    if not os.path.exists(filepath):
        # Fallback search
        alt_paths = [
            filepath,
            os.path.join('models', 'saved_model.pkl'),
            os.path.join('..', filepath),
            os.path.join('..', 'models', 'saved_model.pkl')
        ]
        found = False
        for alt in alt_paths:
            if os.path.exists(alt):
                filepath = alt
                found = True
                break
        if not found:
            raise FileNotFoundError(f"Cannot find pipeline file: {filepath}")

    with open(filepath, 'rb') as f:
        pipeline = pickle.load(f)
    return pipeline


def search_songs(query, df_unique, limit=20):
    """
    Search for songs in the dataset by track name or artist.
    Returns matched dataframe rows.
    """
    if not query or not query.strip():
        return df_unique.head(0)

    q = query.strip().lower()
    mask = (
        df_unique['track_name'].str.lower().str.contains(q, na=False) |
        df_unique['track_artist'].str.lower().str.contains(q, na=False)
    )
    return df_unique[mask].head(limit)


def recommend_similar_songs(track_idx_or_id, df_unique, X_scaled, top_n=5, same_cluster_only=False):
    """
    Recommend songs using Cosine Similarity on scaled audio features.
    
    Parameters:
    - track_idx_or_id: Integer row index in df_unique or string track_id.
    - df_unique: Preprocessed DataFrame of unique tracks.
    - X_scaled: Scaled audio features matrix corresponding to df_unique.
    - top_n: Number of recommendations to return.
    - same_cluster_only: If True, restricts recommendations to the same K-Means cluster.
    
    Returns:
    - (target_song, recommendations_df)
    """
    # Locate target track row
    if isinstance(track_idx_or_id, str):
        match = df_unique[df_unique['track_id'] == track_idx_or_id]
        if match.empty:
            raise ValueError(f"Track ID '{track_idx_or_id}' not found.")
        target_idx = match.index[0]
        # Map to integer row position in df_unique
        row_pos = df_unique.index.get_loc(target_idx)
    else:
        row_pos = int(track_idx_or_id)

    target_song = df_unique.iloc[row_pos]
    target_cluster = target_song['cluster']
    target_vector = X_scaled[row_pos].reshape(1, -1)

    # Compute cosine similarities across tracks
    similarities = cosine_similarity(target_vector, X_scaled).flatten()

    # Filter candidate indices
    if same_cluster_only:
        candidate_indices = np.where(df_unique['cluster'].values == target_cluster)[0]
    else:
        candidate_indices = np.arange(len(df_unique))

    # Exclude target song itself and songs with identical name & artist
    target_name_lower = str(target_song['track_name']).strip().lower()
    target_artist_lower = str(target_song['track_artist']).strip().lower()

    # Filter candidates
    valid_candidates = []
    for idx in candidate_indices:
        if idx == row_pos:
            continue
        c_name = str(df_unique.iloc[idx]['track_name']).strip().lower()
        c_artist = str(df_unique.iloc[idx]['track_artist']).strip().lower()
        if c_name == target_name_lower and c_artist == target_artist_lower:
            continue
        valid_candidates.append(idx)
    candidate_indices = np.array(valid_candidates)

    # Rank by similarity score descending
    sorted_sub_indices = candidate_indices[np.argsort(similarities[candidate_indices])[::-1]]
    top_indices = sorted_sub_indices[:top_n]

    # Build recommendations DataFrame
    rec_df = df_unique.iloc[top_indices].copy()
    rec_df['similarity_score'] = similarities[top_indices]
    rec_df['similarity_pct'] = (rec_df['similarity_score'] * 100).round(1)

    # Calculate audio similarity explanation
    explanations = []
    for idx in top_indices:
        cand_song = df_unique.iloc[idx]
        diffs = {}
        for feat in ['danceability', 'energy', 'loudness', 'tempo', 'valence', 'acousticness']:
            diff = abs(target_song[feat] - cand_song[feat])
            diffs[feat] = diff
        # Find the 2 closest features
        closest_feats = sorted(diffs.items(), key=lambda x: x[1])[:2]
        explanation = f"Matching {closest_feats[0][0]} and {closest_feats[1][0]}"
        explanations.append(explanation)

    rec_df['match_reason'] = explanations

    return target_song, rec_df


if __name__ == '__main__':
    print("Loading raw dataset...")
    df_raw = load_raw_data()
    print(f"Loaded {len(df_raw)} rows.")

    print("Preprocessing & Scaling audio features...")
    df_unique, X_scaled, scaler = preprocess_data(df_raw, deduplicate=True)
    print(f"Prepared {len(df_unique)} unique tracks.")

    print("Training K-Means (k=6) and PCA pipeline...")
    pipeline = train_clustering_pipeline(df_unique, X_scaled, n_clusters=6)

    # Save to root and models directory
    save_pipeline(pipeline, 'saved_model.pkl')
    save_pipeline(pipeline, os.path.join('models', 'saved_model.pkl'))

    # Quick test recommendation
    sample_song = df_unique.iloc[0]
    print(f"\nTesting recommendation for: '{sample_song['track_name']}' by {sample_song['track_artist']}")
    target, recs = recommend_similar_songs(0, df_unique, X_scaled, top_n=3)
    for _, row in recs.iterrows():
        print(f" - {row['track_name']} by {row['track_artist']} (Match: {row['similarity_pct']}%, Cluster: {row['cluster_name']})")
    print("\nModel pipeline trained and verified successfully!")

