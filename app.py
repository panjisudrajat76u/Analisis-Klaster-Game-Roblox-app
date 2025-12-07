import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ---------------------- CUSTOM STREAMLIT STYLE ----------------------
st.markdown("""
    <style>
        /* Background gradient */
        .stApp {
            background: linear-gradient(135deg, #1e1e2f, #2d2d44);
            color: #ffffff;
        }
        /* Card style */
        .card {
            padding: 20px;
            background-color: #ffffff10;
            border-radius: 15px;
            border: 1px solid #ffffff20;
            margin-bottom: 20px;
        }
        /* Section Title */
        .section-title {
            font-size: 26px;
            font-weight: 700;
            color: #f5f5f5;
            margin-bottom: -10px;
        }
        /* Headers */
        h1, h2, h3, h4 {
            color: #ffffff !important;
        }
        /* Dataframe container */
        .stDataFrame {
            background-color: #ffffff10 !important;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(layout="wide", page_title="Analisis Klaster Roblox")

st.title("🎮 **Dashboard Analisis Klaster Game Roblox**")
st.markdown("### Visualisasi interaktif menggunakan K-Means & PCA")

# ---------------------- SIDEBAR ----------------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/1/1f/Roblox_logo_2022.svg")
    st.header("⚙️ Pengaturan Analisis")

    k_value = st.slider("Jumlah Klaster (K-Means)", 2, 8, 4)
    show_raw = st.checkbox("Tampilkan Data Mentah", False)

    st.markdown("---")
    st.caption("Dibuat oleh: **Dashboard ML Streamlit Premium** ✨")

# ---------------------- DATA CLEANING ----------------------
@st.cache_data
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    cols_to_clean = ['Rank', 'Active', 'Visits', 'Favourites', 'Likes', 'Dislikes']

    def clean_and_convert(df, cols):
        for col in cols:
            df[col] = df[col].astype(str).str.replace(r'[#\,]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    df = clean_and_convert(df.copy(), cols_to_clean)

    for col in ['Active', 'Visits', 'Favourites', 'Likes', 'Dislikes', 'Rating']:
        df[col] = df[col].fillna(df[col].mean())

    return df

# ---------------------- CLUSTERING & PCA ----------------------
@st.cache_data
def perform_clustering(df_clean, k):
    features = ['Active', 'Visits', 'Favourites', 'Likes', 'Dislikes', 'Rating']
    X = df_clean[features].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df_clean['Cluster'] = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_scaled)

    df_clean['PC1'] = pcs[:, 0]
    df_clean['PC2'] = pcs[:, 1]

    return df_clean, pca.explained_variance_ratio_


# ---------------------- MAIN APP ----------------------
try:
    df = load_and_clean_data('roblox_games.csv')
    df_clustered, ev = perform_clustering(df.copy(), k_value)

    # ---------------------- RAW DATA ----------------------
    if show_raw:
        st.markdown("### 📄 Data Mentah")
        st.dataframe(df.head())

    # ---------------------- SECTION 1 ----------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">📊 Data Game & Klaster</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Sample Data")
        st.dataframe(df_clustered[['Rank','Name','Active','Visits','Rating','Cluster']].head())

    with col2:
        st.markdown("#### Statistik Rata-rata Tiap Klaster")
        stats = df_clustered.groupby("Cluster")[['Active','Visits','Favourites','Likes','Dislikes','Rating']].mean()
        st.dataframe(stats.style.format("{:,.0f}"))

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------- SECTION 2 (PCA VISUALIZATION) ----------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">🎨 Visualisasi PCA</p>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(
        x='PC1', y='PC2',
        data=df_clustered,
        hue='Cluster', palette='viridis',
        s=120, alpha=0.9, ax=ax
    )

    ax.set_title("PCA Clustering", color='white')
    ax.set_facecolor("#2a2a40")
    fig.patch.set_facecolor("#2a2a40")
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)

    st.caption(f"PC1 menjelaskan {ev[0]*100:.2f}% varian • PC2 menjelaskan {ev[1]*100:.2f}% varian")

    st.markdown('</div>', unsafe_allow_html=True)

except FileNotFoundError:
    st.error("❌ File 'roblox_games.csv' tidak ditemukan.")
except Exception as e:
    st.error(f"Terjadi error: {e}")
