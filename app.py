import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- 1. Konfigurasi Halaman ---
st.set_page_config(layout="wide", page_title="Analisis Klaster Game Roblox")
st.title("Aplikasi Streamlit untuk Analisis Klaster Game Roblox 🎮")
st.markdown("Aplikasi ini melakukan Clustering K-Means pada data game Roblox berdasarkan metrik popularitas, kemudian memvisualisasikan hasilnya menggunakan PCA.")
st.markdown("---")

# --- 2. Fungsi Pembersihan Data (Diadaptasi dari Notebook) ---
@st.cache_data
def load_and_clean_data(file_path):
    # Memuat data
    df = pd.read_csv(file_path)
    
    # Kolom-kolom yang perlu dibersihkan (menghapus koma dan '#')
    cols_to_clean = ['Rank', 'Active', 'Visits', 'Favourites', 'Likes', 'Dislikes']

    def clean_and_convert(dataframe, columns):
        for col in columns:
            # Menghapus '#' dan koma
            dataframe[col] = dataframe[col].astype(str).str.replace(r'[#\\,]', '', regex=True)
            # Mengonversi ke numerik
            dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
        return dataframe

    # Menerapkan pembersihan
    df = clean_and_convert(df.copy(), cols_to_clean)
    
    # Mengisi NaN pada kolom fitur dengan rata-rata (jika ada, seperti dalam kasus data Anda)
    features = ['Active', 'Visits', 'Favourites', 'Likes', 'Dislikes', 'Rating']
    for col in features:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())
    
    return df

# --- 3. Fungsi Analisis Klaster dan PCA (Diadaptasi dari Notebook) ---
@st.cache_data
def perform_clustering_and_pca(df_clean):
    # Kolom fitur untuk clustering
    features = ['Active', 'Visits', 'Favourites', 'Likes', 'Dislikes', 'Rating']
    X = df_clean[features].copy()

    # Standardisasi
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means Clustering (k=4 seperti di notebook)
    k = 4
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df_clean['Cluster'] = kmeans.fit_predict(X_scaled)

    # PCA (Reduksi Dimensi)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    df_clean['PC1'] = principal_components[:, 0]
    df_clean['PC2'] = principal_components[:, 1]
    
    explained_variance_ratio_pc1 = pca.explained_variance_ratio_[0]
    explained_variance_ratio_pc2 = pca.explained_variance_ratio_[1]

    return df_clean, explained_variance_ratio_pc1, explained_variance_ratio_pc2

# --- 4. Main App Logic ---
try:
    df = load_and_clean_data('roblox_games.csv')
    
    # Mengambil hasil analisis
    df_clustered, evr1, evr2 = perform_clustering_and_pca(df.copy())
    
    # --- 5. Tampilkan Data dan Statistik ---
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Data Awal & Hasil Klaster")
        st.info("Tampilkan 5 baris pertama data setelah pembersihan dan penambahan kolom `Cluster`.")
        st.dataframe(df_clustered[['Rank', 'Name', 'Active', 'Visits', 'Rating', 'Cluster']].head())
    
    with col2:
        st.subheader(f"2. Statistik Rata-rata per Cluster (k=4)")
        cluster_stats = df_clustered.groupby('Cluster')[['Active', 'Visits', 'Favourites', 'Likes', 'Dislikes', 'Rating']].mean().sort_values(by='Visits', ascending=False)
        st.dataframe(cluster_stats.style.format("{:,.0f}"))
        st.caption("Statistik ini membantu mengidentifikasi karakteristik setiap kelompok (klaster).")
    
    st.markdown("---")

    # --- 6. Tampilkan Visualisasi PCA ---
    st.subheader("3. Visualisasi Clustering K-Means (PCA 2 Komponen)")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(
        x='PC1',
        y='PC2',
        data=df_clustered,
        hue='Cluster',
        palette='viridis',
        style='Cluster',
        s=60,
        legend='full',
        ax=ax
    )

    ax.set_title('Visualisasi Clustering Game Roblox')
    ax.set_xlabel(f'Principal Component 1 ({evr1:.2%} Varian Dijelaskan)')
    ax.set_ylabel(f'Principal Component 2 ({evr2:.2%} Varian Dijelaskan)')
    ax.grid(True)
    ax.legend(title='Cluster')
    
    st.pyplot(fig)

    st.caption(f"PC1 menjelaskan {evr1:.2%} dan PC2 menjelaskan {evr2:.2%} dari total varian data.")

except FileNotFoundError:
    st.error("File 'roblox_games.csv' tidak ditemukan. Pastikan file tersebut ada di direktori yang sama dengan `app.py` saat deployment.")
except Exception as e:
    st.error(f"Terjadi kesalahan saat menjalankan analisis: {e}")