import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ---------------------- CUSTOM STYLE ----------------------
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #1e1e2f, #2d2d44);
            color: #ffffff;
        }
        .card {
            padding: 20px;
            background-color: #ffffff10;
            border-radius: 15px;
            border: 1px solid #ffffff20;
            margin-bottom: 20px;
        }
        .section-title {
            font-size: 26px;
            font-weight: 700;
            color: #f5f5f5;
            margin-bottom: -10px;
        }
        h1, h2, h3, h4 {
            color: #ffffff !important;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(layout="wide", page_title="Analisis Klaster Roblox")
st.title("🎮 **Dashboard Analisis Klaster Game Roblox**")
st.markdown("### Visualisasi interaktif menggunakan K-Means, PCA, Line Chart, dan Ranking Otomatis ✨")

# ---------------------- SIDEBAR ----------------------
with st.sidebar:
    st.header("⚙️ Pengaturan Analisis")
    k_value = st.slider("Jumlah Klaster (K-Means)", 2, 8, 4)
    show_raw = st.checkbox("Tampilkan Data Mentah", False)

    rank_metric = st.selectbox(
        "Urutkan Ranking Berdasarkan:",
        ["Visits", "Active", "Likes", "Dislikes", "Rating"]
    )

    st.caption("Dashboard ML Premium ✨")

# ---------------------- DATA CLEANING ----------------------
@st.cache_data
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    cols_to_clean = ['Rank', 'Active', 'Visits', 'Favourites', 'Likes', 'Dislikes']

    for col in cols_to_clean:
        df[col] = df[col].astype(str).replace(r"[#,]", "", regex=True)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ['Active', 'Visits', 'Favourites', 'Likes', 'Dislikes', 'Rating']:
        df[col] = df[col].fillna(df[col].mean())

    return df

# ---------------------- CLUSTERING + PCA ----------------------
@st.cache_data
def perform_clustering(df, k):
    features = ['Active', 'Visits', 'Favourites', 'Likes', 'Dislikes', 'Rating']
    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_scaled)

    df["PC1"], df["PC2"] = pcs[:, 0], pcs[:, 1]

    return df, pca.explained_variance_ratio_

# ---------------------- MAIN APP ----------------------
df = load_and_clean_data("roblox_games.csv")
df_clustered, ev = perform_clustering(df.copy(), k_value)

# ---------------------- RAW DATA ----------------------
if show_raw:
    st.markdown("### 📄 Data Mentah")
    st.dataframe(df.head())

# ---------------------- SECTION 1 — KONTEN UTAMA ----------------------
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

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- SECTION 2 — PCA VISUAL ----------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<p class="section-title">🎨 Visualisasi PCA (2D)</p>', unsafe_allow_html=True)

fig, ax = plt.subplots(figsize=(10, 8))
sns.scatterplot(
    data=df_clustered, x="PC1", y="PC2",
    hue="Cluster", palette="viridis", s=120, ax=ax
)
ax.set_facecolor("#2a2a40")
fig.patch.set_facecolor("#2a2a40")
ax.grid(alpha=0.3)
st.pyplot(fig)

st.caption(f"PC1: {ev[0]*100:.2f}% • PC2: {ev[1]*100:.2f}%")
st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------
# 🌟 SECTION 3 — LINE CHART (FITUR BARU)
# ------------------------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<p class="section-title">📈 Grafik Line — Trend Antar Cluster</p>', unsafe_allow_html=True)

selected_metric = st.selectbox("Pilih metrik untuk Line Chart:", ["Active", "Visits", "Likes", "Rating"])

line_df = df_clustered.groupby("Cluster")[selected_metric].mean().reset_index()

fig2, ax2 = plt.subplots(figsize=(10, 5))
sns.lineplot(data=line_df, x="Cluster", y=selected_metric, marker="o", linewidth=3, ax=ax2)
ax2.set_facecolor("#2a2a40")
fig2.patch.set_facecolor("#2a2a40")
ax2.grid(alpha=0.3)
ax2.set_title(f"Trend '{selected_metric}' Antar Cluster", color="white")

st.pyplot(fig2)
st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------
# 🌟 SECTION 4 — RANKING OTOMATIS (FITUR BARU)
# ------------------------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<p class="section-title">🏆 Ranking Otomatis Game Roblox</p>', unsafe_allow_html=True)

ranking_df = df_clustered.sort_values(by=rank_metric, ascending=False)[['Name', rank_metric, 'Cluster']].head(10)
ranking_df.index = ranking_df.index + 1  # Rank mulai dari 1

st.markdown(f"### Ranking Teratas berdasarkan **{rank_metric}**")
st.dataframe(ranking_df.style.highlight_max(axis=0, color="yellow"))

st.markdown("</div>", unsafe_allow_html=True)
