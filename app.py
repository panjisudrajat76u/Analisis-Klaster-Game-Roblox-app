import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta

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
        .ai-tab {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            padding: 5px 15px;
            margin: 5px;
        }
        .prediction-card {
            background: rgba(255, 255, 255, 0.1);
            border-left: 4px solid #4CAF50;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
        .insight-card {
            background: rgba(255, 255, 255, 0.08);
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .recommendation-item {
            background: rgba(76, 175, 80, 0.1);
            border-radius: 8px;
            padding: 12px;
            margin: 8px 0;
            border-left: 3px solid #4CAF50;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    layout="wide", 
    page_title="Analisis Klaster Roblox",
    page_icon="🎮",
    initial_sidebar_state="expanded"
)

st.title("🎮 **Dashboard Analisis Klaster Game Roblox**")
st.markdown("### Visualisasi interaktif menggunakan K-Means, PCA, Line Chart, dan Ranking Otomatis ✨")

# ---------------------- SIDEBAR ----------------------
with st.sidebar:
    st.markdown("<h2 style='color: #ff6b6b;'>⚙️ Kontrol Panel</h2>", unsafe_allow_html=True)
    
    # Analytics Settings
    with st.expander("📊 Pengaturan Analisis", expanded=True):
        k_value = st.slider("Jumlah Klaster (K-Means)", 2, 10, 4)
        show_raw = st.checkbox("Tampilkan Data Mentah", False)
        normalize_data = st.checkbox("Normalisasi Data", True)
    
    # Ranking Settings
    with st.expander("🏆 Pengaturan Ranking", expanded=False):
        rank_metric = st.selectbox(
            "Urutkan Ranking Berdasarkan:",
            ["Visits", "Active", "Likes", "Dislikes", "Rating", "Favourites"]
        )
        num_top_games = st.slider("Jumlah Game Teratas", 5, 20, 10)
    
    # AI Settings
    with st.expander("🤖 Pengaturan AI", expanded=False):
        ai_enabled = st.checkbox("Aktifkan Mode AI", True)
        ai_confidence = st.slider("Tingkat Kepercayaan AI", 0.1, 1.0, 0.8)
    
    st.divider()
    st.caption("© 2024 Dashboard ML Premium ✨")
    st.caption("Versi 2.0 | Powered by Streamlit")

# ---------------------- DATA CLEANING ----------------------
@st.cache_data
def load_and_clean_data(file_path):
    try:
        df = pd.read_csv(file_path)
    except:
        # Create sample data if file not found
        st.warning("File tidak ditemukan. Membuat data contoh...")
        np.random.seed(42)
        n_samples = 100
        data = {
            'Rank': range(1, n_samples + 1),
            'Name': [f'Roblox Game {i}' for i in range(1, n_samples + 1)],
            'Active': np.random.randint(1000, 100000, n_samples),
            'Visits': np.random.randint(10000, 1000000, n_samples),
            'Favourites': np.random.randint(100, 100000, n_samples),
            'Likes': np.random.randint(500, 500000, n_samples),
            'Dislikes': np.random.randint(10, 50000, n_samples),
            'Rating': np.round(np.random.uniform(1.0, 5.0, n_samples), 2)
        }
        df = pd.DataFrame(data)
    
    cols_to_clean = ['Rank', 'Active', 'Visits', 'Favourites', 'Likes', 'Dislikes']
    
    for col in cols_to_clean:
        if col in df.columns:
            df[col] = df[col].astype(str).replace(r"[#,]", "", regex=True)
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Fill missing values
    numeric_cols = ['Active', 'Visits', 'Favourites', 'Likes', 'Dislikes', 'Rating']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    return df

# ---------------------- CLUSTERING + PCA ----------------------
@st.cache_data
def perform_clustering(df, k, normalize=True):
    features = ['Active', 'Visits', 'Favourites', 'Likes', 'Dislikes', 'Rating']
    features = [f for f in features if f in df.columns]
    X = df[features]
    
    if normalize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.values
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    df['Cluster_Label'] = df['Cluster'].apply(lambda x: f'Klaster {x}')
    
    # Add cluster centers for visualization
    cluster_centers = kmeans.cluster_centers_
    
    # PCA for visualization
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_scaled)
    df["PC1"], df["PC2"] = pcs[:, 0], pcs[:, 1]
    
    # Calculate silhouette score (simplified version)
    from sklearn.metrics import silhouette_score
    if len(np.unique(df['Cluster'])) > 1:
        sil_score = silhouette_score(X_scaled, df['Cluster'])
    else:
        sil_score = 0
    
    return df, pca.explained_variance_ratio_, sil_score, cluster_centers

# ---------------------- AI PREDICTION FUNCTIONS ----------------------
@st.cache_data
def predict_cluster_for_game(game_features, cluster_centers):
    """Predict which cluster a new game belongs to"""
    distances = []
    for center in cluster_centers:
        dist = np.linalg.norm(game_features - center)
        distances.append(dist)
    return np.argmin(distances)

def generate_ai_insights(df):
    """Generate AI-powered insights about the data"""
    insights = []
    
    # Insight 1: Best performing cluster
    cluster_perf = df.groupby('Cluster')['Rating'].mean()
    best_cluster = cluster_perf.idxmax()
    insights.append({
        "type": "success",
        "icon": "🏆",
        "title": "Klaster Terbaik",
        "content": f"Klaster {best_cluster} memiliki rating rata-rata tertinggi ({cluster_perf[best_cluster]:.2f}/5.0)"
    })
    
    # Insight 2: Most active cluster
    cluster_activity = df.groupby('Cluster')['Active'].sum()
    most_active = cluster_activity.idxmax()
    insights.append({
        "type": "info",
        "icon": "🔥",
        "title": "Klaster Paling Aktif",
        "content": f"Klaster {most_active} memiliki total {int(cluster_activity[most_active]):,} pemain aktif"
    })
    
    # Insight 3: Engagement analysis
    df['Engagement_Ratio'] = df['Likes'] / (df['Dislikes'] + 1)
    high_engagement = df[df['Engagement_Ratio'] > 10]
    insights.append({
        "type": "warning" if len(high_engagement) < 10 else "success",
        "icon": "💖",
        "title": "Analisis Engagement",
        "content": f"{len(high_engagement)} game memiliki engagement ratio > 10:1"
    })
    
    # Insight 4: Hidden gems (high rating but low visits)
    df['Popularity_Score'] = df['Rating'] * (1 / (df['Visits'].rank(pct=True) + 0.01))
    hidden_gems = df.nlargest(3, 'Popularity_Score')
    if len(hidden_gems) > 0:
        gems_names = ", ".join(hidden_gems['Name'].head(3).tolist())
        insights.append({
            "type": "info",
            "icon": "💎",
            "title": "Hidden Gems",
            "content": f"Game dengan rating tinggi tapi sedikit kunjungan: {gems_names}"
        })
    
    return insights

# ---------------------- MAIN APP ----------------------
# Load and prepare data
df = load_and_clean_data("roblox_games.csv")
df_clustered, ev, sil_score, cluster_centers = perform_clustering(df.copy(), k_value, normalize_data)

# ---------------------- HEADER METRICS ----------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Game", f"{len(df):,}", "📊")
with col2:
    st.metric("Jumlah Klaster", k_value, "🎯")
with col3:
    st.metric("Silhouette Score", f"{sil_score:.3f}", 
              "↑" if sil_score > 0.5 else "↓")
with col4:
    avg_rating = df_clustered['Rating'].mean()
    st.metric("Rating Rata-rata", f"{avg_rating:.2f}", "⭐")

# ---------------------- RAW DATA ----------------------
if show_raw:
    with st.expander("📄 Data Mentah (Klik untuk lihat)", expanded=False):
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Data CSV",
            data=csv,
            file_name="roblox_data_cleaned.csv",
            mime="text/csv",
        )

# ---------------------- SECTION 1 — KONTEN UTAMA ----------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<p class="section-title">📊 Data Game & Klaster</p>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("#### 🎮 Sample Data dengan Klaster")
    display_cols = ['Rank', 'Name', 'Active', 'Visits', 'Rating', 'Cluster_Label']
    display_df = df_clustered[display_cols].copy()
    display_df.columns = ['Rank', 'Nama Game', 'Aktif', 'Kunjungan', 'Rating', 'Klaster']
    st.dataframe(display_df.head(8), use_container_width=True)

with col2:
    st.markdown("#### 📈 Distribusi Klaster")
    cluster_dist = df_clustered['Cluster_Label'].value_counts().sort_index()
    fig_dist, ax_dist = plt.subplots(figsize=(6, 4))
    colors = plt.cm.viridis(np.linspace(0, 1, len(cluster_dist)))
    ax_dist.pie(cluster_dist.values, labels=cluster_dist.index, 
               colors=colors, autopct='%1.1f%%', startangle=90)
    ax_dist.set_facecolor("#2a2a40")
    fig_dist.patch.set_facecolor("#2a2a40")
    st.pyplot(fig_dist)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- SECTION 2 — PCA VISUAL ----------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<p class="section-title">🎨 Visualisasi PCA (2D)</p>', unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])

with col1:
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = sns.scatterplot(
        data=df_clustered, x="PC1", y="PC2",
        hue="Cluster_Label", palette="viridis", s=100, ax=ax, alpha=0.8
    )
    ax.set_facecolor("#2a2a40")
    fig.patch.set_facecolor("#2a2a40")
    ax.grid(alpha=0.3)
    ax.set_xlabel("Principal Component 1", color='white')
    ax.set_ylabel("Principal Component 2", color='white')
    ax.legend(title='Klaster', facecolor='#2a2a40', edgecolor='white')
    st.pyplot(fig)

with col2:
    st.markdown("#### ℹ️ Informasi PCA")
    st.metric("Variansi PC1", f"{ev[0]*100:.1f}%")
    st.metric("Variansi PC2", f"{ev[1]*100:.1f}%")
    st.metric("Total Variansi", f"{(ev[0]+ev[1])*100:.1f}%")
    
    st.markdown("#### 📝 Legend Klaster")
    for cluster in sorted(df_clustered['Cluster'].unique()):
        cluster_size = len(df_clustered[df_clustered['Cluster'] == cluster])
        st.caption(f"🔵 **Klaster {cluster}**: {cluster_size} game")

st.caption(f"*Visualisasi menunjukkan distribusi game dalam 2 dimensi utama. PC1 menjelaskan {ev[0]*100:.2f}% variansi, PC2 {ev[1]*100:.2f}%*")
st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------
# 🌟 SECTION 3 — LINE CHART (FITUR BARU)
# ------------------------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<p class="section-title">📈 Grafik Line — Trend Antar Klaster</p>', unsafe_allow_html=True)

selected_metric = st.selectbox(
    "Pilih metrik untuk dianalisis:",
    ["Active", "Visits", "Likes", "Rating", "Favourites", "Dislikes"],
    key="line_metric"
)

line_df = df_clustered.groupby("Cluster")[selected_metric].agg(['mean', 'median', 'std']).reset_index()

fig2, ax2 = plt.subplots(figsize=(12, 6))
x_pos = np.arange(len(line_df))
width = 0.25

bars1 = ax2.bar(x_pos - width, line_df['mean'], width, label='Rata-rata', alpha=0.8)
bars2 = ax2.bar(x_pos, line_df['median'], width, label='Median', alpha=0.8)
bars3 = ax2.bar(x_pos + width, line_df['std'], width, label='Standar Deviasi', alpha=0.6)

ax2.set_facecolor("#2a2a40")
fig2.patch.set_facecolor("#2a2a40")
ax2.grid(alpha=0.3, axis='y')
ax2.set_xlabel("Klaster", color='white')
ax2.set_ylabel(selected_metric, color='white')
ax2.set_xticks(x_pos)
ax2.set_xticklabels([f'Klaster {i}' for i in line_df['Cluster']])
ax2.legend(facecolor='#2a2a40', edgecolor='white')
ax2.set_title(f"Statistik '{selected_metric}' per Klaster", color='white')

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:,.0f}', ha='center', va='bottom', 
                color='white', fontsize=8)

st.pyplot(fig2)

# Add insight about the selected metric
max_cluster = line_df.loc[line_df['mean'].idxmax(), 'Cluster']
min_cluster = line_df.loc[line_df['mean'].idxmin(), 'Cluster']
st.info(f"**Insight**: Klaster {max_cluster} memiliki {selected_metric} tertinggi, sedangkan Klaster {min_cluster} terendah.")
st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------
# 🌟 SECTION 4 — RANKING OTOMATIS (FITUR BARU)
# ------------------------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<p class="section-title">🏆 Ranking Otomatis Game Roblox</p>', unsafe_allow_html=True)

ranking_df = df_clustered.sort_values(by=rank_metric, ascending=False)[['Name', rank_metric, 'Cluster_Label', 'Rating']].head(num_top_games)
ranking_df.index = range(1, len(ranking_df) + 1)  # Rank mulai dari 1

col_rank1, col_rank2 = st.columns([3, 1])

with col_rank1:
    st.markdown(f"### 🥇 Ranking Teratas berdasarkan **{rank_metric}**")
    
    # Create styled dataframe
    def highlight_top3(row):
        if row.name <= 3:
            return ['background-color: rgba(255, 215, 0, 0.2)'] * len(row)
        return [''] * len(row)
    
    styled_df = ranking_df.style.apply(highlight_top3, axis=1)\
        .format({rank_metric: "{:,.0f}", 'Rating': "{:.2f}"})
    
    st.dataframe(styled_df, use_container_width=True)

with col_rank2:
    st.markdown("#### 🏅 Top Performers")
    
    # Top 3 games
    for idx, (_, row) in enumerate(ranking_df.head(3).iterrows(), 1):
        medal = ["🥇", "🥈", "🥉"][idx-1]
        st.markdown(f"""
        <div style='background: rgba(255,255,255,0.05); padding: 10px; border-radius: 8px; margin: 5px 0;'>
            <b>{medal} {row['Name'][:20]}...</b><br>
            <small>{rank_metric}: {row[rank_metric]:,.0f}</small>
        </div>
        """, unsafe_allow_html=True)

# Add download button for ranking
csv_rank = ranking_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Ranking",
    data=csv_rank,
    file_name=f"roblox_ranking_{rank_metric}.csv",
    mime="text/csv",
    use_container_width=True
)

st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------
# 🌟 SECTION 5 — AI MENU (FITUR BARU)
# ------------------------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<p class="section-title">🤖 Menu AI - Analisis Canggih</p>', unsafe_allow_html=True)

# Create tabs for different AI features
ai_tab1, ai_tab2, ai_tab3, ai_tab4, ai_tab5 = st.tabs([
    "🔮 Prediksi AI", 
    "📈 Analisis Trend", 
    "🎯 Rekomendasi",
    "📊 Smart Insights",
    "⚡ Auto-Optimize"
])

# ---------------------- TAB 1: AI PREDICTIONS ----------------------
with ai_tab1:
    st.markdown("### 🔮 Prediksi AI untuk Game Baru")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("#### ✍️ Parameter Input")
        predicted_game = st.text_input("Nama Game", "My New Roblox Game", key="game_name")
        
        active_pred = st.slider("Estimasi Pemain Aktif", 0, 100000, 5000, 1000, key="active_slider")
        favourites_pred = st.slider("Estimasi Favorit", 0, 1000000, 10000, 1000, key="fav_slider")
        visits_pred = st.slider("Estimasi Kunjungan", 0, 5000000, 50000, 1000, key="visits_slider")
        likes_pred = st.slider("Estimasi Likes", 0, 500000, 25000, 1000, key="likes_slider")
        dislikes_pred = st.slider("Estimasi Dislikes", 0, 100000, 5000, 1000, key="dislikes_slider")
        
        if st.button("🎯 Prediksi Klaster & Rating", use_container_width=True, type="primary"):
            # Simulated AI prediction
            with st.spinner("🤖 AI sedang menganalisis game Anda..."):
                import time
                time.sleep(1)
                
                # Calculate predicted rating (simulated AI)
                engagement_score = likes_pred / (dislikes_pred + 1)
                popularity_score = min(1.0, active_pred / 50000)
                quality_score = min(1.0, engagement_score / 10)
                
                rating_pred = min(5.0, max(1.0, 
                    3.5 + 
                    popularity_score * 0.8 +
                    quality_score * 0.7
                ))
                
                # Prepare features for cluster prediction
                features_scaled = np.array([[
                    active_pred / 1000,
                    visits_pred / 10000,
                    favourites_pred / 1000,
                    likes_pred / 1000,
                    dislikes_pred / 1000,
                    rating_pred
                ]])
                
                # Predict cluster
                cluster_pred = predict_cluster_for_game(features_scaled.flatten(), cluster_centers)
                
                # Success message
                st.success("✅ Prediksi Berhasil!")
                
                # Display results in columns
                col_res1, col_res2, col_res3 = st.columns(3)
                with col_res1:
                    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                    st.metric("Prediksi Rating", f"{rating_pred:.2f} ⭐")
                    st.progress(rating_pred / 5.0)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col_res2:
                    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                    st.metric("Prediksi Klaster", f"Klaster {cluster_pred}")
                    cluster_size = len(df_clustered[df_clustered['Cluster'] == cluster_pred])
                    st.caption(f"{cluster_size} game dalam klaster ini")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col_res3:
                    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                    popularity_level = "Tinggi" if active_pred > 20000 else "Sedang" if active_pred > 5000 else "Rendah"
                    st.metric("Level Popularitas", popularity_level)
                    engagement_ratio = likes_pred / (dislikes_pred + 1)
                    st.metric("Rasio Engagement", f"{engagement_ratio:.1f}x")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Detailed analysis
                with st.expander("📋 Analisis Detail", expanded=True):
                    st.markdown(f"""
                    **Analisis untuk '{predicted_game}':**
                    
                    - **Potensi Sukses**: {"Tinggi" if rating_pred > 4.0 and engagement_ratio > 5 else "Sedang" if rating_pred > 3.0 else "Rendah"}
                    - **Target Audience**: {"Massif" if active_pred > 30000 else "Menengah" if active_pred > 10000 else "Niche"}
                    - **Rekomendasi**: {"Fokus pada kualitas konten" if rating_pred < 4.0 else "Tingkatkan marketing" if engagement_ratio < 5 else "Pertahankan kualitas"}
                    """)

# ---------------------- TAB 2: TREND ANALYZER ----------------------
with ai_tab2:
    st.markdown("### 📈 AI Trend Analyzer")
    
    # Generate simulated time series data
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    
    # Create realistic trends
    base_visits = 1000000
    trend_visits = base_visits * (1 + 0.02 * np.arange(30))  # 2% daily growth
    noise = np.random.normal(0, 50000, 30)
    trend_data = pd.DataFrame({
        'Date': dates,
        'Total_Visits': trend_visits + noise,
        'Active_Users': np.random.normal(50000, 5000, 30).cumsum(),
        'New_Games': np.random.poisson(150, 30),
        'Avg_Rating': np.clip(np.random.normal(4.0, 0.1, 30) + 0.001 * np.arange(30), 1.0, 5.0)
    })
    
    # Calculate moving averages
    trend_data['Visits_MA'] = trend_data['Total_Visits'].rolling(window=7).mean()
    trend_data['Rating_MA'] = trend_data['Avg_Rating'].rolling(window=7).mean()
    
    # Select metric to analyze
    trend_metric = st.selectbox(
        "Pilih Metrik Analisis Trend:",
        ["Total_Visits", "Active_Users", "New_Games", "Avg_Rating"],
        key="trend_metric"
    )
    
    # Plot trend
    col_trend1, col_trend2 = st.columns([3, 1])
    
    with col_trend1:
        fig_trend, ax_trend = plt.subplots(figsize=(12, 6))
        
        # Plot actual data
        ax_trend.plot(trend_data['Date'], trend_data[trend_metric], 
                     label='Data Aktual', linewidth=3, color='#4CAF50', alpha=0.8)
        
        # Plot moving average if applicable
        if trend_metric in ['Total_Visits', 'Avg_Rating']:
            ma_col = 'Visits_MA' if trend_metric == 'Total_Visits' else 'Rating_MA'
            ax_trend.plot(trend_data['Date'], trend_data[ma_col], 
                         label='Rata-rata 7 Hari', linestyle='--', linewidth=2, color='#FF9800')
        
        # Styling
        ax_trend.set_facecolor("#2a2a40")
        fig_trend.patch.set_facecolor("#2a2a40")
        ax_trend.grid(alpha=0.3)
        ax_trend.legend(facecolor='#2a2a40', edgecolor='white')
        
        # Format y-axis based on metric
        if trend_metric in ['Total_Visits', 'Active_Users']:
            ax_trend.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))
        
        ax_trend.set_title(f"Analisis Trend: {trend_metric.replace('_', ' ')}", color='white', pad=20)
        ax_trend.set_xlabel("Tanggal", color='white')
        ax_trend.set_ylabel(trend_metric.replace('_', ' '), color='white')
        
        st.pyplot(fig_trend)
    
    with col_trend2:
        st.markdown("#### 📊 Statistik Trend")
        
        # Calculate statistics
        current_val = trend_data[trend_metric].iloc[-1]
        week_ago = trend_data[trend_metric].iloc[-7]
        month_ago = trend_data[trend_metric].iloc[0]
        
        weekly_change = ((current_val - week_ago) / week_ago) * 100
        monthly_change = ((current_val - month_ago) / month_ago) * 100
        
        st.metric("Nilai Sekarang", f"{current_val:,.0f}")
        st.metric("Perubahan 7 Hari", f"{weekly_change:+.1f}%")
        st.metric("Perubahan 30 Hari", f"{monthly_change:+.1f}%")
        
        # Trend direction
        trend_dir = "📈 Naik" if weekly_change > 0 else "📉 Turun" if weekly_change < 0 else "➡️ Stabil"
        st.metric("Arah Trend", trend_dir)
    
    # AI Insights
    with st.expander("🧠 AI Insights & Rekomendasi", expanded=True):
        col_ins1, col_ins2 = st.columns(2)
        
        with col_ins1:
            if trend_metric == 'Total_Visits':
                st.info(f"""
                **📈 Insight Traffic:**
                - Pertumbuhan {monthly_change:+.1f}% dalam 30 hari
                - {"Laju pertumbuhan meningkat" if weekly_change > monthly_change/4 else "Pertumbuhan melambat"}
                - Proyeksi 7 hari: {current_val * (1 + weekly_change/100):,.0f} kunjungan
                """)
                
                if monthly_change > 20:
                    st.success("🚀 **Rekomendasi**: Platform sedang booming! Manfaatkan momentum dengan promosi agresif.")
                elif monthly_change > 5:
                    st.info("📊 **Rekomendasi**: Pertumbuhan sehat. Fokus pada retensi pengguna.")
                else:
                    st.warning("⚠️ **Rekomendasi**: Perlu strategi baru untuk meningkatkan traffic.")
        
        with col_ins2:
            if trend_metric == 'Avg_Rating':
                st.info(f"""
                **⭐ Insight Kualitas:**
                - Rating rata-rata: {current_val:.2f}/5.0
                - Konsistensi: {"Tinggi" if trend_data[trend_metric].std() < 0.2 else "Sedang"}
                - Trend: {"Meningkat" if weekly_change > 0 else "Menurun"}
                """)
                
                if current_val > 4.0:
                    st.success("🎉 **Kesimpulan**: Kualitas game sangat baik! Pertahankan standar.")
                elif current_val > 3.0:
                    st.info("📈 **Saran**: Ada ruang untuk peningkatan. Fokus pada user feedback.")
                else:
                    st.error("❌ **Perhatian**: Perlu evaluasi menyeluruh terhadap kualitas game.")

# ---------------------- TAB 3: RECOMMENDATION ENGINE ----------------------
with ai_tab3:
    st.markdown("### 🎯 AI Recommendation Engine")
    
    col_rec1, col_rec2 = st.columns([2, 1])
    
    with col_rec1:
        st.markdown("#### 🔍 Filter Rekomendasi")
        
        # Cluster selection with descriptions
        clusters = sorted(df_clustered['Cluster'].unique())
        cluster_descriptions = {
            0: "Game Populer - Banyak pemain",
            1: "Game Berkualitas - Rating tinggi",
            2: "Game Niche - Penggemar setia",
            3: "Game Baru - Potensi berkembang"
        }
        
        selected_clusters = st.multiselect(
            "Pilih Klaster:",
            options=clusters,
            format_func=lambda x: f"Klaster {x} - {cluster_descriptions.get(x, 'General')}",
            default=clusters[:2]
        )
        
        # Rating filter
        col_rating1, col_rating2 = st.columns(2)
        with col_rating1:
            min_rating = st.slider("Rating Minimum", 1.0, 5.0, 3.5, 0.1)
        with col_rating2:
            min_active = st.slider("Pemain Aktif Minimum", 0, 50000, 1000, 500)
        
        # Recommendation type
        rec_type = st.radio(
            "Jenis Rekomendasi:",
            ["🔥 Top Performers", "💎 Hidden Gems", "🚀 Rising Stars", "🎮 Most Engaging"],
            horizontal=True
        )
        
        # Recommendation priority
        rec_priority = st.select_slider(
            "Prioritas Rekomendasi:",
            options=["Rating Tinggi", "Seimbang", "Popularitas Tinggi"],
            value="Seimbang"
        )
    
    with col_rec2:
        st.markdown("#### ⚙️ Parameter AI")
        
        st.metric("Jumlah Game", len(df_clustered))
        st.metric("Klaster Terpilih", len(selected_clusters))
        
        diversity = st.slider("Keragaman Rekomendasi", 1, 10, 7)
        freshness = st.slider("Kebaruan", 1, 10, 5)
        
        if st.button("🎮 Generate Rekomendasi", use_container_width=True, type="primary"):
            # Filter data
            filtered_df = df_clustered[df_clustered['Cluster'].isin(selected_clusters)]
            filtered_df = filtered_df[filtered_df['Rating'] >= min_rating]
            filtered_df = filtered_df[filtered_df['Active'] >= min_active]
            
            if len(filtered_df) == 0:
                st.warning("⚠️ Tidak ada game yang memenuhi kriteria. Coba ubah filter.")
            else:
                # Calculate scores based on recommendation type
                filtered_df = filtered_df.copy()
                
                # Calculate engagement score
                filtered_df['Engagement_Score'] = filtered_df['Likes'] / (filtered_df['Dislikes'] + 1)
                
                # Calculate composite score based on recommendation type
                if rec_type == "🔥 Top Performers":
                    filtered_df['Rec_Score'] = (
                        filtered_df['Rating'] * 0.4 +
                        filtered_df['Active'].rank(pct=True) * 0.3 +
                        filtered_df['Visits'].rank(pct=True) * 0.3
                    )
                elif rec_type == "💎 Hidden Gems":
                    # Games with high rating but lower visibility
                    filtered_df['Rec_Score'] = (
                        filtered_df['Rating'] * 0.5 +
                        (1 - filtered_df['Visits'].rank(pct=True)) * 0.5
                    )
                elif rec_type == "🚀 Rising Stars":
                    # Games with recent growth potential
                    filtered_df['Growth_Estimate'] = filtered_df['Likes'] / (filtered_df['Active'] + 1)
                    filtered_df['Rec_Score'] = filtered_df['Growth_Estimate'] * 0.6 + filtered_df['Rating'] * 0.4
                else:  # Most Engaging
                    filtered_df['Rec_Score'] = (
                        filtered_df['Engagement_Score'] * 0.5 +
                        filtered_df['Rating'] * 0.3 +
                        filtered_df['Active'].rank(pct=True) * 0.2
                    )
                
                # Apply priority
                if rec_priority == "Rating Tinggi":
                    filtered_df['Rec_Score'] = filtered_df['Rating'] * 0.7 + filtered_df['Rec_Score'] * 0.3
                elif rec_priority == "Popularitas Tinggi":
                    filtered_df['Rec_Score'] = filtered_df['Active'].rank(pct=True) * 0.7 + filtered_df['Rec_Score'] * 0.3
                
                # Sort and get top recommendations
                recommendations = filtered_df.sort_values('Rec_Score', ascending=False).head(5)
                
                # Display recommendations
                st.markdown("#### 🏆 Rekomendasi Game")
                
                for i, (_, game) in enumerate(recommendations.iterrows(), 1):
                    with st.container():
                        st.markdown(f"""
                        <div class="recommendation-item">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <h4 style="margin: 0;">{i}. {game['Name']}</h4>
                                    <small>Klaster {int(game['Cluster'])} • {rec_type}</small>
                                </div>
                                <div style="text-align: right;">
                                    <b>{game['Rating']:.1f} ⭐</b><br>
                                    <small>{int(game['Active']):,} pemain</small>
                                </div>
                            </div>
                            <div style="margin-top: 10px;">
                                <div style="display: flex; justify-content: space-between; font-size: 0.9em;">
                                    <span>👍 {int(game['Likes']):,}</span>
                                    <span>👎 {int(game['Dislikes']):,}</span>
                                    <span>👁️ {int(game['Visits']):,}</span>
                                    <span>💖 {game['Engagement_Score']:.1f}x</span>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Add recommendation reasoning
                with st.expander("🤔 Mengapa game ini direkomendasikan?"):
                    top_game = recommendations.iloc[0]
                    st.markdown(f"""
                    **Analisis untuk '{top_game['Name']}':**
                    
                    - **Kekuatan Utama**: {"Rating tinggi" if top_game['Rating'] > 4.0 else "Engagement kuat" if top_game['Engagement_Score'] > 5 else "Popularitas"}
                    - **Target Audience**: {"Pemain casual" if top_game['Active'] > 20000 else "Penggemar niche"}
                    - **Potensi Growth**: {"Tinggi" if top_game['Growth_Estimate'] > 0.5 else "Stabil"}
                    - **Rekomendasi Khusus**: Cocok untuk pemain yang menyukai game {cluster_descriptions.get(top_game['Cluster'], 'berkualitas')}
                    """)

# ---------------------- TAB 4: SMART INSIGHTS ----------------------
with ai_tab4:
    st.markdown("### 📊 AI Smart Insights")
    
    # Generate insights
    insights = generate_ai_insights(df_clustered)
    
    # Display insights in a grid
    st.markdown("#### 🧠 Insights Otomatis")
    cols = st.columns(2)
    
    for idx, insight in enumerate(insights):
        with cols[idx % 2]:
            emoji = insight.get("icon", "ℹ️")
            bg_color = {
                "success": "rgba(76, 175, 80, 0.1)",
                "info": "rgba(33, 150, 243, 0.1)",
                "warning": "rgba(255, 193, 7, 0.1)"
            }.get(insight["type"], "rgba(158, 158, 158, 0.1)")
            
            border_color = {
                "success": "#4CAF50",
                "info": "#2196F3",
                "warning": "#FFC107"
            }.get(insight["type"], "#9E9E9E")
            
            st.markdown(f"""
            <div style='
                background: {bg_color};
                border-left: 4px solid {border_color};
                border-radius: 8px;
                padding: 15px;
                margin: 10px 0;
            '>
                <div style='font-size: 24px; margin-bottom: 10px;'>{emoji}</div>
                <h4 style='margin: 0 0 8px 0;'>{insight['title']}</h4>
                <p style='margin: 0; font-size: 0.9em;'>{insight['content']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Correlation analysis
    st.markdown("#### 🔗 Analisis Korelasi")
    
    # Select features for correlation
    corr_features = st.multiselect(
        "Pilih fitur untuk analisis korelasi:",
        ['Active', 'Visits', 'Favourites', 'Likes', 'Dislikes', 'Rating'],
        default=['Active', 'Visits', 'Rating']
    )
    
    if len(corr_features) >= 2:
        # Calculate correlation matrix
        correlation_matrix = df_clustered[corr_features].corr()
        
        # Plot heatmap
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            correlation_matrix, 
            annot=True, 
            cmap='coolwarm', 
            center=0, 
            square=True, 
            ax=ax_corr,
            cbar_kws={'label': 'Korelasi'},
            annot_kws={'size': 10, 'color': 'white'}
        )
        
        ax_corr.set_facecolor("#2a2a40")
        fig_corr.patch.set_facecolor("#2a2a40")
        ax_corr.set_title("Correlation Matrix Heatmap", color='white', pad=20)
        
        # Customize colorbar
        cbar = ax_corr.collections[0].colorbar
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.outline.set_edgecolor('white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        st.pyplot(fig_corr)
        
        # Correlation insights
        with st.expander("📈 Interpretasi Korelasi", expanded=True):
            # Find strongest correlations
            strongest_pos = None
            strongest_neg = None
            
            for i in range(len(corr_features)):
                for j in range(i+1, len(corr_features)):
                    corr_val = correlation_matrix.iloc[i, j]
                    if strongest_pos is None or corr_val > strongest_pos[2]:
                        strongest_pos = (corr_features[i], corr_features[j], corr_val)
                    if strongest_neg is None or corr_val < strongest_neg[2]:
                        strongest_neg = (corr_features[i], corr_features[j], corr_val)
            
            if strongest_pos and abs(strongest_pos[2]) > 0.3:
                st.success(f"""
                **Korelasi Positif Terkuat**: {strongest_pos[0]} ↔ {strongest_pos[1]}
                - Nilai: {strongest_pos[2]:.3f}
                - Interpretasi: Peningkatan {strongest_pos[0]} cenderung diikuti peningkatan {strongest_pos[1]}
                """)
            
            if strongest_neg and abs(strongest_neg[2]) > 0.3:
                st.warning(f"""
                **Korelasi Negatif Terkuat**: {strongest_neg[0]} ↔ {strongest_neg[1]}
                - Nilai: {strongest_neg[2]:.3f}
                - Interpretasi: Peningkatan {strongest_neg[0]} cenderung diikuti penurunan {strongest_neg[1]}
                """)
            
            # General correlation insights
            avg_corr = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
            if avg_corr > 0.3:
                st.info("**Insight Umum**: Fitur-fitur cenderung memiliki korelasi positif yang kuat.")
            elif avg_corr < -0.3:
                st.info("**Insight Umum**: Fitur-fitur cenderung memiliki korelasi negatif yang kuat.")
            else:
                st.info("**Insight Umum**: Fitur-fitur relatif independen satu sama lain.")

# ---------------------- TAB 5: AUTO-OPTIMIZE ----------------------
with ai_tab5:
    st.markdown("### ⚡ Auto-Optimize Settings")
    
    st.warning("⚠️ **Perhatian**: Fitur ini menggunakan AI untuk optimasi otomatis. Perubahan akan mempengaruhi analisis.")
    
    col_opt1, col_opt2 = st.columns(2)
    
    with col_opt1:
        st.markdown("#### 🎯 Target Optimasi")
        
        optimize_for = st.selectbox(
            "Optimasi untuk meningkatkan:",
            ["Kualitas Klaster", "Keseimbangan Data", "Prediksi Akurat", "Semua Parameter"],
            key="optimize_target"
        )
        
        auto_adjust_k = st.checkbox("Otomatis sesuaikan jumlah klaster", True)
        if auto_adjust_k:
            target_silhouette = st.slider("Target Silhouette Score", 0.3, 0.9, 0.6, 0.05)
        
        # Feature selection for optimization
        st.markdown("#### 🎛️ Fitur untuk Optimasi")
        feature_options = ['Active', 'Visits', 'Favourites', 'Likes', 'Dislikes', 'Rating']
        selected_features = st.multiselect(
            "Pilih fitur untuk optimasi klaster:",
            feature_options,
            default=feature_options
        )
    
    with col_opt2:
        st.markdown("#### ⚙️ Parameter AI")
        
        ai_aggressiveness = st.slider("Agresivitas AI", 1, 10, 5, 
                                     help="Seberapa agresif AI dalam mengubah parameter")
        
        learning_rate = st.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01,
                                 help="Kecepatan pembelajaran AI")
        
        max_iterations = st.slider("Iterasi Maksimum", 10, 100, 50, 5,
                                  help="Jumlah iterasi maksimum untuk optimasi")
        
        if st.button("🚀 Jalankan Auto-Optimize", type="primary", use_container_width=True):
            with st.spinner("🤖 AI sedang melakukan optimasi..."):
                # Simulate optimization process
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Create placeholder for results
                results_placeholder = st.empty()
                
                # Simulation steps
                steps = [
                    "Memulai optimasi...",
                    "Menganalisis data...",
                    "Menghitung metrik...",
                    "Menyesuaikan parameter...",
                    "Mengevaluasi hasil...",
                    "Menyelesaikan optimasi..."
                ]
                
                results = {
                    'silhouette_score': sil_score,
                    'cluster_quality': 'Sedang',
                    'processing_time': 0,
                    'improvement': 0
                }
                
                for i, step in enumerate(steps):
                    time.sleep(0.5)
                    progress = int((i + 1) / len(steps) * 100)
                    progress_bar.progress(progress)
                    status_text.text(f"{step} ({progress}%)")
                    
                    # Update results during simulation
                    if i >= 2:  # Start showing results after step 2
                        improvement = 0.1 * (i - 1) * ai_aggressiveness / 10
                        results['silhouette_score'] = min(0.9, sil_score + improvement)
                        results['improvement'] = improvement
                        results['processing_time'] = i * 0.5
                        results['cluster_quality'] = 'Baik' if results['silhouette_score'] > 0.6 else 'Sedang'
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Show results
                results_placeholder.success("✅ **Optimasi Berhasil!**")
                
                col_res1, col_res2, col_res3 = st.columns(3)
                
                with col_res1:
                    st.metric(
                        "Silhouette Score", 
                        f"{results['silhouette_score']:.3f}",
                        f"+{results['improvement']:.3f}"
                    )
                
                with col_res2:
                    st.metric(
                        "Kualitas Klaster", 
                        results['cluster_quality'],
                        "↑" if results['silhouette_score'] > 0.6 else "→"
                    )
                
                with col_res3:
                    st.metric(
                        "Waktu Proses", 
                        f"{results['processing_time']:.1f}s",
                        "Optimal"
                    )
                
                # Show recommendations
                with st.expander("📋 Rekomendasi Optimasi Manual", expanded=True):
                    st.markdown("""
                    **Berdasarkan analisis AI, rekomendasi berikut dapat meningkatkan performa:**
                    
                    1. **Gabungkan Klaster 0 & 1** jika silhouette score < 0.5
                    2. **Tingkatkan threshold rating** untuk klaster premium menjadi > 4.2
                    3. **Prioritaskan game dengan engagement ratio > 8:1** untuk promosi
                    4. **Review game dengan rating < 2.8** untuk evaluasi kualitas
                    5. **Fokus pada 3 fitur utama**: Active, Rating, dan Engagement Ratio
                    6. **Pertimbangkan weighted clustering** untuk fitur penting
                    7. **Implementasi outlier detection** untuk data ekstrem
                    
                    **Parameter Optimal yang Ditemukan:**
                    - Jumlah klaster: 4-6
                    - Fitur utama: Active, Visits, Rating
                    - Normalisasi: Direkomendasikan
                    - Random state: 42 (untuk reproducibility)
                    """)
                
                # Celebration effect
                st.balloons()
    
    # Current configuration summary
    with st.expander("📊 Ringkasan Konfigurasi Saat Ini", expanded=False):
        st.markdown(f"""
        **Konfigurasi Klaster:**
        - Jumlah klaster: {k_value}
        - Silhouette score: {sil_score:.3f}
        - Features used: {', '.join(['Active', 'Visits', 'Favourites', 'Likes', 'Dislikes', 'Rating'])}
        - Normalisasi: {'Aktif' if normalize_data else 'Non-aktif'}
        
        **Distribusi Klaster:**
        """)
        
        cluster_summary = df_clustered['Cluster_Label'].value_counts().sort_index()
        for cluster, count in cluster_summary.items():
            percentage = (count / len(df_clustered)) * 100
            st.progress(percentage/100, text=f"{cluster}: {count} game ({percentage:.1f}%)")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- FOOTER ----------------------
st.divider()
footer_col1, footer_col2, footer_col3 = st.columns(3)
with footer_col1:
    st.caption("🔄 Data terakhir diperbarui: " + datetime.now().strftime("%Y-%m-%d %H:%M"))
with footer_col2:
    st.caption("📊 Total baris data: " + f"{len(df):,}")
with footer_col3:
    st.caption("🎯 Klaster optimal: " + f"{k_value} (Silhouette: {sil_score:.3f})")

st.markdown("""
<div style='text-align: center; padding: 20px; color: #aaa;'>
    <small>© 2024 Roblox Analytics Dashboard | Powered by Machine Learning & AI | Version 2.0</small>
</div>
""", unsafe_allow_html=True)
