import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os

# ─────────────────────────────────────────────
# Sayfa ayarları
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Parkinson Tahmin Sistemi",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CSS — koyu, klinik-fütüristik tema
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&display=swap');

/* Global arka plan */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: #080c14 !important;
    font-family: 'DM Mono', monospace !important;
    color: #c8d8f0 !important;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { background: #0d1220 !important; }
.block-container { padding: 2rem 3rem 4rem 3rem !important; max-width: 1300px; }

/* Başlık alanı */
.hero {
    text-align: center;
    padding: 3rem 2rem 2rem;
    position: relative;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.hero-sub {
    font-size: 0.85rem;
    color: #4a6080;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    margin-top: 0.5rem;
}
.hero-line {
    height: 1px;
    background: linear-gradient(90deg, transparent, #1e3a5f, #3b82f6, #1e3a5f, transparent);
    margin: 2rem auto;
    max-width: 600px;
}

/* Section başlıkları */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #3b82f6;
    margin-bottom: 1rem;
    padding-left: 0.75rem;
    border-left: 2px solid #3b82f6;
}

/* Kart */
.card {
    background: #0d1525;
    border: 1px solid #1a2c4a;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #1e3a5f, #3b82f6, #1e3a5f);
}

/* Input widget override */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    background: #0a1020 !important;
    border: 1px solid #1e3a5f !important;
    color: #c8d8f0 !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
}
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextInput"] input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.15) !important;
}
label {
    color: #7aa0c4 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
}

/* Slider */
[data-testid="stSlider"] > div > div > div > div {
    background: #3b82f6 !important;
}

/* Buton */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #1d4ed8, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.75rem 2.5rem !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
    cursor: pointer !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(99,102,241,0.4) !important;
}

/* Sonuç kutuları */
.result-positive {
    background: linear-gradient(135deg, #1a0a0a, #2d0f0f);
    border: 1px solid #dc2626;
    border-radius: 16px;
    padding: 2.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    animation: glow-red 2s ease-in-out infinite alternate;
}
.result-negative {
    background: linear-gradient(135deg, #071a12, #0d2b1e);
    border: 1px solid #16a34a;
    border-radius: 16px;
    padding: 2.5rem;
    text-align: center;
    animation: glow-green 2s ease-in-out infinite alternate;
}
@keyframes glow-red {
    from { box-shadow: 0 0 20px rgba(220,38,38,0.2); }
    to   { box-shadow: 0 0 40px rgba(220,38,38,0.45); }
}
@keyframes glow-green {
    from { box-shadow: 0 0 20px rgba(22,163,74,0.2); }
    to   { box-shadow: 0 0 40px rgba(22,163,74,0.45); }
}
.result-icon { font-size: 3rem; margin-bottom: 0.75rem; }
.result-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    margin: 0;
}
.result-prob {
    font-size: 3.5rem;
    font-weight: 800;
    font-family: 'Syne', sans-serif;
    margin: 0.5rem 0;
}
.result-desc { font-size: 0.8rem; color: #7aa0c4; line-height: 1.6; }

/* Olasılık çubuğu */
.prob-bar-container {
    margin: 1.5rem 0;
    background: #0a1020;
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
    border: 1px solid #1a2c4a;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.8s ease;
}

/* Uyarı kutusu */
.warning-box {
    background: #1a1400;
    border: 1px solid #854d0e;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.78rem;
    color: #ca8a04;
    margin-top: 1.5rem;
    line-height: 1.7;
}

/* Metrik kartlar */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
.metric-card {
    flex: 1;
    background: #0a1020;
    border: 1px solid #1a2c4a;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.metric-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #60a5fa;
}
.metric-lbl { font-size: 0.68rem; color: #4a6080; letter-spacing: 0.1em; text-transform: uppercase; }

/* Divider */
.divider { border: none; border-top: 1px solid #1a2c4a; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Model yükleme
# ─────────────────────────────────────────────

class PyTorchANN(nn.Module):
    def __init__(self, input_dim, hidden_layers):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for units in hidden_layers:
            layers.append(nn.Linear(prev_dim, units))
            layers.append(nn.ReLU())
            prev_dim = units
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


@st.cache_resource
def load_model_and_scaler():
    MODEL_PATH = "best_medicaldataset_pytorch_model_expanded.pth"
    DATA_PATH  = "parkinsons_updrs.csv"

    # Scaler'ı eğitim verisiyle fit et
    df = pd.read_csv(DATA_PATH)
    median_updrs = df['total_UPDRS'].median()
    df['Result_num'] = (df['total_UPDRS'] > median_updrs).astype(int)
    data = df.drop_duplicates().reset_index(drop=True)
    X = data.drop(columns=['Result_num', 'subject#', 'test_time'])

    imp = SimpleImputer(strategy='median')
    X_imp = imp.fit_transform(X)
    scaler = StandardScaler()
    scaler.fit(X_imp)

    model = PyTorchANN(input_dim=20, hidden_layers=[8])
    state = torch.load(MODEL_PATH, map_location='cpu')
    model.load_state_dict(state)
    model.eval()

    feature_cols = X.columns.tolist()
    return model, scaler, imp, feature_cols, median_updrs


def predict(model, scaler, imp, values):
    arr = np.array(values, dtype=np.float32).reshape(1, -1)
    arr_imp = imp.transform(arr)
    arr_scaled = scaler.transform(arr_imp)
    tensor = torch.tensor(arr_scaled, dtype=torch.float32)
    with torch.no_grad():
        logit = model(tensor).item()
    prob = 1 / (1 + np.exp(-logit))
    return prob


# ─────────────────────────────────────────────
# Hero başlık
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <p class="hero-sub">🧠 Yapay Sinir Ağı Tabanlı Klinik Karar Destek</p>
    <h1 class="hero-title">Parkinson Tahmin Sistemi</h1>
    <div class="hero-line"></div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Model yükle
# ─────────────────────────────────────────────
try:
    model, scaler, imp, feature_cols, median_updrs = load_model_and_scaler()
    st.markdown("""
    <div style="text-align:center; margin-bottom:2rem;">
        <span style="background:#071a12; border:1px solid #16a34a; color:#4ade80;
                     font-size:0.72rem; border-radius:999px; padding:0.3rem 1rem;
                     letter-spacing:0.1em;">
            ✓ MODEL HAZIR — 20 ÖZELLİK / ANN [20 → 16 → 1]
        </span>
    </div>
    """, unsafe_allow_html=True)
    model_loaded = True
except FileNotFoundError as e:
    st.error(f"Model veya veri dosyası bulunamadı. `app.py` ile aynı klasöre kopyaladığınızdan emin olun.\n\nEksik dosya: {e}")
    model_loaded = False
    st.stop()

# ─────────────────────────────────────────────
# 2 sütun düzeni
# ─────────────────────────────────────────────
left, right = st.columns([3, 2], gap="large")

with left:
    # ── Hasta bilgileri ─────────────────────────
    st.markdown('<div class="section-label">👤 Kişisel Bilgiler</div>', unsafe_allow_html=True)
    with st.container():
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input(
                "Yaş",
                min_value=30, max_value=100, value=70, step=1,
                help="Hastanın yaşını girin (30-100 arası).")
        with c2:
            sex = st.selectbox(
                "Cinsiyet",
                options=[("Erkek", 0), ("Kadın", 1)],
                format_func=lambda x: x[0],
                help="Hastanın biyolojik cinsiyetini seçin.")[1]
        with c3:
            motor_updrs = st.number_input(
                "Motor Hareket Puanı",
                min_value=0.0, max_value=100.0, value=25.0, step=0.1, format="%.2f",
                help="Hastanın motor (hareket) bozukluğu şiddetini gösteren klinik puan. "
                     "Düşük değer → hafif belirti, yüksek değer → ağır belirti. (0–100)")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Ses titreme ölçümleri ───────────────────
    st.markdown('<div class="section-label">🎙️ Seste Frekans Titremesi (Jitter)</div>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.75rem;color:#4a6080;margin:-0.5rem 0 1rem;">Ses frekansının ne kadar düzensiz değiştiğini ölçer. Yüksek değerler ses titremesine işaret eder.</p>', unsafe_allow_html=True)
    with st.container():
        j1, j2, j3 = st.columns(3)
        with j1:
            jitter_pct = st.number_input(
                "Titreme Oranı (%)",
                min_value=0.0, max_value=0.1, value=0.005, step=0.0001, format="%.5f",
                help="Ses frekansındaki titreme yüzdesi. Normal ses: ~%0.4, titreyen ses: >%1")
        with j2:
            jitter_abs = st.number_input(
                "Mutlak Titreme Süresi (sn)",
                min_value=0.0, max_value=0.001, value=0.00003, step=0.000001, format="%.6f",
                help="Ardışık ses periyotları arasındaki ortalama süre farkı (saniye cinsinden).")
        with j3:
            jitter_rap = st.number_input(
                "Kısa Dönem Titreme (RAP)",
                min_value=0.0, max_value=0.1, value=0.003, step=0.0001, format="%.5f",
                help="3 ardışık periyodun ortalama titreme oranı. Küçük değer daha istikrarlı ses demektir.")
        j4, j5, j6 = st.columns(3)
        with j4:
            jitter_ppq5 = st.number_input(
                "5 Periyotlu Titreme (PPQ5)",
                min_value=0.0, max_value=0.1, value=0.003, step=0.0001, format="%.5f",
                help="5 ardışık periyodun ortalama titreme oranı.")
        with j5:
            jitter_ddp = st.number_input(
                "Fark Titreme Oranı (DDP)",
                min_value=0.0, max_value=0.2, value=0.009, step=0.0001, format="%.5f",
                help="RAP değerinin 3 katına eşdeğer, fark tabanlı titreme ölçümü.")
        with j6:
            st.empty()

    st.markdown('<div class="section-label" style="margin-top:1rem">📊 Seste Genlik Titremesi (Shimmer)</div>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.75rem;color:#4a6080;margin:-0.5rem 0 1rem;">Sesin yüksekliğinin (amplitüd) ne kadar düzensiz değiştiğini ölçer. Yüksek değer ses kısıklığına işaret edebilir.</p>', unsafe_allow_html=True)
    with st.container():
        s1, s2, s3 = st.columns(3)
        with s1:
            shimmer = st.number_input(
                "Genlik Titremesi (Oran)",
                min_value=0.0, max_value=1.0, value=0.03, step=0.001, format="%.4f",
                help="Ardışık ses dalgaları arasındaki yükseklik farkının oranı. Normal: ~0.03")
        with s2:
            shimmer_db = st.number_input(
                "Genlik Titremesi (dB)",
                min_value=0.0, max_value=2.0, value=0.28, step=0.01, format="%.3f",
                help="Ses yüksekliği titremesinin desibel cinsinden ifadesi. Normal: ~0.25 dB")
        with s3:
            shimmer_apq3 = st.number_input(
                "3 Periyotlu Genlik Titremesi",
                min_value=0.0, max_value=1.0, value=0.015, step=0.001, format="%.4f",
                help="3 ardışık ses dalgasının genlik titremesi ortalaması.")
        s4, s5, s6 = st.columns(3)
        with s4:
            shimmer_apq5 = st.number_input(
                "5 Periyotlu Genlik Titremesi",
                min_value=0.0, max_value=1.0, value=0.018, step=0.001, format="%.4f",
                help="5 ardışık ses dalgasının genlik titremesi ortalaması.")
        with s5:
            shimmer_apq11 = st.number_input(
                "11 Periyotlu Genlik Titremesi",
                min_value=0.0, max_value=1.0, value=0.025, step=0.001, format="%.4f",
                help="11 ardışık ses dalgasının genlik titremesi ortalaması.")
        with s6:
            shimmer_dda = st.number_input(
                "Fark Genlik Titremesi (DDA)",
                min_value=0.0, max_value=1.0, value=0.046, step=0.001, format="%.4f",
                help="APQ3 değerinin 3 katına eşdeğer fark tabanlı genlik titremesi.")

    st.markdown('<div class="section-label" style="margin-top:1rem">🔬 Ses Kalitesi & Karmaşıklık Ölçümleri</div>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.75rem;color:#4a6080;margin:-0.5rem 0 1rem;">Sesin matematiksel düzensizliğini ve gürültü oranını ölçen gelişmiş parametreler.</p>', unsafe_allow_html=True)
    with st.container():
        h1, h2, h3 = st.columns(3)
        with h1:
            nhr = st.number_input(
                "Gürültü / Ses Oranı (NHR)",
                min_value=0.0, max_value=1.0, value=0.015, step=0.001, format="%.4f",
                help="Sesteki istenmeyen gürültünün harmonik sese oranı. Düşük değer temiz ses demektir.")
        with h2:
            hnr = st.number_input(
                "Ses Netliği (HNR, dB)",
                min_value=0.0, max_value=40.0, value=21.0, step=0.1, format="%.3f",
                help="Harmonik sesin gürültüye oranı. Yüksek değer daha net ve sağlıklı ses demektir. Normal: >20 dB")
        with h3:
            rpde = st.number_input(
                "Ses Düzensizlik Skoru (RPDE)",
                min_value=0.0, max_value=1.0, value=0.45, step=0.01, format="%.4f",
                help="Sesin ne kadar düzensiz/kaotik olduğunu ölçer. 0'a yakın: düzenli ses, 1'e yakın: çok düzensiz ses.")
        h4, h5, h6 = st.columns(3)
        with h4:
            dfa = st.number_input(
                "Ses Akış Tutarlılığı (DFA)",
                min_value=0.0, max_value=1.0, value=0.55, step=0.01, format="%.4f",
                help="Sesin uzun dönemli tutarlılığını ölçer. 0.5 civarı normal, sapmalar düzensizliğe işaret eder.")
        with h5:
            ppe = st.number_input(
                "Perde Değişkenliği (PPE)",
                min_value=0.0, max_value=1.0, value=0.18, step=0.01, format="%.4f",
                help="Ses perdesinin (tonunun) ne kadar değişken olduğunu ölçer. Yüksek değer ses kontrolündeki zorluğu gösterir.")
        with h6:
            total_updrs = st.number_input(
                "Genel Hastalık Şiddet Puanı",
                min_value=0.0, max_value=100.0, value=25.0, step=0.1, format="%.2f",
                help="Parkinson hastalığının genel şiddetini gösteren klinik ölçek puanı (UPDRS). "
                     "Düşük: hafif, yüksek: ileri evre. Eşik değer: ~27.6")

    st.markdown('<br>', unsafe_allow_html=True)
    predict_btn = st.button("🔍  Tahmin Et", use_container_width=True)

# ─────────────────────────────────────────────
# SAĞ: Sonuç paneli
# ─────────────────────────────────────────────
with right:
    st.markdown('<div class="section-label">Analiz Sonucu</div>', unsafe_allow_html=True)

    if predict_btn:
        values = [
            age, sex, motor_updrs, total_updrs,
            jitter_pct, jitter_abs, jitter_rap, jitter_ppq5, jitter_ddp,
            shimmer, shimmer_db, shimmer_apq3, shimmer_apq5, shimmer_apq11, shimmer_dda,
            nhr, hnr, rpde, dfa, ppe
        ]

        prob = predict(model, scaler, imp, values)
        prob_pct = prob * 100

        if prob >= 0.5:
            st.markdown(f"""
            <div class="result-positive">
                <div class="result-icon">⚠️</div>
                <p class="result-title" style="color:#f87171;">Yüksek Risk Saptandı</p>
                <p class="result-prob" style="color:#ef4444;">{prob_pct:.1f}%</p>
                <p class="result-desc">
                    Model, girilen vokal parametrelere dayanarak<br>
                    <strong style="color:#f87171;">Parkinson semptomları ile uyumlu bir profil</strong> tespit etti.<br>
                    (total UPDRS medyan eşiği: {median_updrs:.1f})
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-negative">
                <div class="result-icon">✅</div>
                <p class="result-title" style="color:#4ade80;">Düşük Risk</p>
                <p class="result-prob" style="color:#22c55e;">{prob_pct:.1f}%</p>
                <p class="result-desc">
                    Model, girilen parametrelerin<br>
                    <strong style="color:#4ade80;">normal profil</strong> ile uyumlu olduğunu öngörüyor.<br>
                    (total UPDRS medyan eşiği: {median_updrs:.1f})
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Olasılık çubuğu
        bar_color = "#ef4444" if prob >= 0.5 else "#22c55e"
        st.markdown(f"""
        <div style="margin-top:1.5rem;">
            <div style="display:flex; justify-content:space-between; font-size:0.7rem; color:#4a6080; margin-bottom:0.4rem;">
                <span>Düşük Risk</span><span>Yüksek Risk</span>
            </div>
            <div class="prob-bar-container">
                <div class="prob-bar-fill" style="width:{prob_pct:.1f}%; background:{bar_color};"></div>
            </div>
            <div style="text-align:center; font-size:0.7rem; color:#4a6080; margin-top:0.4rem;">
                Olasılık: {prob_pct:.2f}% &nbsp;|&nbsp; Eşik: 50%
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Ham değerler
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Model Detayları</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-val">{prob:.4f}</div>
                <div class="metric-lbl">Sigmoid Çıktısı</div>
            </div>
            <div class="metric-card">
                <div class="metric-val">{"1" if prob >= 0.5 else "0"}</div>
                <div class="metric-lbl">Binary Tahmin</div>
            </div>
            <div class="metric-card">
                <div class="metric-val">20</div>
                <div class="metric-lbl">Özellik Sayısı</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Uyarı
        st.markdown("""
        <div class="warning-box">
            ⚠️ <strong>Tıbbi Uyarı:</strong> Bu sistem yalnızca akademik ve eğitim amaçlıdır.
            Herhangi bir tıbbi karar için lisanslı bir nörolog veya hekim tarafından değerlendirme
            yapılması zorunludur. Bu araç teşhis koyamaz.
        </div>
        """, unsafe_allow_html=True)

    else:
        # Boş durum
        st.markdown("""
        <div style="
            border: 1px dashed #1a2c4a;
            border-radius: 16px;
            padding: 4rem 2rem;
            text-align: center;
            color: #2a4060;
        ">
            <div style="font-size:2.5rem; margin-bottom:1rem;">🧬</div>
            <p style="font-family:'Syne',sans-serif; font-size:1rem; color:#3a5070; margin:0;">
                Parametreleri girin ve<br>"Tahmin Et" butonuna basın
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── Model hakkında bilgi ────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Model Hakkında</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card" style="font-size:0.78rem; line-height:1.9; color:#6a8aaa;">
        <strong style="color:#60a5fa;">Mimari</strong><br>
        ANN: 20 → 16 → 1 (ReLU · BCEWithLogitsLoss)<br><br>
        <strong style="color:#60a5fa;">Veri Seti</strong><br>
        parkinsons_updrs.csv — 5.875 kayıt, 22 sütun<br><br>
        <strong style="color:#60a5fa;">Hedef Değişken</strong><br>
        total_UPDRS > medyan (27.58) → Yüksek Risk (1)<br><br>
        <strong style="color:#60a5fa;">Özellikler</strong><br>
        Jitter, Shimmer, NHR, HNR, RPDE, DFA, PPE + demografik bilgiler
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Alt bilgi
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; margin-top:3rem; font-size:0.68rem; color:#2a3a50; letter-spacing:0.15em;">
    YAPAY SİNİR AĞLARI PROJESİ &nbsp;·&nbsp; PYTORCH ANN MODELİ &nbsp;·&nbsp; YALNIZCA EĞİTİM AMAÇLIDIR
</div>
""", unsafe_allow_html=True)