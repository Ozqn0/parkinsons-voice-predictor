# 🧠 Parkinson Tahmin Sistemi — Streamlit Uygulaması

## Kurulum ve Çalıştırma

### 1. Gerekli dosyaları aynı klasöre koyun
```
proje/
├── app.py
├── best_medicaldataset_pytorch_model_expanded.pth
├── parkinsons_updrs.csv
└── requirements.txt
```

### 2. Bağımlılıkları yükleyin
```bash
pip install -r requirements.txt
```

### 3. Uygulamayı başlatın
```bash
streamlit run app.py
```

Tarayıcınızda otomatik olarak `http://localhost:8501` açılacaktır.

---

## requirements.txt içeriği
```
streamlit
torch
pandas
numpy
scikit-learn
```

---

## Model Bilgisi
- **Mimari:** ANN — 20 giriş → 16 gizli → 1 çıkış (ReLU aktivasyon)
- **Veri Seti:** parkinsons_updrs.csv (5.875 kayıt)
- **Hedef:** total_UPDRS > medyan (≈27.58) → Yüksek Risk
- **Loss:** BCEWithLogitsLoss

## ⚠️ Uyarı
Bu uygulama yalnızca akademik/eğitim amaçlıdır. Tıbbi teşhis için kullanılamaz.
