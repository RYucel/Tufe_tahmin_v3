# app.py

import streamlit as st
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
import plotly.graph_objects as go

# --- Sayfa Konfigürasyonu (Başlık, İkon vb.) ---
st.set_page_config(
    page_title="KKTC Enflasyon Tahmini",
    page_icon="📈",
    layout="wide"
)

# --- UYGULAMA BAŞLIĞI VE AÇIKLAMASI ---
st.title("📈 KKTC Tüketici Fiyat Endeksi (TÜFE) Tahmini")
st.markdown("""
Bu uygulama, halka açık verileri kullanarak Kuzey Kıbrıs Türk Cumhuriyeti için gelecek 12 aylık enflasyon tahminini yapar.
Tahminler, istatistiksel bir model olan **AutoARIMA** kullanılarak otomatik olarak üretilmektedir. Her yenilemede veriler güncellenir.
""")

# --- VERİ YÜKLEME ---
# ÖNEMLİ: Bu URL'yi kendi GitHub'daki AllDataSets.csv dosyanızın "Raw" URL'si ile değiştirin!
GITHUB_CSV_URL = "https://raw.githubusercontent.com/KULLANICI_ADINIZ/REPO_ADINIZ/main/AllDataSets.csv"

@st.cache_data(ttl="1h") # Veriyi 1 saatliğine önbellekte tut, sürekli GitHub'ı yorma
def load_data_from_github(url):
    """
    Veriyi GitHub'dan okur, temizler ve tahmin için hazırlar.
    """
    try:
        df = pd.read_csv(url)
        # Sütun adlarındaki olası boşlukları temizle
        df.columns = df.columns.str.strip()
        # Tarih sütununu işle
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        # Sadece gerekli sütunları al ve eksik verileri temizle
        df_cpi = df[['Date', 'KKTCGenel']].copy().dropna()
        df_cpi = df_cpi.rename(columns={'Date': 'ds', 'KKTCGenel': 'y'})
        # Model için gerekli olan 'unique_id' sütununu ekle
        df_cpi['unique_id'] = 'KKTCGenel'
        return df_cpi
    except Exception as e:
        st.error(f"Veri yüklenirken bir hata oluştu: {e}")
        st.error("Lütfen GitHub URL'sinin doğru olduğundan ve dosyanın public (herkese açık) olduğundan emin olun.")
        return None

# Veriyi yükle ve yüklenirken kullanıcıya bilgi ver
with st.spinner('Güncel veriler GitHub üzerinden yükleniyor...'):
    data = load_data_from_github(GITHUB_CSV_URL)

if data is not None and not data.empty:
    # --- SON BİLİNEN VERİYİ GÖSTER ---
    last_known_date = data['ds'].max().strftime('%B %Y')
    last_known_value = data.sort_values('ds')['y'].iloc[-1]

    st.subheader("Mevcut Durum")
    st.metric(label=f"Son Bilinen Endeks Değeri ({last_known_date})", value=f"{last_known_value:,.2f}")

    # --- TAHMİN MODELİNİ ÇALIŞTIR ---
    st.subheader("Gelecek 12 Aylık Tahmin Sonuçları")
    with st.spinner('AutoARIMA modeli ile 12 aylık tahmin hesaplanıyor...'):
        # Modeli tanımla
        model = StatsForecast(
            models=[AutoARIMA(season_length=12)],
            freq='MS' # Aylık Frekans
        )
        # Modeli tüm veriyle eğit
        model.fit(data)
        # 12 ay ileriye tahmin yap
        forecast = model.predict(h=12)

    # --- TAHMİN SONUÇLARINI İŞLE ---
    # Tahmin edilen değerleri ve tarihleri al
    predicted_values = forecast['AutoARIMA'].values
    future_dates = pd.date_range(start=data['ds'].max() + pd.DateOffset(months=1), periods=12, freq='MS')

    # Sonuçları bir DataFrame'de topla
    results_df = pd.DataFrame({
        'Tarih': future_dates,
        'Tahmin Edilen Endeks': predicted_values
    })

    # Aylık ve Kümülatif Enflasyon Hesaplamaları
    # İlk ayın değişimini hesaplamak için son bilinen gerçek değeri seriye ekle
    full_series = pd.concat([
        pd.Series([last_known_value]),
        pd.Series(predicted_values)
    ])

    # Aylık Değişim (%)
    monthly_change = (full_series.pct_change() * 100).iloc[1:].values
    results_df['Aylık Değişim (%)'] = monthly_change

    # Kümülatif Enflasyon (%)
    cumulative_inflation = ((predicted_values / last_known_value) - 1) * 100
    results_df['Son Veriye Göre Kümülatif Enflasyon (%)'] = cumulative_inflation


    # --- SONUÇLARI GÖSTER (TABLO) ---
    st.markdown("#### Tahmin Tablosu")
    st.dataframe(results_df.style.format({
        'Tahmin Edilen Endeks': '{:,.2f}',
        'Aylık Değişim (%)': '{:,.2f}%',
        'Son Veriye Göre Kümülatif Enflasyon (%)': '{:,.2f}%'
    }), use_container_width=True)

    # --- SONUÇLARI GÖSTER (GRAFİK) ---
    st.markdown("#### Tahmin Grafiği")
    fig = go.Figure()

    # Geçmiş verileri grafiğe ekle (son 36 ay)
    fig.add_trace(go.Scatter(
        x=data['ds'].tail(36),
        y=data['y'].tail(36),
        mode='lines+markers',
        name='Geçmiş Gerçekleşen Endeks Değerleri',
        line=dict(color='royalblue')
    ))

    # Tahmin verilerini grafiğe ekle
    fig.add_trace(go.Scatter(
        x=results_df['Tarih'],
        y=results_df['Tahmin Edilen Endeks'],
        mode='lines+markers',
        name='12 Aylık Tahmin',
        line=dict(color='crimson', dash='dot')
    ))

    # Grafiği güzelleştir
    fig.update_layout(
        title_text='Geçmiş ve Tahmin Edilen TÜFE Endeksi Değerleri',
        xaxis_title='Tarih',
        yaxis_title='Endeks Değeri (TÜFE)',
        legend_title_text='Veri',
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("Veri yüklenemediği için uygulama başlatılamıyor.")