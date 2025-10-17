# app.py

import streamlit as st
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import numpy as np

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
GITHUB_CSV_URL = "https://raw.githubusercontent.com/RYucel/Tufe_tahmin_v3/main/AllDataSets2.csv"

@st.cache_data(ttl="1h") # Veriyi 1 saatliğine önbellekte tut
def load_data_from_github(url):
    try:
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip()
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        df_cpi = df[['Date', 'KKTCGenel']].copy().dropna()
        df_cpi = df_cpi.rename(columns={'Date': 'ds', 'KKTCGenel': 'y'})
        df_cpi['unique_id'] = 'KKTCGenel'
        return df_cpi
    except Exception as e:
        st.error(f"Veri yüklenirken bir hata oluştu: {e}")
        return None

# --- Modelin Geçmiş Performansını Son 3 Ay İçin Hesapla ---
@st.cache_data(ttl="1h")
def calculate_performance_metrics(data):
    if len(data) < 24: return None
    train_df = data.iloc[:-3]
    test_df = data.iloc[-3:]
    model = StatsForecast(models=[AutoARIMA(season_length=12)], freq='MS')
    model.fit(train_df)
    forecast = model.predict(h=3)
    y_true = test_df['y'].values
    y_pred = forecast['AutoARIMA'].values
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {'MAPE': mape, 'MAE': mae, 'RMSE': rmse}

# --- Ana Uygulama Akışı ---
with st.spinner('Güncel veriler GitHub üzerinden yükleniyor...'):
    data = load_data_from_github(GITHUB_CSV_URL)

if data is not None and not data.empty:
    # Türkçe ay isimleri listesi (locale bağımlılığını ortadan kaldırır)
    turkish_months = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]

    last_known_date = data['ds'].max()
    last_known_value = data.sort_values('ds')['y'].iloc[-1]
    
    # Son bilinen tarihi Türkçe formatla
    last_known_date_str = f"{turkish_months[last_known_date.month - 1]} {last_known_date.year}"

    st.subheader("Mevcut Durum")
    st.metric(label=f"Son Bilinen Endeks Değeri ({last_known_date_str})", value=f"{last_known_value:,.2f}")

    # --- MODEL PERFORMANSI (SON 3 AY) ---
    st.subheader("Modelin Kısa Dönem Geçmiş Performansı (Son 3 Ay)")
    with st.spinner('Modelin yakın geçmişteki performansı test ediliyor...'):
        metrics = calculate_performance_metrics(data)

    if metrics:
        col1, col2, col3 = st.columns(3)
        col1.metric("Ortalama % Hata (MAPE)", f"{metrics['MAPE']:.2f}%", help="Modelin son 3 aydaki tahminlerinin ortalama olarak gerçek değerlerden yüzde kaç saptığını gösterir. Düşük olması daha iyidir.")
        col2.metric("Ortalama Hata (MAE)", f"{metrics['MAE']:.2f}", help="Modelin tahminlerinin ortalama olarak kaç endeks puanı saptığını gösterir.")
        col3.metric("Karesel Hata (RMSE)", f"{metrics['RMSE']:.2f}", help="Büyük hataları daha fazla cezalandıran bir hata metriğidir. MAE'ye yakın olması tutarlı tahminler anlamına gelir.")

    # --- TAHMİN MODELİNİ ÇALIŞTIR ---
    st.subheader("Gelecek 12 Aylık Tahmin Sonuçları")
    with st.spinner('AutoARIMA modeli ile 12 aylık tahmin ve güven aralıkları hesaplanıyor...'):
        model_full_data = StatsForecast(models=[AutoARIMA(season_length=12)], freq='MS')
        model_full_data.fit(data)
        forecast = model_full_data.predict(h=12, level=[95])

    # --- TAHMİN SONUÇLARINI İŞLE ---
    predicted_values = forecast['AutoARIMA'].values
    lower_bound = forecast['AutoARIMA-lo-95'].values
    upper_bound = forecast['AutoARIMA-hi-95'].values

    # --- DÜZELTİLMİŞ KOD: Gelecek 12 ay için daha sağlam tarih aralığı oluşturma ---
    start_forecast_date = last_known_date + pd.offsets.MonthBegin(1)
    future_dates = pd.date_range(start=start_forecast_date, periods=12, freq='MS')
    results_df = pd.DataFrame({
        'Tarih_ts': future_dates,
        'Tahmin Edilen Endeks': predicted_values,
        'En Düşük Tahmin (%95 Güven)': lower_bound,
        'En Yüksek Tahmin (%95 Güven)': upper_bound
    })

    full_series = pd.concat([pd.Series([last_known_value]), pd.Series(predicted_values)])
    monthly_change = (full_series.pct_change() * 100).iloc[1:].values
    results_df['Aylık Değişim (%)'] = monthly_change
    cumulative_inflation = ((predicted_values / last_known_value) - 1) * 100
    results_df['Son Veriye Göre Kümülatif Enflasyon (%)'] = cumulative_inflation

    # --- SONUÇLARI GÖSTER (TABLO) ---
    st.markdown("#### Tahmin Tablosu")
    display_df = results_df.copy()
    
    # --- DÜZELTİLMİŞ KOD: Manuel Türkçe tarih formatlama ---
    display_df['Tarih'] = display_df['Tarih_ts'].apply(lambda dt: f"{turkish_months[dt.month - 1]} {dt.year}")
    
    display_df = display_df[['Tarih', 'Tahmin Edilen Endeks', 'Aylık Değişim (%)', 'Son Veriye Göre Kümülatif Enflasyon (%)', 'En Düşük Tahmin (%95 Güven)', 'En Yüksek Tahmin (%95 Güven)']]
    st.dataframe(display_df.style.format({
        'Tahmin Edilen Endeks': '{:,.2f}',
        'Aylık Değişim (%)': '{:,.2f}%',
        'Son Veriye Göre Kümülatif Enflasyon (%)': '{:,.2f}%',
        'En Düşük Tahmin (%95 Güven)': '{:,.2f}',
        'En Yüksek Tahmin (%95 Güven)': '{:,.2f}',
    }), use_container_width=True, hide_index=True)

    # --- SONUÇLARI GÖSTER (GRAFİK) ---
    st.markdown("#### Tahmin Grafiği")
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=results_df['Tarih_ts'], y=results_df['En Düşük Tahmin (%95 Güven)'],
        mode='lines', line=dict(width=0), showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=results_df['Tarih_ts'], y=results_df['En Yüksek Tahmin (%95 Güven)'],
        mode='lines', line=dict(width=0), fill='tonexty',
        fillcolor='rgba(220, 53, 69, 0.2)', name='95% Güven Aralığı', showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=data['ds'].tail(36), y=data['y'].tail(36),
        mode='lines+markers', name='Geçmiş Gerçekleşen Endeks', line=dict(color='royalblue')
    ))
    fig.add_trace(go.Scatter(
        x=results_df['Tarih_ts'], y=results_df['Tahmin Edilen Endeks'],
        mode='lines+markers', name='12 Aylık Tahmin (Orta Senaryo)', line=dict(color='crimson', dash='dot')
    ))
    fig.update_layout(
        title_text='Geçmiş ve Tahmin Edilen TÜFE Endeksi Değerleri',
        xaxis_title='Tarih', yaxis_title='Endeks Değeri (TÜFE)',
        legend_title_text='Veri', hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("Veri yüklenemediği için uygulama başlatılamıyor.")