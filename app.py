import streamlit as st
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import numpy as np

# --- Sayfa Konfigürasyonu ---
st.set_page_config(
    page_title="KKTC Enflasyon Tahmini",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS for Professional Look ---
st.markdown("""
<style>
    /* Modern card design */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: white;
        margin-bottom: 1rem;
    }
    
    .metric-card-neutral {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: white;
        margin-bottom: 1rem;
    }
    
    .metric-card-success {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: white;
        margin-bottom: 1rem;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        font-weight: 500;
    }
    
    .metric-sublabel {
        font-size: 0.8rem;
        opacity: 0.8;
        margin-top: 0.3rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Responsive typography */
    @media (max-width: 640px) {
        .metric-value {
            font-size: 1.8rem;
        }
        .metric-card {
            padding: 1rem;
        }
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        color: #1f2937;
    }
    
    /* Info badge */
    .info-badge {
        display: inline-block;
        background: #f0f9ff;
        color: #0369a1;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 500;
        margin-left: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# --- VERİ YÜKLEME ---
GITHUB_CSV_URL = "https://raw.githubusercontent.com/RYucel/Tufe_tahmin_v3/main/AllDataSets2.csv"

@st.cache_data(ttl="1h")
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

# --- HEADER ---
st.markdown("# 📈 KKTC Tüketici Fiyat Endeksi")
st.markdown("**12 Aylık Enflasyon Tahmini** <span class='info-badge'>AutoARIMA Model</span>", unsafe_allow_html=True)

# --- Ana Uygulama Akışı ---
with st.spinner('📊 Güncel veriler yükleniyor...'):
    data = load_data_from_github(GITHUB_CSV_URL)

if data is not None and not data.empty:
    turkish_months = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", 
                      "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]

    last_known_date = data['ds'].max()
    last_known_value = data.sort_values('ds')['y'].iloc[-1]
    last_known_date_str = f"{turkish_months[last_known_date.month - 1]} {last_known_date.year}"
    
    # Bir önceki ayın değeri (aylık değişim için)
    prev_value = data.sort_values('ds')['y'].iloc[-2] if len(data) > 1 else last_known_value
    monthly_change_current = ((last_known_value - prev_value) / prev_value) * 100
    
    # Yıllık değişim (12 ay öncesine göre)
    yearly_change = None
    if len(data) >= 13:
        value_12m_ago = data.sort_values('ds')['y'].iloc[-13]
        yearly_change = ((last_known_value - value_12m_ago) / value_12m_ago) * 100

    # --- ÜST KART SATIRI: MEVCUT DURUM ---
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Son Bilinen Endeks</div>
            <div class="metric-value">{last_known_value:,.2f}</div>
            <div class="metric-sublabel">{last_known_date_str}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card-neutral">
            <div class="metric-label">Aylık Değişim</div>
            <div class="metric-value">{monthly_change_current:+.2f}%</div>
            <div class="metric-sublabel">Bir önceki aya göre</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if yearly_change is not None:
            st.markdown(f"""
            <div class="metric-card-success">
                <div class="metric-label">Yıllık Enflasyon</div>
                <div class="metric-value">{yearly_change:+.2f}%</div>
                <div class="metric-sublabel">12 aylık değişim</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card-success">
                <div class="metric-label">Yıllık Enflasyon</div>
                <div class="metric-value">N/A</div>
                <div class="metric-sublabel">Yetersiz veri</div>
            </div>
            """, unsafe_allow_html=True)

    # --- MODEL PERFORMANSI ---
    st.markdown("<div class='section-header'>🎯 Model Performansı (Son 3 Ay Test)</div>", unsafe_allow_html=True)
    
    with st.spinner('Model performansı hesaplanıyor...'):
        metrics = calculate_performance_metrics(data)

    if metrics:
        col1, col2, col3 = st.columns(3)
        col1.metric("MAPE (Ort. % Hata)", f"{metrics['MAPE']:.2f}%", 
                   help="Düşük olması daha iyi. Model tahminlerinin ortalama yüzde sapması.")
        col2.metric("MAE (Ort. Hata)", f"{metrics['MAE']:.2f}", 
                   help="Tahminlerin gerçek değerlerden ortalama sapması (endeks puanı).")
        col3.metric("RMSE (Karesel Hata)", f"{metrics['RMSE']:.2f}", 
                   help="Büyük hataları cezalandıran metrik.")

    # --- TAHMİN MODELİ ---
    st.markdown("<div class='section-header'>🔮 12 Aylık Tahmin</div>", unsafe_allow_html=True)
    
    with st.spinner('Tahminler hesaplanıyor...'):
        model_full_data = StatsForecast(models=[AutoARIMA(season_length=12)], freq='MS')
        model_full_data.fit(data)
        forecast = model_full_data.predict(h=12, level=[95])

    predicted_values = forecast['AutoARIMA'].values
    lower_bound = forecast['AutoARIMA-lo-95'].values
    upper_bound = forecast['AutoARIMA-hi-95'].values

    # Tarih aralığı oluşturma - düzeltilmiş
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

    # --- ÖNE ÇIKAN TAHMİNLER (3 ve 12 ay) ---
    col1, col2 = st.columns(2)
    
    forecast_3m_value = predicted_values[2]
    forecast_3m_change = cumulative_inflation[2]
    forecast_12m_value = predicted_values[11]
    forecast_12m_change = cumulative_inflation[11]
    
    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                    padding: 1.5rem; border-radius: 12px; color: white; text-align: center;">
            <div style="font-size: 0.9rem; font-weight: 500;">3 Ay Sonra Tahmini</div>
            <div style="font-size: 2rem; font-weight: bold; margin: 0.5rem 0;">{forecast_3m_value:,.2f}</div>
            <div style="font-size: 0.85rem; opacity: 0.9;">Kümülatif Enflasyon: {forecast_3m_change:+.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                    padding: 1.5rem; border-radius: 12px; color: #1f2937; text-align: center;">
            <div style="font-size: 0.9rem; font-weight: 500;">12 Ay Sonra Tahmini</div>
            <div style="font-size: 2rem; font-weight: bold; margin: 0.5rem 0;">{forecast_12m_value:,.2f}</div>
            <div style="font-size: 0.85rem; opacity: 0.8;">Kümülatif Enflasyon: {forecast_12m_change:+.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # --- GRAFİK ---
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=results_df['Tarih_ts'], y=results_df['En Düşük Tahmin (%95 Güven)'],
        mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=results_df['Tarih_ts'], y=results_df['En Yüksek Tahmin (%95 Güven)'],
        mode='lines', line=dict(width=0), fill='tonexty',
        fillcolor='rgba(102, 126, 234, 0.2)', name='%95 Güven Aralığı'
    ))
    fig.add_trace(go.Scatter(
        x=data['ds'].tail(36), y=data['y'].tail(36),
        mode='lines+markers', name='Geçmiş Veri', 
        line=dict(color='#667eea', width=3),
        marker=dict(size=6)
    ))
    fig.add_trace(go.Scatter(
        x=results_df['Tarih_ts'], y=results_df['Tahmin Edilen Endeks'],
        mode='lines+markers', name='Tahmin', 
        line=dict(color='#f5576c', width=3, dash='dot'),
        marker=dict(size=8, symbol='diamond')
    ))
    
    fig.update_layout(
        title={'text': 'TÜFE Endeksi: Geçmiş ve Tahmin', 'x': 0.5, 'xanchor': 'center'},
        xaxis_title='Tarih', 
        yaxis_title='Endeks Değeri',
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial, sans-serif", size=12)
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.1)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.1)')
    
    st.plotly_chart(fig, use_container_width=True)

    # --- DETAYLI TABLO ---
    with st.expander("📋 Detaylı Tahmin Tablosu", expanded=False):
        display_df = results_df.copy()
        display_df['Tarih'] = display_df['Tarih_ts'].apply(
            lambda dt: f"{turkish_months[dt.month - 1]} {dt.year}"
        )
        display_df = display_df[['Tarih', 'Tahmin Edilen Endeks', 'Aylık Değişim (%)', 
                                'Son Veriye Göre Kümülatif Enflasyon (%)', 
                                'En Düşük Tahmin (%95 Güven)', 'En Yüksek Tahmin (%95 Güven)']]
        
        st.dataframe(
            display_df.style.format({
                'Tahmin Edilen Endeks': '{:,.2f}',
                'Aylık Değişim (%)': '{:+.2f}%',
                'Son Veriye Göre Kümülatif Enflasyon (%)': '{:+.2f}%',
                'En Düşük Tahmin (%95 Güven)': '{:,.2f}',
                'En Yüksek Tahmin (%95 Güven)': '{:,.2f}',
            }).background_gradient(subset=['Aylık Değişim (%)'], cmap='RdYlGn_r'),
            use_container_width=True, 
            hide_index=True,
            height=450
        )

    # --- FOOTER ---
    st.markdown("---")
    st.caption("📊 Veri Kaynağı: GitHub | Model: AutoARIMA | Son Güncelleme: Her saat başı")

else:
    st.error("⚠️ Veri yüklenemedi. Lütfen daha sonra tekrar deneyin.")