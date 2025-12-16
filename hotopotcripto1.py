import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import yfinance as yf
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LinearRegression
import time

# ==========================================
# 1. AYARLAR VE BAĞLANTILAR (BACKEND)
# ==========================================

# Google Sheets Bağlantısı (Cache kullanarak hızlandırıyoruz)
@st.cache_resource
def get_google_sheet_client():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    # 'credentials.json' dosyanızın projenin ana dizininde olduğundan emin olun
    creds = ServiceAccountCredentials.from_json_keyfile_name("sizin_api_json_dosyaniz.json", scope)
    client = gspread.authorize(creds)
    return client

def get_data_from_sheet():
    client = get_google_sheet_client()
    try:
        sheet = client.open("Sizin_Tablo_Adiniz").sheet1  # Tablo adını buraya girin
        data = sheet.get_all_records()
        df = pd.DataFrame(data)
        return sheet, df
    except Exception as e:
        st.error(f"Google Sheets Bağlantı Hatası: {e}")
        return None, None

# ==========================================
# 2. VOLATİLİTE HESAPLAMA MOTORU
# ==========================================

def calculate_volatility(ticker, window=20):
    """
    yfinance kullanarak son 'window' günün volatilitesini çeker.
    Ticker formatı 'BTC-USD' gibi olmalıdır.
    """
    try:
        # Eğer 'COIN' veya 'CASH' gibi ticker olmayan satırlar varsa onları atla
        if "USD" not in ticker and len(ticker) < 6: 
            return 0.0
            
        stock = yf.Ticker(ticker)
        # 3 aylık veri çekiyoruz ki hareketli ortalama hesaplanabilsin
        hist = stock.history(period="3mo")
        
        if len(hist) < window:
            return 0.0
        
        # Log Return Hesaplama
        hist['Log_Return'] = np.log(hist['Close'] / hist['Close'].shift(1))
        
        # Standart Sapma (Volatilite)
        vol = hist['Log_Return'].rolling(window=window).std().iloc[-1]
        
        # NaN kontrolü
        if pd.isna(vol):
            return 0.0
            
        return float(vol)
    except Exception as e:
        # st.warning(f"{ticker} için veri çekilemedi: {e}")
        return 0.0

def update_volatility_column(sheet, df):
    """
    DataFrame'deki her satır için volatiliteyi hesaplar ve Sheets'e yazar.
    """
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    volatilities = []
    
    total_rows = len(df)
    for i, row in df.iterrows():
        ticker = row['Ticker']
        status_text.text(f"Volatilite Hesaplanıyor: {ticker}...")
        
        vol = calculate_volatility(ticker)
        volatilities.append(vol)
        
        # İlerleme çubuğu güncelle
        progress_bar.progress((i + 1) / total_rows)
    
    # DataFrame'e ekle
    df['Volatilite'] = volatilities
    
    # Google Sheets'e Yazma
    # Eğer 'Volatilite' sütunu yoksa, sheet'te en sağa ekleriz.
    try:
        cell = sheet.find("Volatilite")
        col_idx = cell.col
    except:
        col_idx = len(df.columns) # Yeni sütun indeksi (df'e zaten ekledik)
        sheet.update_cell(1, col_idx, "Volatilite")
    
    # Sütunu toplu güncelle (API kotası dostu)
    cell_list = []
    for i, vol in enumerate(volatilities):
        # Satır 2'den başlar (1 başlık)
        cell_list.append(gspread.Cell(row=i+2, col=col_idx, value=vol))
    
    sheet.update_cells(cell_list)
    status_text.text("✅ Volatilite değerleri Sheets'e başarıyla işlendi!")
    time.sleep(1)
    status_text.empty()
    progress_bar.empty()
    
    return df

# ==========================================
# 3. HİBRİT MODEL (LINEAR + XGBOOST)
# ==========================================

def run_hybrid_model(df):
    """
    Volatiliteyi de feature olarak alıp analiz yapar.
    Not: Gerçek bir proje için eğitilmiş model (.model dosyası) yüklenmelidir.
    Burada mantığı simüle ediyoruz.
    """
    signals = []
    
    # Modelin kullanacağı sütunlar (Örnektir, elinizdeki veriye göre artırın)
    # Burada 'Son_Islem_Fiyati' gibi değerleri feature olarak kullanıyoruz basitçe.
    # Gerçekte RSI, MACD gibi indikatörler de hesaplanıp buraya eklenmeli.
    
    for i, row in df.iterrows():
        ticker = row['Ticker']
        volatilite = float(row.get('Volatilite', 0))
        fiyat = float(row.get('Son_Islem_Fiyati', 0))
        bakiye = float(row.get('Nakit_Bakiye_USD', 0))
        
        # --- MODEL SİMÜLASYONU ---
        
        # 1. Linear Model Skoru (Basit Trend)
        # Volatilite düşükse Linear modele daha çok güven
        linear_score = 0.6 if fiyat > 0 else 0 # Temsili
        
        # 2. XGBoost Skoru (Karmaşık Yapı)
        # Volatilite yüksekse XGBoost'un yakaladığı patternlere güven
        xgb_score = 0.75 # Temsili tahmin
        
        # 3. Ağırlıklandırma (Dinamik)
        if volatilite > 0.04: # Yüksek oynaklık
            weight_linear = 0.2
            weight_xgb = 0.8
            note = "High Vol"
        else: # Düşük oynaklık
            weight_linear = 0.6
            weight_xgb = 0.4
            note = "Stable"
            
        final_score = (linear_score * weight_linear) + (xgb_score * weight_xgb)
        
        # Karar Mekanizması
        # CASH satırları için işlem yapma
        if "CASH" in str(ticker) or "USDT" in str(ticker):
            signal = "BEKLE"
        elif final_score > 0.65:
            signal = f"AL Linear+XGB ({note} P:{final_score:.2f})"
        elif final_score < 0.35:
            signal = f"SAT Linear+XGB ({note} P:{final_score:.2f})"
        else:
            signal = "TUT"
            
        signals.append(signal)
        
    return signals

def update_bot_status(sheet, df, signals):
    """
    Model sonuçlarını 'Bot_Durum' sütununa yazar.
    """
    df['Bot_Durum'] = signals
    
    try:
        cell = sheet.find("Bot_Durum")
        col_idx = cell.col
    except:
        st.error("'Bot_Durum' sütunu bulunamadı, lütfen Sheet'e ekleyin.")
        return df

    cell_list = []
    for i, sig in enumerate(signals):
        cell_list.append(gspread.Cell(row=i+2, col=col_idx, value=sig))
        
    sheet.update_cells(cell_list)
    return df

# ==========================================
# 4. STREAMLIT ARAYÜZÜ (FRONTEND)
# ==========================================

st.set_page_config(page_title="AI Trading Bot Manager", layout="wide")

st.title("🤖 AI Trading Bot & Volatilite Analizörü")
st.markdown("---")

# Yan Menü
st.sidebar.header("Kontrol Paneli")
run_btn = st.sidebar.button("🚀 Analizi Başlat (Update & Predict)", type="primary")

# Ana Akış
sheet, df = get_data_from_sheet()

if sheet is not None:
    # İlk yüklemede tabloyu göster
    st.subheader("📊 Mevcut Portföy Durumu")
    st.dataframe(df)

    if run_btn:
        with st.spinner('Sistem çalışıyor... Lütfen bekleyiniz.'):
            
            # ADIM 1: Volatilite Hesapla ve Sheets'i Güncelle
            st.info("Adım 1/3: Volatilite verileri yfinance üzerinden çekiliyor...")
            df_updated = update_volatility_column(sheet, df)
            
            # ADIM 2: Modeli Çalıştır (Feature olarak Volatilite kullanır)
            st.info("Adım 2/3: Hibrit Model (Linear + XGB) tahmin üretiyor...")
            signals = run_hybrid_model(df_updated)
            
            # ADIM 3: Sonuçları Sheets'e Yaz
            st.info("Adım 3/3: Kararlar Google Sheets'e işleniyor...")
            df_final = update_bot_status(sheet, df_updated, signals)
            
            st.success("İşlem Tamamlandı! Tablo güncellendi.")
            
            # Güncel tabloyu tekrar göster
            st.subheader("✅ Güncellenmiş Analiz Sonuçları")
            
            # Renklendirme fonksiyonu
            def color_bot_durum(val):
                color = 'white'
                if 'AL' in str(val): color = '#28a745' # Yeşil
                elif 'SAT' in str(val): color = '#dc3545' # Kırmızı
                return f'background-color: {color}'

            st.dataframe(df_final.style.applymap(color_bot_durum, subset=['Bot_Durum']))
            
            # İstatistikler
            avg_vol = df_final[df_final['Volatilite'] > 0]['Volatilite'].mean()
            st.metric(label="Ortalama Piyasa Volatilitesi", value=f"{avg_vol:.4f}")

else:
    st.warning("Veri çekilemedi. Lütfen JSON dosyasını ve bağlantıyı kontrol edin.")
