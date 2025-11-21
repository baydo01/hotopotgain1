import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import time
import warnings
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import pytz

# Uyarıları gizle
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# AYARLAR VE SABİTLER
# ---------------------------------------------------------
SHEET_ID = "16zjLeps0t1P26OF3o7XQ-djEKKZtZX6t5lFxLmnsvpE"

st.set_page_config(page_title="Hedge Fund Bot: Pro Edition", layout="wide")

# ---------------------------------------------------------
# GOOGLE SHEETS BAĞLANTISI
# ---------------------------------------------------------
def connect_sheet():
    """Streamlit Secrets üzerinden Google Sheets'e bağlanır."""
    scope = ["https://spreadsheets.google.com/feeds",
             "https://www.googleapis.com/auth/drive"]
    
    try:
        json_key = dict(st.secrets["gcp_service_account"])

        # Private Key düzeltmesi
        if "private_key" in json_key:
            json_key["private_key"] = json_key["private_key"].replace("\\n", "\n")

        creds = ServiceAccountCredentials.from_json_keyfile_dict(json_key, scope)
        client = gspread.authorize(creds)
        sheet = client.open_by_key(SHEET_ID).sheet1
        return sheet
    except Exception as e:
        st.error(f"Google Sheets Bağlantı Hatası: {e}")
        st.stop()

def save_to_google_sheets(df):
    """Dataframe'i Google Sheets'e yazar."""
    try:
        sheet = connect_sheet()
        sheet.clear()
        # Tarih formatlarını stringe çevir (Excel hatasını önlemek için)
        df_export = df.copy()
        df_export = df_export.astype(str) 
        sheet.update([df_export.columns.values.tolist()] + df_export.values.tolist())
    except Exception as e:
        st.error(f"Kaydetme Hatası: {e}")

def load_from_google_sheets():
    """Veriyi okur, sayısal formatları düzeltir ve EKSİK SÜTUNLARI TAMAMLAR."""
    try:
        sheet = connect_sheet()
        data = sheet.get_all_records()
        df = pd.DataFrame(data)
        
        # Beklenen tüm sütunlar
        required_cols = ["Ticker","Durum","Miktar","Son_Islem_Fiyati","Nakit_Bakiye_USD","Baslangic_USD","Kaydedilen_Deger_USD","Son_Islem_Log","Son_Islem_Zamani"]
        
        # Sayfa boşsa standart yapıyı dön
        if df.empty:
            return pd.DataFrame(columns=required_cols)

        # KRİTİK DÜZELTME: Eğer yeni sütunlar eski tabloda yoksa, onları ekle (Çökmemesi için)
        for col in required_cols:
            if col not in df.columns:
                if "USD" in col or "Miktar" in col or "Fiyat" in col:
                    df[col] = 0.0
                else:
                    df[col] = "-"

        # Sayısal dönüşümler
        numeric_cols = ["Miktar", "Son_Islem_Fiyati", "Nakit_Bakiye_USD", "Baslangic_USD", "Kaydedilen_Deger_USD"]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        return df
    except Exception as e:
        st.warning(f"Veri okuma uyarısı (Sıfırla butonunu kullanın): {e}")
        # Hata durumunda boş ama doğru formatta tablo dön
        return pd.DataFrame(columns=["Ticker","Durum","Miktar","Son_Islem_Fiyati","Nakit_Bakiye_USD","Baslangic_USD","Kaydedilen_Deger_USD","Son_Islem_Log","Son_Islem_Zamani"])

# ---------------------------------------------------------
# YARDIMCI FONKSİYONLAR
# ---------------------------------------------------------
if 'logs' not in st.session_state:
    st.session_state.logs = []

def add_log(message):
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.logs.insert(0, f"[{timestamp}] {message}")

def get_current_time_str():
    # Türkiye saati
    tz = pytz.timezone('Europe/Istanbul')
    return datetime.now(tz).strftime("%d-%m %H:%M")

def load_portfolio():
    return load_from_google_sheets()

def save_portfolio(df):
    save_to_google_sheets(df)

# ---------------------------------------------------------
# SİMÜLASYON KURULUMU (SIFIRLAMA)
# ---------------------------------------------------------
def init_simulation(tickers, amount_per_coin=10):
    data = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, ticker in enumerate(tickers):
        status_text.text(f"{ticker} verisi alınıyor...")
        df_price = get_data_cached(ticker)
        
        if df_price is not None and not df_price.empty:
            current_price = float(df_price['close'].iloc[-1])
            coin_amount = float(amount_per_coin) / current_price
            
            data.append({
                "Ticker": ticker,
                "Durum": "COIN", 
                "Miktar": coin_amount,
                "Son_Islem_Fiyati": current_price,
                "Nakit_Bakiye_USD": 0.0,
                "Baslangic_USD": float(amount_per_coin),
                "Kaydedilen_Deger_USD": float(amount_per_coin), # Son kayıttaki değer
                "Son_Islem_Log": "Başlangıç",
                "Son_Islem_Zamani": get_current_time_str()
            })
        else:
            add_log(f"UYARI: {ticker} verisi alınamadı.")
            
        progress_bar.progress((i + 1) / len(tickers))
        
    df = pd.DataFrame(data)
    save_portfolio(df)
    st.session_state.logs = []
    add_log("Portföy SIFIRLANDI ve Google Sheets'e kaydedildi.")
    
    status_text.empty()
    progress_bar.empty()
    return df

# ---------------------------------------------------------
# VERİ ÇEKME & HMM ANALİZİ
# ---------------------------------------------------------
def calculate_custom_score(df):
    if len(df) < 5: return pd.Series(0, index=df.index)
    s1 = np.where(df['close'] > df['close'].shift(10), 1, -1)
    s2 = np.where(df['close'] > df['close'].shift(30), 1, -1)
    vol = df['close'].pct_change().rolling(5).std().fillna(0)
    s3 = np.where(vol < vol.shift(5), 1, -1)
    return s1 + s2 + s3

def prepare_data(df):
    df = df.copy()
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['range'] = (df['high'] - df['low']) / df['close']
    df['custom_score'] = calculate_custom_score(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

@st.cache_data(ttl=600) # 10 dk cache
def get_data_cached(ticker, start_date="2022-01-01"):
    try:
        df = yf.download(ticker, start=start_date, progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        if 'close' not in df.columns and 'adj close' in df.columns: 
            df['close'] = df['adj close']
        df.dropna(inplace=True)
        return df
    except: 
        return None

def get_bulk_signals(tickers):
    results = []
    progress = st.progress(0)
    for i, ticker in enumerate(tickers):
        try:
            df = get_data_cached(ticker)
            if df is None or len(df) < 20:
                results.append({"Ticker": ticker, "Sinyal": "VERI_YOK", "Fiyat": 0.0, "Skor": 0})
                continue
            
            # Veri Hazırlığı
            df = prepare_data(df)
            X = df[['log_ret','range']].values
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)
            
            # HMM
            model = GaussianHMM(n_components=3, covariance_type="full", n_iter=50, random_state=42)
            model.fit(X_s)
            states = model.predict(X_s)
            
            # Sinyal Mantığı
            state_means = pd.DataFrame({'state': states, 'ret': df['log_ret']}).groupby('state')['ret'].mean()
            bull_state = state_means.idxmax()
            bear_state = state_means.idxmin()
            last_state = states[-1]
            
            hmm_signal = 1 if last_state == bull_state else (-1 if last_state == bear_state else 0)
            score = df['custom_score'].iloc[-1]
            score_signal = 1 if score > 0 else (-1 if score < 0 else 0)
            
            final_val = 0.6*hmm_signal + 0.4*score_signal
            decision = "AL" if final_val > 0.2 else ("SAT" if final_val < -0.2 else "BEKLE")
            
            results.append({"Ticker":ticker,"Sinyal":decision,"Fiyat":float(df['close'].iloc[-1]),"Skor":int(score)})
        except:
            results.append({"Ticker":ticker,"Sinyal":"HATA","Fiyat":0.0,"Skor":0})
        progress.progress((i+1)/len(tickers))
    progress.empty()
    return pd.DataFrame(results)

# ---------------------------------------------------------
# BOT MANTIĞI (GÜNCELLEME)
# ---------------------------------------------------------
def run_bot_logic(portfolio_df, signals_df):
    updated_portfolio = portfolio_df.copy()
    time_str = get_current_time_str()
    
    for idx, row in updated_portfolio.iterrows():
        ticker = row['Ticker']
        signal_row = signals_df[signals_df['Ticker']==ticker]
        
        if signal_row.empty: continue
        
        current_price = float(signal_row.iloc[0]['Fiyat'])
        signal = signal_row.iloc[0]['Sinyal']
        
        if current_price <= 0: continue
        
        # --- SATIŞ ---
        if row['Durum']=='COIN' and signal=='SAT':
            cash_obtained = float(row['Miktar']) * current_price
            updated_portfolio.at[idx,'Durum'] = 'CASH'
            updated_portfolio.at[idx,'Nakit_Bakiye_USD'] = cash_obtained
            updated_portfolio.at[idx,'Miktar'] = 0.0
            updated_portfolio.at[idx,'Son_Islem_Fiyati'] = current_price
            updated_portfolio.at[idx,'Son_Islem_Log'] = "SATILDI"
            updated_portfolio.at[idx,'Son_Islem_Zamani'] = time_str
            add_log(f"🔴 {ticker}: SATIŞ yapıldı (${cash_obtained:.2f})")
            
        # --- ALIŞ ---
        elif row['Durum']=='CASH' and signal=='AL':
            cash_available = float(row['Nakit_Bakiye_USD'])
            if cash_available > 0:
                new_amount = cash_available / current_price
                updated_portfolio.at[idx,'Durum'] = 'COIN'
                updated_portfolio.at[idx,'Miktar'] = new_amount
                updated_portfolio.at[idx,'Nakit_Bakiye_USD'] = 0.0
                updated_portfolio.at[idx,'Son_Islem_Fiyati'] = current_price
                updated_portfolio.at[idx,'Son_Islem_Log'] = "ALINDI"
                updated_portfolio.at[idx,'Son_Islem_Zamani'] = time_str
                add_log(f"🟢 {ticker}: ALIŞ yapıldı (${cash_available:.2f})")
        
        else:
            pass

        # Her işlem döngüsünde güncel değeri kaydet
        if updated_portfolio.at[idx, 'Durum'] == 'COIN':
            val = float(updated_portfolio.at[idx, 'Miktar']) * current_price
        else:
            val = float(updated_portfolio.at[idx, 'Nakit_Bakiye_USD'])
        updated_portfolio.at[idx, 'Kaydedilen_Deger_USD'] = val

    save_portfolio(updated_portfolio)
    return updated_portfolio

# ---------------------------------------------------------
# ARAYÜZ (UI)
# ---------------------------------------------------------
st.title("🧠 Hedge Fund Bot: Pro Edition")

# Sidebar
with st.sidebar:
    st.header("⚙️ Ayarlar")
    default_tickers = ["BTC-USD","ETH-USD","SOL-USD","BNB-USD","XRP-USD","AVAX-USD","DOGE-USD","ADA-USD"]
    selected_tickers = st.multiselect("Coinler", default_tickers, default=default_tickers)
    
    if st.button("⚠️ SİMÜLASYONU SIFIRLA (Tabloyu Yenile)"):
        init_simulation(selected_tickers, 10)
        st.success("Portföy sıfırlandı ve yeni sütunlar eklendi.")
        time.sleep(1)
        st.rerun()
        
    st.markdown("---")
    st.info("Bot, 'Analiz Et' butonuna bastığında işlem yapar ve Google Sheets'i günceller.")

# Ana Ekran - Veri Yükleme
pf_df = load_portfolio()

if pf_df.empty:
    st.warning("Veri bulunamadı veya tablo boş. Lütfen soldan 'Simülasyonu Sıfırla' butonuna basın.")
else:
    # 1. ANLIK FİYATLARI ÇEK VE DEĞER HESAPLA
    current_prices = {}
    total_current_value = 0.0
    
    # 'Kaydedilen_Deger_USD' sütunu eksikse 0 kabul et (Hata önleyici)
    if 'Kaydedilen_Deger_USD' in pf_df.columns:
        total_saved_value = pf_df['Kaydedilen_Deger_USD'].sum()
    else:
        total_saved_value = 0.0
        
    total_invested = pf_df['Baslangic_USD'].sum()
    tickers_list = pf_df['Ticker'].tolist()
    
    with st.spinner("Piyasa verileri güncelleniyor..."):
        for t in tickers_list:
            d = get_data_cached(t)
            if d is not None:
                current_prices[t] = float(d['close'].iloc[-1])
            else:
                current_prices[t] = 0.0

    # 2. TABLOYU HAZIRLA
    display_data = []
    
    for idx, row in pf_df.iterrows():
        curr_price = current_prices.get(row['Ticker'], 0.0)
        
        if row['Durum'] == 'COIN':
            asset_val = float(row['Miktar']) * curr_price
        else:
            asset_val = float(row['Nakit_Bakiye_USD'])
        
        total_current_value += asset_val
        
        pnl = asset_val - float(row['Baslangic_USD'])
        pnl_pct = (pnl / float(row['Baslangic_USD'])) * 100 if float(row['Baslangic_USD']) > 0 else 0.0
        
        # Sütunlar eksikse "-" yaz
        son_islem = row.get('Son_Islem_Log', '-')
        son_zaman = row.get('Son_Islem_Zamani', '-')

        display_data.append({
            "Coin": row['Ticker'],
            "Durum": row['Durum'],
            "Fiyat": curr_price,
            "Değer ($)": asset_val,
            "Son İşlem": f"{son_islem} ({son_zaman})",
            "Net Kâr ($)": pnl,
            "Kâr %": pnl_pct
        })

    # 3. METRİKLER
    change_since_last = total_current_value - total_saved_value
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Toplam Portföy", f"${total_current_value:.2f}")
    m2.metric("Son Girişten Beri", f"${change_since_last:+.2f}", delta_color="normal")
    m3.metric("Net Kâr (Genel)", f"${(total_current_value - total_invested):+.2f}")
    
    # 4. BOT BUTONU
    col_btn, col_empty = st.columns([1,3])
    signals_df = None
    with col_btn:
        if st.button("🤖 ANALİZ ET VE İŞLEM YAP (Botu Çalıştır)", type="primary"):
            with st.spinner("HMM Modelleri ve İndikatörler Çalışıyor..."):
                signals_df = get_bulk_signals(tickers_list)
                pf_df = run_bot_logic(pf_df, signals_df)
                st.success("İşlemler tamamlandı ve kaydedildi!")
                time.sleep(1)
                st.rerun()

    # 5. TABLO GÖSTERİMİ
    final_table = pd.DataFrame(display_data)
    
    if signals_df is not None:
        st.write("### 📊 Anlık Sinyaller")
        st.dataframe(signals_df.style.format({"Fiyat": "${:.2f}"}))

    st.write("### 💰 Portföy Detayı")
    st.dataframe(final_table.style.format({
        "Fiyat": "${:.2f}", 
        "Değer ($)": "${:.2f}",
        "Net Kâr ($)": "{:+.2f}", 
        "Kâr %": "{:+.2f}%"
    }))
