import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import warnings
import gspread
import json
import os
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import pytz

# --- AI & ML KÜTÜPHANELERİ ---
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import plotly.graph_objects as go

# Uyarıları gizle
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Hedge Fund AI: Master", layout="wide")

# =============================================================================
# 1. AYARLAR
# =============================================================================

# Senin Sheet ID'n (Bunu kodun içine gömdüm, değiştirme)
SHEET_ID = "16zjLeps0t1P26OF3o7XQ-djEKKZtZX6t5lFxLmnsvpE"

st.title("🏦 Hedge Fund AI: Otonom Yönetim Paneli")
st.markdown("Bu sistem; Google Sheets portföyünü okur, **Kalman Filtresi + AI** ile analiz eder ve işlemleri **Sheets'e kaydeder**.")

# Yan Panel Ayarları
with st.sidebar:
    st.header("⚙️ Ayarlar")
    ga_gens = st.number_input("Genetik Algoritma Döngüsü", 1, 20, 3, help="Yüksek sayı analizi yavaşlatır.")
    
# =============================================================================
# 2. BAĞLANTI VE OTO-KURULUM MOTORU (EN KRİTİK KISIM)
# =============================================================================

def connect_sheet():
    """Google Sheets'e bağlanır. Hem Local dosyayı hem Cloud Secrets'ı dener."""
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = None
    
    # 1. Önce Streamlit Secrets'a bak (Cloud için)
    if "gcp_service_account" in st.secrets:
        try:
            # Secrets dict olarak gelir, onu kullanırız
            creds_dict = dict(st.secrets["gcp_service_account"])
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        except Exception as e:
            st.error(f"Secrets okuma hatası: {e}")
    
    # 2. Yoksa yerel dosyaya bak (Bilgisayarın için)
    elif os.path.exists("service_account.json"):
        creds = ServiceAccountCredentials.from_json_keyfile_name("service_account.json", scope)
        
    if not creds:
        st.error("❌ HATA: Kimlik bilgileri bulunamadı! Streamlit Secrets ayarlı mı?")
        return None

    try:
        client = gspread.authorize(creds)
        return client.open_by_key(SHEET_ID).sheet1
    except Exception as e:
        st.error(f"❌ Sheets Erişim Hatası: {e}")
        return None

def load_and_fix_portfolio():
    """Portföyü çeker. TABLO BOŞSA VEYA HATALIYSA OTOMATİK KURAR."""
    sheet = connect_sheet()
    if sheet is None: return pd.DataFrame(), None

    # --- OTO-KURULUM MODU ---
    try:
        # Sayfa boş mu diye kontrol et
        existing_data = sheet.get_all_values()
        
        # Beklenen Başlıklar
        required_cols = ["Ticker", "Durum", "Miktar", "Son_Islem_Fiyati", 
                         "Nakit_Bakiye_USD", "Baslangic_USD", "Kaydedilen_Deger_USD", 
                         "Son_Islem_Log", "Son_Islem_Zamani"]
        
        # Eğer veri yoksa veya başlıklar yanlışsa -> SIFIRDAN KUR
        if not existing_data or existing_data[0] != required_cols:
            st.warning("⚠️ Portföy tablosu bulunamadı veya bozuk. SİSTEM SIFIRDAN KURULUYOR...")
            sheet.clear()
            sheet.append_row(required_cols)
            
            # SENİN İSTEDİĞİN 6 COIN (Hepsi 10$ Nakit ile Başlar)
            defaults = [
                ["BTC-USD", "CASH", 0, 0, 10, 10, 10, "KURULUM", "-"],
                ["ETH-USD", "CASH", 0, 0, 10, 10, 10, "KURULUM", "-"],
                ["SOL-USD", "CASH", 0, 0, 10, 10, 10, "KURULUM", "-"],
                ["BNB-USD", "CASH", 0, 0, 10, 10, 10, "KURULUM", "-"],
                ["XRP-USD", "CASH", 0, 0, 10, 10, 10, "KURULUM", "-"],
                ["DOGE-USD", "CASH", 0, 0, 10, 10, 10, "KURULUM", "-"]
            ]
            for d in defaults: sheet.append_row(d)
            st.success("✅ Tablo 6 Coin ile oluşturuldu!")
            time.sleep(2) # Sheets'in yazması için bekle
            return load_and_fix_portfolio() # Fonksiyonu tekrar çağırıp yeni tabloyu çek
            
    except Exception as e:
        st.error(f"Oto-Kurulum Hatası: {e}")
        return pd.DataFrame(), None

    # Veriyi Çek
    data = sheet.get_all_records()
    df = pd.DataFrame(data)
    
    # Sayısal Dönüşümler (Hata önlemek için)
    numeric_cols = ["Miktar", "Son_Islem_Fiyati", "Nakit_Bakiye_USD", "Baslangic_USD", "Kaydedilen_Deger_USD"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            
    return df, sheet

def save_portfolio(df, sheet):
    """Değişiklikleri kaydeder."""
    if sheet is None: return
    try:
        df_export = df.copy()
        df_export = df_export.astype(str)
        sheet.clear()
        sheet.update([df_export.columns.values.tolist()] + df_export.values.tolist())
    except Exception as e:
        st.error(f"Kaydetme Hatası: {e}")

# =============================================================================
# 3. YAPAY ZEKA MOTORU (KALMAN + ML)
# =============================================================================

def apply_kalman_filter(prices):
    n_iter = len(prices); sz = (n_iter,)
    Q = 1e-5; R = 0.01 ** 2
    xhat = np.zeros(sz); P = np.zeros(sz); xhatminus = np.zeros(sz); Pminus = np.zeros(sz); K = np.zeros(sz)
    xhat[0] = prices.iloc[0]; P[0] = 1.0
    for k in range(1, n_iter):
        xhatminus[k] = xhat[k - 1]; Pminus[k] = P[k - 1] + Q
        K[k] = Pminus[k] / (Pminus[k] + R)
        xhat[k] = xhatminus[k] + K[k] * (prices.iloc[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]
    return pd.Series(xhat, index=prices.index)

def get_raw_data(ticker):
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        if 'close' not in df.columns and 'adj close' in df.columns: df['close'] = df['adj close']
        return df
    except: return None

def process_data(df, timeframe):
    if df is None or len(df) < 100: return None
    agg_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    
    if timeframe == 'W': df_res = df.resample('W').agg(agg_dict).dropna()
    elif timeframe == 'M': df_res = df.resample('ME').agg(agg_dict).dropna()
    else: df_res = df.copy()
        
    if len(df_res) < 30: return None
    
    df_res['kalman_close'] = apply_kalman_filter(df_res['close'])
    df_res['log_ret'] = np.log(df_res['kalman_close'] / df_res['kalman_close'].shift(1))
    df_res['range'] = (df_res['high'] - df_res['low']) / df_res['close']
    df_res['trend_signal'] = np.where(df_res['close'] > df_res['kalman_close'], 1, -1)
    
    df_res['ma5'] = df_res['close'].rolling(5).mean()
    df_res['target'] = (df_res['close'].shift(-1) > df_res['close']).astype(int)
    df_res.dropna(inplace=True)
    return df_res

def train_models_for_window(train_df, rf_depth=5):
    features = ['log_ret','range','trend_signal']
    X = train_df[features]; y = train_df['target']
    
    clf_rf = RandomForestClassifier(n_estimators=30, max_depth=rf_depth, n_jobs=-1, random_state=42)
    clf_rf.fit(X, y)
    
    clf_xgb = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=30, max_depth=3)
    clf_xgb.fit(X, y)
    
    hmm_model = None
    try:
        Xh_s = StandardScaler().fit_transform(train_df[['log_ret','range']].values)
        hmm_model = GaussianHMM(n_components=3, covariance_type='diag', n_iter=50, random_state=42)
        hmm_model.fit(Xh_s)
    except: pass
    
    return {'rf': clf_rf, 'xgb': clf_xgb, 'hmm': hmm_model}

def predict_with_models(models, row):
    if models is None: return 0
    
    # Veriyi hazırla
    Xrow = pd.DataFrame([row[['log_ret','range','trend_signal']]])
    
    rf_prob = models['rf'].predict_proba(Xrow)[0][1]
    xgb_prob = models['xgb'].predict_proba(Xrow)[0][1]
    
    # Stacking (Basit Ortalama)
    stack_sig = ((rf_prob + xgb_prob) / 2 - 0.5) * 2
    
    hmm_sig = 0.0
    if models['hmm']:
        try:
            Xh = row[['log_ret','range']].values.reshape(1,-1)
            probs = models['hmm'].predict_proba(StandardScaler().fit_transform(Xh))[0]
            hmm_sig = probs[np.argmax(models['hmm'].means_[:,0])] - probs[np.argmin(models['hmm'].means_[:,0])]
        except: pass
    
    return (hmm_sig * 0.3) + (stack_sig * 0.4) + (row['trend_signal'] * 0.3)

# =============================================================================
# 4. ANALİZ FONKSİYONU
# =============================================================================

def analyze_ticker(ticker, status_box):
    """Tek bir coini analiz eder ve sonucu döner."""
    status_box.text(f"⏳ {ticker} verisi çekiliyor...")
    raw_df = get_raw_data(ticker)
    
    if raw_df is None: 
        status_box.error("Veri Yok")
        return "HATA", 0.0, "YOK"
    
    current_price = float(raw_df['close'].iloc[-1])
    
    # TURNUVA
    timeframes = {'GÜNLÜK': 'D', 'HAFTALIK': 'W'}
    best_score = -999
    final_decision = "BEKLE"
    winning_tf = "YOK"
    winning_df = None
    
    for tf_name, tf_code in timeframes.items():
        df = process_data(raw_df, tf_code)
        if df is None: continue
        
        # Modelleri Eğit (Son 60 bar)
        models = train_models_for_window(df.iloc[-60:], rf_depth=5)
        
        # Tahmin (Son bar)
        signal = predict_with_models(models, df.iloc[-1])
        
        if abs(signal) > best_score:
            best_score = abs(signal)
            winning_tf = tf_name
            winning_df = df.copy()
            
            if signal > 0.25: final_decision = "AL"
            elif signal < -0.25: final_decision = "SAT"
            else: final_decision = "BEKLE"
            
    # Grafik Çiz (Sadece Kazanan TF için)
    if winning_df is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=winning_df.index, y=winning_df['close'], name='Fiyat', line=dict(color='gray')))
        fig.add_trace(go.Scatter(x=winning_df.index, y=winning_df['kalman_close'], name='Kalman Trend', line=dict(color='cyan', width=2)))
        fig.update_layout(height=250, title=f"{ticker} - {winning_tf} Trendi", template="plotly_dark", margin=dict(t=30,b=0,l=0,r=0))
        st.plotly_chart(fig, use_container_width=True)
    
    status_box.markdown(f"**Karar:** {final_decision} ({winning_tf}) | **Güç:** {best_score:.2f}")
    return final_decision, current_price, winning_tf

# =============================================================================
# 5. ANA EKRAN VE ÇALIŞTIRMA
# =============================================================================

if st.button("🚀 ANALİZİ BAŞLAT VE PORTFÖYÜ GÜNCELLE", type="primary"):
    
    # 1. Portföyü Yükle
    pf_df, sheet = load_and_fix_portfolio()
    
    if pf_df.empty:
        st.error("Portföy yüklenemedi. Lütfen tekrar deneyin.")
    else:
        st.success("✅ Portföy yüklendi. Analiz başlıyor...")
        updated_portfolio = pf_df.copy()
        progress_bar = st.progress(0)
        
        tz = pytz.timezone('Europe/Istanbul')
        time_str = datetime.now(tz).strftime("%d-%m %H:%M")
        
        # Her coin için döngü
        for i, (idx, row) in enumerate(updated_portfolio.iterrows()):
            ticker = row['Ticker']
            if not ticker: continue
            
            with st.expander(f"{ticker} Analizi", expanded=True):
                status_box = st.empty()
                decision, price, tf_name = analyze_ticker(ticker, status_box)
                
                if price > 0 and decision != "HATA":
                    status = row['Durum']
                    log_msg = row['Son_Islem_Log']
                    
                    # --- İŞLEM YAP ---
                    if status == 'COIN' and decision == 'SAT':
                        amount = float(row['Miktar'])
                        if amount > 0:
                            cash_val = amount * price
                            updated_portfolio.at[idx, 'Durum'] = 'CASH'
                            updated_portfolio.at[idx, 'Nakit_Bakiye_USD'] = cash_val
                            updated_portfolio.at[idx, 'Miktar'] = 0.0
                            updated_portfolio.at[idx, 'Son_Islem_Fiyati'] = price
                            log_msg = f"SATILDI ({tf_name})"
                            updated_portfolio.at[idx, 'Son_Islem_Zamani'] = time_str
                            
                    elif status == 'CASH' and decision == 'AL':
                        cash_val = float(row['Nakit_Bakiye_USD'])
                        if cash_val > 1.0:
                            amount = cash_val / price
                            updated_portfolio.at[idx, 'Durum'] = 'COIN'
                            updated_portfolio.at[idx, 'Miktar'] = amount
                            updated_portfolio.at[idx, 'Nakit_Bakiye_USD'] = 0.0
                            updated_portfolio.at[idx, 'Son_Islem_Fiyati'] = price
                            log_msg = f"ALINDI ({tf_name})"
                            updated_portfolio.at[idx, 'Son_Islem_Zamani'] = time_str
                    
                    # Değer Güncelle
                    val = (float(updated_portfolio.at[idx, 'Miktar']) * price) if updated_portfolio.at[idx, 'Durum'] == 'COIN' else float(updated_portfolio.at[idx, 'Nakit_Bakiye_USD'])
                    updated_portfolio.at[idx, 'Kaydedilen_Deger_USD'] = val
                    updated_portfolio.at[idx, 'Son_Islem_Log'] = log_msg

            progress_bar.progress((i + 1) / len(updated_portfolio))
        
        # Kaydet
        save_portfolio(updated_portfolio, sheet)
        st.success("Tüm işlemler tamamlandı ve Sheets güncellendi!")

# Mevcut Durum Tablosu
st.divider()
st.subheader("📋 Mevcut Portföy (Canlı)")
try:
    df_view, _ = load_and_fix_portfolio()
    if not df_view.empty:
        st.dataframe(df_view)
        total = df_view['Kaydedilen_Deger_USD'].sum()
        st.metric("Toplam Portföy Değeri", f"${total:,.2f}")
except: pass
