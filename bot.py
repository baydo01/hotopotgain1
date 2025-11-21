import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import warnings
import gspread
import os
import json
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
from deap import base, creator, tools, algorithms

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Hedge Fund AI: Master", layout="wide")
st.title("🏦 Hedge Fund AI: Oto-Ağırlıklı Hibrit Yönetim")

# =============================================================================
# 1. AYARLAR VE GÜVENLİK
# =============================================================================

SHEET_ID = "16zjLeps0t1P26OF3o7XQ-djEKKZtZX6t5lFxLmnsvpE"
CREDENTIALS_FILE = "service_account.json"
ga_gens = 5

with st.sidebar:
    st.header("⚙️ Kontrol Paneli")
    use_ga = st.checkbox("Genetic Algoritma Kullan", value=False)
    ga_gens = st.number_input("GA Jenerasyon Sayısı", 1, 50, 5)

# -------------------- HEURISTIC (5 Adımlı Puanlama Mantığı) --------------------
def calculate_heuristic_score(df):
    """
    Kullanıcının istediği 5 adımlı (yüzdesel/trend) puanlama sistemini hesaplar.
    """
    if len(df) < 252: return pd.Series(0.0, index=df.index)
    
    # 1. Kısa Vade Yön (5 Bar %)
    s1_pct = df['close'].pct_change(5).fillna(0) * 10 
    
    # 2. Orta Vade Yön (30 Bar %)
    s2_pct = df['close'].pct_change(30).fillna(0) * 5 
    
    # 3. Uzun Vade Makro Eğilim (252 Bar MA Eğim)
    ma_252 = df['close'].rolling(252).mean()
    s3_trend = np.where(df['close'] > ma_252, 1, -1) * 2 # Fiyat MA'nın üzerindeyse +2

    # Toplam Skoru döndür (Bu, LogReg'e bir özellik olarak girecektir)
    score = s1_pct + s2_pct + s3_trend.fillna(0)
    return score.clip(-5, 5)

# =============================================================================
# 2. GOOGLE SHEETS VE BAĞLANTI
# =============================================================================

def connect_sheet():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = None
    if "gcp_service_account" in st.secrets:
        try:
            creds_dict = dict(st.secrets["gcp_service_account"])
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        except: pass
    if not creds and os.path.exists(CREDENTIALS_FILE):
        creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, scope)
        
    if not creds: return None
    try:
        client = gspread.authorize(creds)
        return client.open_by_key(SHEET_ID).sheet1
    except Exception as e: return None

def load_and_fix_portfolio():
    sheet = connect_sheet()
    if sheet is None: return pd.DataFrame(), None

    # OTO-KURULUM
    try:
        headers = sheet.row_values(1)
        required_cols = ["Ticker", "Durum", "Miktar", "Son_Islem_Fiyati", 
                         "Nakit_Bakiye_USD", "Baslangic_USD", "Kaydedilen_Deger_USD", 
                         "Son_Islem_Log", "Son_Islem_Zamani"]
        
        if not headers or headers[0] != "Ticker":
            sheet.clear(); sheet.append_row(required_cols)
            # 6 COIN BAŞLANGICI
            defaults = [
                ["BTC-USD", "CASH", 0, 0, 10, 10, 10, "KURULUM", "-"],
                ["ETH-USD", "CASH", 0, 0, 10, 10, 10, "KURULUM", "-"],
                ["SOL-USD", "CASH", 0, 0, 10, 10, 10, "KURULUM", "-"],
                ["BNB-USD", "CASH", 0, 0, 10, 10, 10, "KURULUM", "-"],
                ["XRP-USD", "CASH", 0, 0, 10, 10, 10, "KURULUM", "-"],
                ["DOGE-USD", "CASH", 0, 0, 10, 10, 10, "KURULUM", "-"]
            ]
            for d in defaults: sheet.append_row(d)
            time.sleep(2)
            
    except Exception as e: print(f"Oto-Kurulum Hatası: {e}")
        
    data = sheet.get_all_records()
    df = pd.DataFrame(data)
    
    # Veri temizliği
    df = df[df['Ticker'].str.contains('-USD', na=False)]
    numeric_cols = ["Miktar", "Son_Islem_Fiyati", "Nakit_Bakiye_USD", "Baslangic_USD", "Kaydedilen_Deger_USD"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            
    return df, sheet

def save_portfolio(df, sheet):
    if sheet is None: return
    try:
        df_export = df.copy(); df_export = df_export.astype(str)
        sheet.clear()
        sheet.update([df_export.columns.values.tolist()] + df_export.values.tolist())
    except Exception as e: st.error(f"Kaydetme Hatası: {e}")

# =============================================================================
# 3. YAPAY ZEKA MOTORU VE HİBRİT PREDİCT (LR ENTEGRASYONU)
# =============================================================================

def apply_kalman_filter(prices):
    n_iter = len(prices); sz = (n_iter,); Q = 1e-5; R = 0.01 ** 2
    xhat = np.zeros(sz); P = np.zeros(sz); xhatminus = np.zeros(sz); Pminus = np.zeros(sz); K = np.zeros(sz)
    xhat[0] = prices.iloc[0]; P[0] = 1.0
    for k in range(1, n_iter):
        xhatminus[k] = xhat[k - 1]; Pminus[k] = P[k - 1] + Q
        K[k] = Pminus[k] / (Pminus[k] + R); xhat[k] = xhatminus[k] + K[k] * (prices.iloc[k] - xhatminus[k])
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
    
    if len(df_res) < 252: return None # Heuristic için yeterli veri olmalı (252 bar)
    
    df_res['kalman_close'] = apply_kalman_filter(df_res['close'])
    df_res['log_ret'] = np.log(df_res['kalman_close'] / df_res['kalman_close'].shift(1))
    df_res['range'] = (df_res['high'] - df_res['low']) / df_res['close']
    df_res['trend_signal'] = np.where(df_res['close'] > df_res['kalman_close'], 1, -1)
    
    # YENİ HEURISTIC SCORE (ÖZELLİK OLARAK EKLE)
    df_res['heuristic_score'] = calculate_heuristic_score(df_res)
    
    df_res['target'] = (df_res['close'].shift(-1) > df_res['close']).astype(int)
    df_res.dropna(inplace=True)
    return df_res

def train_models_for_window(train_df, rf_depth=5, xgb_params=None, n_hmm=3):
    # LogReg'in öğreneceği TÜM ÖZELLİKLER (Heuristic dahil)
    features_base = ['log_ret','range','trend_signal', 'heuristic_score']
    X = train_df[features_base]; y = train_df['target']
    
    # 1. Klasik Modelleri Eğit (XGB, RF)
    clf_rf = RandomForestClassifier(n_estimators=30, max_depth=rf_depth, n_jobs=-1, random_state=42)
    clf_rf.fit(X, y)
    
    if xgb_params is None: xgb_params = {'n_estimators':30, 'max_depth':3, 'learning_rate':0.1,'n_jobs':-1, 'enable_categorical':True}
    clf_xgb = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', **xgb_params)
    clf_xgb.fit(X, y)
    
    # 2. HMM Eğit (Sadece 2 temel özellik)
    hmm_model = None
    try:
        Xh_s = StandardScaler().fit_transform(train_df[['log_ret','range']].values)
        hmm_model = GaussianHMM(n_components=3, covariance_type='diag', n_iter=50, random_state=42)
        hmm_model.fit(Xh_s)
    except: pass

    # 3. META-MODEL (LOGISTIC REGRESSION) Eğitimi
    # Girdiler: 1) HMM Sinyali, 2) XGBoost Olasılığı, 3) RF Olasılığı, 4) Kullanıcının Heuristic Puanı
    
    hmm_sig_train = np.array([predict_hmm_signal(models={'hmm':hmm_model}, row=row) for _, row in train_df.iterrows()])
    
    meta_features = pd.DataFrame({
        'hmm_sig': hmm_sig_train,
        'xgb_prob': clf_xgb.predict_proba(X)[:, 1],
        'rf_prob': clf_rf.predict_proba(X)[:, 1],
        'heuristic_score': train_df['heuristic_score'].values
    })
    
    # LogReg, bu 4 özelliği (HMM, XGB, RF, Heuristic) kullanarak kararı öğrenecek.
    meta_clf = LogisticRegression(max_iter=200); meta_clf.fit(meta_features, y)
    
    return {'rf': clf_rf, 'xgb': clf_xgb, 'hmm': hmm_model, 'meta': meta_clf, 'scaler': StandardScaler().fit(X)}

def predict_hmm_signal(models, row):
    if models['hmm']:
        Xh = row[['log_ret','range']].values.reshape(1,-1)
        try:
            probs = models['hmm'].predict_proba(StandardScaler().fit_transform(Xh))[0]
            return probs[np.argmax(models['hmm'].means_[:,0])] - probs[np.argmin(models['hmm'].means_[:,0])]
        except: return 0.0
    return 0.0

def predict_with_models(models, row):
    if models is None: return 0
    
    # 1. BASE MODELLERDEN VERİ AL
    Xrow = pd.DataFrame([row[['log_ret','range','trend_signal', 'heuristic_score']]], columns=['log_ret','range','trend_signal', 'heuristic_score'])
    rf_prob = models['rf'].predict_proba(Xrow)[0][1]
    xgb_prob = models['xgb'].predict_proba(Xrow)[0][1]
    
    # 2. HMM Sinyali
    hmm_sig = predict_hmm_signal(models, row)
    
    # 3. META-MODEL GİRDİSİ OLUŞTUR
    meta_input = pd.DataFrame({
        'hmm_sig': [hmm_sig],
        'xgb_prob': [xgb_prob],
        'rf_prob': [rf_prob],
        'heuristic_score': [row['heuristic_score']]
    })

    # 4. LOGREG İLE FİNAL TAHMİNİ (KENDİ ÖĞRENDİĞİ AĞIRLIKLARLA)
    # LogReg'in çıktısı 0 ile 1 arası bir olasılıktır.
    final_prob = models['meta'].predict_proba(meta_input)[0][1]
    
    # 5. ÖLÇEKLEME (-1 ile +1 arası sinyal)
    return (final_prob - 0.5) * 2

def analyze_ticker(ticker, status_placeholder):
    raw_df = get_raw_data(ticker)
    if raw_df is None: return "HATA", 0.0, None

    current_price = float(raw_df['close'].iloc[-1])
    timeframes = {'GÜNLÜK': 'D', 'HAFTALIK': 'W'}
    best_score = -999; final_decision = "BEKLE"; winning_tf = "YOK"; winning_df = None
    
    for tf_name, tf_code in timeframes.items():
        status_placeholder.markdown(f"⏳ {ticker} -> **{tf_name}** Modeli Eğitiliyor...")
        df = process_data(raw_df, tf_code)
        if df is None: continue
        
        # GA optimizasyonunu atlayıp default parametrelerle eğitelim
        models = train_models_for_window(df.iloc[-252:], rf_depth=5) # Son 1 yıl (252 bar)
        if models is None: continue

        signal = predict_with_models(models, df.iloc[-1])
        
        if abs(signal) > best_score:
            best_score = abs(signal)
            winning_tf = tf_name
            if signal > 0.25: final_decision = "AL"
            elif signal < -0.25: final_decision = "SAT"
            else: final_decision = "BEKLE"

    status_placeholder.markdown(f"**Karar:** {final_decision} ({winning_tf}) | **Fiyat:** ${current_price:.2f}")
    return final_decision, current_price, winning_tf

# =============================================================================
# 5. ANA ÇALIŞTIRMA VE UI
# =============================================================================

if st.button("🚀 PORTFÖYÜ CANLI ANALİZ ET VE GÜNCELLE", type="primary"):
    tz = pytz.timezone('Europe/Istanbul')
    time_str = datetime.now(tz).strftime("%d-%m %H:%M")
    
    with st.spinner("Google Sheets'e bağlanılıyor ve tablo kontrol ediliyor..."):
        pf_df, sheet = load_and_fix_portfolio()
    
    if pf_df.empty:
        st.error("Portföy yüklenemedi.")
    else:
        st.success("✅ Bağlantı başarılı. Analiz başlıyor...")
        updated_portfolio = pf_df.copy()
        progress_bar = st.progress(0)
        
        for i, (idx, row) in enumerate(updated_portfolio.iterrows()):
            ticker = row['Ticker']
            if not ticker or ticker == "-": continue
            
            with st.expander(f"{ticker} Analizi", expanded=True):
                status_box = st.empty()
                decision, price, tf_name = analyze_ticker(ticker, status_box)
                
                if price > 0 and decision != "HATA":
                    status = row['Durum']
                    
                    if status == 'COIN' and decision == 'SAT':
                        amount = float(row['Miktar'])
                        if amount > 0:
                            cash_val = amount * price
                            updated_portfolio.at[idx, 'Durum'] = 'CASH'; updated_portfolio.at[idx, 'Nakit_Bakiye_USD'] = cash_val
                            updated_portfolio.at[idx, 'Miktar'] = 0.0; updated_portfolio.at[idx, 'Son_Islem_Fiyati'] = price
                            updated_portfolio.at[idx, 'Son_Islem_Log'] = f"SATILDI ({tf_name})"; updated_portfolio.at[idx, 'Son_Islem_Zamani'] = time_str
                            
                    elif status == 'CASH' and decision == 'AL':
                        cash_val = float(row['Nakit_Bakiye_USD'])
                        if cash_val > 1.0:
                            amount = cash_val / price
                            updated_portfolio.at[idx, 'Durum'] = 'COIN'; updated_portfolio.at[idx, 'Miktar'] = amount
                            updated_portfolio.at[idx, 'Nakit_Bakiye_USD'] = 0.0; updated_portfolio.at[idx, 'Son_Islem_Fiyati'] = price
                            updated_portfolio.at[idx, 'Son_Islem_Log'] = f"ALINDI ({tf_name})"; updated_portfolio.at[idx, 'Son_Islem_Zamani'] = time_str
                    
                    val = (float(updated_portfolio.at[idx, 'Miktar']) * price) if updated_portfolio.at[idx, 'Durum'] == 'COIN' else float(updated_portfolio.at[idx, 'Nakit_Bakiye_USD'])
                    updated_portfolio.at[idx, 'Kaydedilen_Deger_USD'] = val

            progress_bar.progress((i + 1) / len(updated_portfolio))
        
        save_portfolio(updated_portfolio, sheet)
        st.success("✅ TÜM İŞLEMLER TAMAMLANDI VE KAYDEDİLDİ!")

st.divider()
st.subheader("📋 Mevcut Portföy Durumu (Sheets'ten Okunan)")
try:
    df_view, _ = load_and_fix_portfolio()
    if not df_view.empty:
        st.dataframe(df_view)
        total = df_view['Kaydedilen_Deger_USD'].sum()
        st.metric("Toplam Portföy Değeri", f"${total:,.2f}")
except: st.warning("Portföy okunamıyor. Butona basarak yeniden deneyin.")
