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

# --- AI & ML Kütüphaneleri ---
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Hedge Fund AI: Canavar Motor", layout="wide")
st.title("🏦 Hedge Fund AI: Canavar Motor")

# =============================================================================
# 1. AYARLAR VE SABİTLER
# =============================================================================
SHEET_ID = "16zjLeps0t1P26OF3o7XQ-djEKKZtZX6t5lFxLmnsvpE"
CREDENTIALS_FILE = "service_account.json"
TARGET_COINS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "DOGE-USD"]
DATA_PERIOD = "3y" # 3 Yıllık veri çekmek için güncellendi

with st.sidebar:
    st.header("⚙️ Ayarlar")
    use_ga = st.checkbox("Genetic Algoritma (GA) Optimizasyonu", value=True)
    ga_gens = st.number_input("GA Döngüsü", 1, 20, 5)
    st.info("Sistem, en yüksek Alpha'yı üreten zaman dilimini (Günlük/Haftalık/Aylık) seçer.")

# =============================================================================
# 2. GOOGLE SHEETS ENTEGRASYONU
# =============================================================================
def connect_sheet():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = None
    if "gcp_service_account" in st.secrets:
        try: creds = ServiceAccountCredentials.from_json_keyfile_dict(dict(st.secrets["gcp_service_account"]), scope)
        except: pass
    elif os.path.exists(CREDENTIALS_FILE):
        creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, scope)
    if not creds: return None
    try:
        client = gspread.authorize(creds)
        return client.open_by_key(SHEET_ID).sheet1
    except: return None

def load_and_fix_portfolio():
    sheet = connect_sheet()
    if sheet is None: return pd.DataFrame(), None
    try:
        headers = sheet.row_values(1)
        required_cols = ["Ticker", "Durum", "Miktar", "Son_Islem_Fiyati", 
                         "Nakit_Bakiye_USD", "Baslangic_USD", "Kaydedilen_Deger_USD", 
                         "Son_Islem_Log", "Son_Islem_Zamani"]
        if not headers or headers[0] != "Ticker":
            sheet.clear(); sheet.append_row(required_cols)
            defaults = []
            for t in TARGET_COINS:
                defaults.append([t, "CASH", 0, 0, 10, 10, 10, "KURULUM", "-"])
            for d in defaults: sheet.append_row(d)
            time.sleep(2)
    except: pass
    data = sheet.get_all_records()
    df = pd.DataFrame(data)
    df = df[df['Ticker'].astype(str).str.len() > 3]
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
        sheet.update([df_export.columns.values.tolist()] + df_export.values.tolist())
    except: pass

# =============================================================================
# 3. AI MOTORU - VERİ İŞLEME VE ÖZELLİK MÜHENDİSLİĞİ
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

def calculate_heuristic_score(df):
    """Mevcut Sezgiselleri (Heuristic) hesaplar."""
    if len(df) < 150: return pd.Series(0.0, index=df.index)
    s1 = np.sign(df['close'].pct_change(5).fillna(0))
    s2 = np.sign(df['close'].pct_change(30).fillna(0))
    s3 = np.where(df['close'] > df['close'].rolling(150).mean(), 1, -1)
    vol = df['close'].pct_change().rolling(20).std()
    s4 = np.where(vol < vol.shift(1), 1, -1)
    s5 = np.sign(df['close'].diff(10).fillna(0))
    momentum = np.sign(df['close'].diff(20).fillna(0))
    return (s1 + s2 + s3 + s4 + s5 + momentum) / 6.0

def get_raw_data(ticker):
    """YFinance üzerinden ham veriyi çeker."""
    try:
        df = yf.download(ticker, period=DATA_PERIOD, interval="1d", progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        if 'close' not in df.columns and 'adj close' in df.columns: df['close'] = df['adj close']
        return df
    except: return None

def process_data(df, timeframe):
    """Veriyi işler, zaman dilimine göre yeniden örnekler ve tüm özellikleri oluşturur."""
    if df is None or len(df) < 150: return None
    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    if timeframe == 'W': df_res = df.resample('W').agg(agg).dropna()
    elif timeframe == 'M': df_res = df.resample('ME').agg(agg).dropna()
    else: df_res = df.copy()
    if len(df_res) < 100: return None

    # TEMEL ÖZELLİKLERİN OLUŞTURULMASI (Hata önleme için önce yapılmalı)
    df_res['kalman_close'] = apply_kalman_filter(df_res['close'])
    df_res['log_ret'] = np.log(df_res['kalman_close'] / df_res['kalman_close'].shift(1))
    df_res['range'] = (df_res['high'] - df_res['low']) / df_res['close'] # 'range' burada oluşturuldu
    df_res['heuristic'] = calculate_heuristic_score(df_res)
    df_res['ret'] = df_res['close'].pct_change() # Yüzdesel getiri

    # YENİ İSTATİSTİKSEL MODELLER/ÖZELLİKLER (İstenen geliştirmeler)
    
    # 1. Tarihsel Ortalama Değişimler (5 ay ve 3 yıl)
    df_res['avg_ret_5m'] = df_res['ret'].rolling(window=100).mean() * 100 
    df_res['avg_ret_3y'] = df_res['ret'].rolling(window=750).mean() * 100 

    # 2. Haftanın Günü Etkisi Puanı
    df_res['day_of_week'] = df_res.index.dayofweek
    day_returns = df_res.groupby('day_of_week')['ret'].mean().fillna(0)
    df_res['day_score'] = df_res['day_of_week'].map(day_returns).fillna(0)
    
    # Yeni ortalamaları birleştiren normalize puan
    avg_feats = df_res[['avg_ret_5m', 'avg_ret_3y', 'day_score']].fillna(0)
    if not avg_feats.empty:
        scaler_avg = StandardScaler()
        df_res['historical_avg_score'] = scaler_avg.fit_transform(avg_feats).mean(axis=1)
    else:
        df_res['historical_avg_score'] = 0.0

    # 3. Oynaklık Değişim Puanı (Range Volatility Delta)
    df_res['range_vol_delta'] = df_res['range'].pct_change(5).fillna(0)

    # Hedef (Target) Sütununun Oluşturulması
    df_res['target'] = (df_res['close'].shift(-1) > df_res['close']).astype(int)
    
    # HATA DÜZELTME: Sonsuz (inf) değerleri NaN ile değiştir ve temizle
    df_res.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    df_res.dropna(inplace=True)
    return df_res

# =============================================================================
# 4. AI MOTORU - MODEL EĞİTİMİ VE ENSEMBLE
# =============================================================================

def ga_optimize(df, n_gen=5):
    """Genetic Algoritma ile basit RF modelini optimize eder."""
    best_depth = 5; best_nest = 50; best_score = -999
    # Yeni özellik setini RF optimizasyonuna dahil et
    features = ['log_ret', 'range', 'heuristic', 'historical_avg_score', 'range_vol_delta']
    
    for d in [3, 5, 7, 9]:
        for n in [20, 50, 100]:
            train = df.iloc[:-30]; test = df.iloc[-30:]
            current_features = [f for f in features if f in train.columns]
            
            if not current_features: continue

            rf = RandomForestClassifier(n_estimators=n, max_depth=d).fit(train[current_features], train['target'])
            score = rf.score(test[current_features], test['target'])
            if score > best_score:
                best_score = score; best_depth = d; best_nest = n
    return {'rf_depth': best_depth, 'rf_nest': best_nest, 'xgb_params': {'max_depth':3, 'n_estimators':50}}


def train_meta_learner(df, params=None):
    """Ana modelleri eğitir ve Lojistik Regresyon ile birleştirir (Meta-Learner)."""
    rf_d = params['rf_depth'] if params else 5
    rf_n = params['rf_nest'] if params else 50
    test_size = 60
    
    if len(df) < test_size + 50: return 0.0, None
    train = df.iloc[:-test_size]; test = df.iloc[-test_size:]
    
    # Tüm base modeller için GENİŞLETİLMİŞ özellik seti
    base_features = ['log_ret', 'range', 'heuristic', 'historical_avg_score', 'range_vol_delta']
    X_tr = train[base_features]; y_tr = train['target']
    X_test = test[base_features]

    # 1. RandomForest, 2. XGBoost eğitimi (Yeni özelliklerle)
    rf = RandomForestClassifier(n_estimators=rf_n, max_depth=rf_d, random_state=42).fit(X_tr, y_tr)
    xgb_c = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=50, max_depth=3).fit(X_tr, y_tr)
    
    # 3. HMM eğitimi (Oynaklık ve Getiri özellikleri ile)
    scaler = StandardScaler()
    X_hmm = scaler.fit_transform(train[['log_ret', 'range_vol_delta']])
    hmm = GaussianHMM(n_components=3, covariance_type='diag', n_iter=50, random_state=42)
    try: hmm.fit(X_hmm)
    except: hmm = None
    
    hmm_pred = np.zeros(len(train))
    if hmm:
        pr = hmm.predict_proba(X_hmm)
        bull = np.argmax(hmm.means_[:,0]); bear = np.argmin(hmm.means_[:,0])
        hmm_pred = pr[:, bull] - pr[:, bear]
        
    # Meta Öğreniciye Girdiler
    meta_X = pd.DataFrame({
        'RF': rf.predict_proba(X_tr)[:,1],
        'XGB': xgb_c.predict_proba(X_tr)[:,1],
        'HMM': hmm_pred,
        'Heuristic': train['heuristic'].values,
        'Historical_Avg_Score': train['historical_avg_score'].values # Yeni Faktör 1
    })
    
    # 4. Meta-Model: Lojistik Regresyon (Tüm Model Çıktılarını Birleştirir)
    meta_model = LogisticRegression().fit(meta_X, y_tr)
    weights = meta_model.coef_[0]
    
    # Simülasyon
    sim_eq=[100]; hodl_eq=[100]; cash=100; coin=0; p0=test['close'].iloc[0]
    
    # Test verisi için HMM tahminleri
    X_hmm_t = scaler.transform(test[['log_ret','range_vol_delta']])
    hmm_p_t = hmm.predict_proba(X_hmm_t) if hmm else np.zeros((len(test),3))
    hmm_s_t = hmm_p_t[:, np.argmax(hmm.means_[:,0])] - hmm_p_t[:, np.argmin(hmm.means_[:,0])] if hmm else np.zeros(len(test))
    
    # Test verisi için Meta Öğrenici Girdileri
    mx_test = pd.DataFrame({
        'RF': rf.predict_proba(X_test)[:,1],
        'XGB': xgb_c.predict_proba(X_test)[:,1],
        'HMM': hmm_s_t,
        'Heuristic': test['heuristic'].values,
        'Historical_Avg_Score': test['historical_avg_score'].values
    })
    
    probs = meta_model.predict_proba(mx_test)[:,1]
    
    # Ticaret Simülasyonu
    for i in range(len(test)):
        p = test['close'].iloc[i]; s=(probs[i]-0.5)*2
        if s>0.25 and cash>0: coin=cash/p; cash=0
        elif s<-0.25 and coin>0: cash=coin*p; coin=0
        sim_eq.append(cash+coin*p); hodl_eq.append((100/p0)*p)
        
    final_signal=(probs[-1]-0.5)*2
    
    # GÜNCELLENMİŞ Model Etki İsimleri (Streamlit için)
    weights_names = ['RandomForest','XGBoost','HMM','Senin Kuralın (Heuristic)','Tarihsel Ortalamalar']
    
    info={'weights': weights, 'weights_names': weights_names, 'bot_eq': sim_eq[1:],'hodl_eq': hodl_eq[1:],'dates': test.index,'alpha': (sim_eq[-1]-hodl_eq[-1]),'bot_roi': (sim_eq[-1]-100),'hodl_roi': (hodl_eq[-1]-100),'conf': probs[-1],'my_score': test['heuristic'].iloc[-1]}
    
    return final_signal, info

# =============================================================================
# 5. TURNUVA FONKSİYONU
# =============================================================================
def analyze_ticker_tournament(ticker, status_placeholder):
    raw_df = get_raw_data(ticker)
    if raw_df is None: 
        status_placeholder.error("Veri Yok")
        return "HATA", 0.0, "YOK", None
    current_price = float(raw_df['close'].iloc[-1])
    timeframes={'GÜNLÜK':'D','HAFTALIK':'W','AYLIK':'M'}
    best_alpha=-9999; final_decision="BEKLE"; winning_tf="YOK"; best_info=None
    
    for tf_name, tf_code in timeframes.items():
        status_placeholder.text(f"Turnuva: {tf_name} grafiği test ediliyor...")
        df = process_data(raw_df, tf_code)
        if df is None: continue
        
        # GA optimizasyonu
        params = ga_optimize(df) if st.session_state.get('use_ga',True) else None
        
        # Meta Öğreniciyi eğit ve sinyal al
        sig, info = train_meta_learner(df, params)
        
        if info is None: continue
        
        if info['alpha']>best_alpha:
            best_alpha=info['alpha']; winning_tf=tf_name; best_info=info
            if sig>0.25: final_decision="AL"
            elif sig<-0.25: final_decision="SAT"
            else: final_decision="BEKLE"
    return final_decision, current_price, winning_tf, best_info

# =============================================================================
# 6. ARAYÜZ (STREAMLIT) VE İŞLEM MANTIĞI
# =============================================================================
if st.button("🚀 PORTFÖYÜ CANLI ANALİZ ET", type="primary"):
    st.session_state['use_ga'] = use_ga
    tz = pytz.timezone('Europe/Istanbul')
    time_str = datetime.now(tz).strftime("%d-%m %H:%M")
    pf_df, sheet = load_and_fix_portfolio()
    
    if pf_df.empty:
        st.error("Hata: Portföy yüklenemedi.")
    else:
        updated = pf_df.copy(); prog = st.progress(0); sim_summary=[]
        
        for i,(idx,row) in enumerate(updated.iterrows()):
            ticker=row['Ticker']
            if len(str(ticker))<3: continue
            
            with st.expander(f"🧠 {ticker} Analiz Raporu", expanded=True):
                ph = st.empty()
                dec, prc, tf, info = analyze_ticker_tournament(ticker, ph)
                
                if dec!="HATA" and info:
                    sim_summary.append({"Coin":ticker,"Kazanan TF":tf,"Bot ROI":info['bot_roi'],"HODL ROI":info['hodl_roi'],"Alpha":info['alpha']})
                    
                    # Model Etki Dağılımının Streamlit'te Gösterilmesi
                    w=info['weights']; w_names=info['weights_names']
                    w_abs=np.abs(w); w_norm=w_abs/(np.sum(w_abs)+1e-9)*100
                    
                    # Etkileri büyükten küçüğe sıralama
                    w_df=pd.DataFrame({'Faktör':w_names,'Etki (%)':w_norm})
                    w_df=w_df.sort_values(by='Etki (%)', ascending=False)
                    
                    c1,c2=st.columns([1,2])
                    with c1:
                        st.markdown(f"### Karar: **{dec}**"); st.caption(f"Seçilen Zaman Dilimi: {tf}"); st.markdown(f"**Senin Puanın:** {info['my_score']:.2f}"); st.markdown("**Model Etki Dağılımı:**")
                        st.dataframe(w_df, hide_index=True) # SIRALI DATAFRAME
                    with c2:
                        # Grafik
                        fig=go.Figure(); fig.add_trace(go.Scatter(x=info['dates'],y=info['bot_eq'],name="Bot",line=dict(color='green',width=2)))
                        fig.add_trace(go.Scatter(x=info['dates'],y=info['hodl_eq'],name="HODL",line=dict(color='gray',dash='dot')))
                        color_ti="green" if info['alpha']>0 else "red"
                        fig.update_layout(title=f"Kazanan Strateji ({tf}) Alpha: ${info['alpha']:.2f}",title_font_color=color_ti,height=250,template="plotly_dark",margin=dict(t=30,b=0,l=0,r=0))
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # İşlem Kaydı ve Portföy Güncellemesi
                    stt=row['Durum']
                    if stt=='COIN' and dec=='SAT':
                        amt=float(row['Miktar'])
                        if amt>0: updated.at[idx,'Durum']='CASH'; updated.at[idx,'Nakit_Bakiye_USD']=amt*prc; updated.at[idx,'Miktar']=0.0; updated.at[idx,'Son_Islem_Fiyati']=prc; updated.at[idx,'Son_Islem_Log']=f"SAT ({tf}) A:{info['alpha']:.1f}"; updated.at[idx,'Son_Islem_Zamani']=time_str
                    elif stt=='CASH' and dec=='AL':
                        cash=float(row['Nakit_Bakiye_USD'])
                        if cash>1: updated.at[idx,'Durum']='COIN'; updated.at[idx,'Miktar']=cash/prc; updated.at[idx,'Nakit_Bakiye_USD']=0.0; updated.at[idx,'Son_Islem_Fiyati']=prc; updated.at[idx,'Son_Islem_Log']=f"AL ({tf}) A:{info['alpha']:.1f}"; updated.at[idx,'Son_Islem_Zamani']=time_str
                        
                    val=(float(updated.at[idx,'Miktar'])*prc) if updated.at[idx,'Durum']=='COIN' else float(updated.at[idx,'Nakit_Bakiye_USD'])
                    updated.at[idx,'Kaydedilen_Deger_USD']=val
                    ph.success(f"Analiz Bitti. En iyi grafik: {tf}")
                    
            prog.progress((i+1)/len(updated))
            
        save_portfolio(updated, sheet)
        
        # Genel Performans Özeti
        st.divider(); st.subheader("🏆 Turnuva Sonuçları & Performans")
        if sim_summary:
            sum_df=pd.DataFrame(sim_summary)
            col1,col2,col3=st.columns(3)
            col1.metric("Ort. Bot Getirisi", f"%{sum_df['Bot ROI'].mean():.2f}")
            col2.metric("Ort. HODL Getirisi", f"%{sum_df['HODL ROI'].mean():.2f}")
            col3.metric("TOPLAM ALPHA", f"%{sum_df['Alpha'].mean():.2f}", delta_color="normal")
            st.dataframe(sum_df.style.format("{:.2f}", subset=["Bot ROI","HODL ROI","Alpha"]))
            
        st.success("✅ Canavar Motor Tamamlandı!")
