import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import warnings
import gspread
import os
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import pytz

# --- ML & QUANT LIBRARIES ---
from arch import arch_model
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import xgboost as xgb
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# =============================================================================
# 1. AYARLAR & UI
# =============================================================================
st.set_page_config(page_title="Hedge Fund AI: V6 Logger", layout="wide", page_icon="🏛️")

st.markdown("""
<style>
    .main {background-color: #0E1117;}
    .header-box {
        background: linear-gradient(135deg, #1c1e21 0%, #363b45 100%);
        padding: 25px; border-radius: 12px; border-left: 5px solid #00CC96;
        margin-bottom: 25px; box-shadow: 0 4px 20px rgba(0,0,0,0.6);
    }
    .header-title {font-size: 32px; font-weight: 700; color: #fff; margin:0;}
    .header-sub {font-size: 14px; color: #b0b0b0; margin-top: 5px;}
</style>
<div class="header-box">
    <div class="header-title">🏛️ Hedge Fund AI: Ultimate V6 (History Logger)</div>
    <div class="header-sub">Full Transaction History • Detailed Sheet Inspector • Auto-Logging System</div>
</div>
""", unsafe_allow_html=True)

SHEET_ID = "16zjLeps0t1P26OF3o7XQ-djEKKZtZX6t5lFxLmnsvpE"
CREDENTIALS_FILE = "service_account.json"
DATA_PERIOD = "730d"

# =============================================================================
# 2. BAĞLANTI KATMANI (GELİŞMİŞ)
# =============================================================================
def connect_sheet_services():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = None
    if "gcp_service_account" in st.secrets:
        try: creds = ServiceAccountCredentials.from_json_keyfile_dict(dict(st.secrets["gcp_service_account"]), scope)
        except: pass
    elif os.path.exists(CREDENTIALS_FILE):
        creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, scope)
    
    if not creds: return None, None
    
    try:
        client = gspread.authorize(creds)
        spreadsheet = client.open_by_key(SHEET_ID)
        
        # 1. Ana Portföy Sayfası (Sheet1)
        portfolio_sheet = spreadsheet.sheet1
        
        # 2. Geçmiş Log Sayfası (Gecmis)
        try:
            history_sheet = spreadsheet.worksheet("Gecmis")
        except:
            # Eğer yoksa oluşturur (Başlıklarla beraber)
            history_sheet = spreadsheet.add_worksheet(title="Gecmis", rows="1000", cols="6")
            history_sheet.append_row(["Tarih", "Ticker", "Islem", "Miktar", "Fiyat", "Model/Sebep"])
            
        return portfolio_sheet, history_sheet
    except Exception as e:
        st.error(f"Bağlantı Hatası: {e}")
        return None, None

def load_portfolio():
    pf_sheet, _ = connect_sheet_services()
    if pf_sheet is None: return pd.DataFrame(), None
    try:
        data = pf_sheet.get_all_records()
        df = pd.DataFrame(data)
        
        # Sayısal sütun temizliği
        cols = ["Miktar", "Son_Islem_Fiyati", "Nakit_Bakiye_USD", "Baslangic_USD", "Kaydedilen_Deger_USD"]
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', '.'), errors='coerce').fillna(0.0)
        
        # İstenen sütun kontrolü: Son_Islem_Tarihi
        if "Son_Islem_Tarihi" not in df.columns:
            df["Son_Islem_Tarihi"] = "-"
            
        return df, pf_sheet
    except: return pd.DataFrame(), None

def load_history_logs():
    _, hist_sheet = connect_sheet_services()
    if hist_sheet is None: return pd.DataFrame()
    try:
        data = hist_sheet.get_all_records()
        return pd.DataFrame(data)
    except: return pd.DataFrame()

def log_transaction(ticker, action, amount, price, model, hist_sheet):
    if hist_sheet is None: return
    now_str = datetime.now(pytz.timezone('Turkey')).strftime('%Y-%m-%d %H:%M')
    try:
        # ["Tarih", "Ticker", "Islem", "Miktar", "Fiyat", "Model/Sebep"]
        hist_sheet.append_row([now_str, ticker, action, float(amount), float(price), model])
    except Exception as e:
        print(f"Log Error: {e}")

def save_portfolio(df, sheet):
    if sheet is None: return
    try:
        df_exp = df.copy().astype(str)
        sheet.clear()
        sheet.update([df_exp.columns.values.tolist()] + df_exp.values.tolist())
    except: pass

# =============================================================================
# 3. AI & QUANT CORE
# =============================================================================
# (Önceki fonksiyonlar aynen korunuyor, sadece yer tasarrufu için özet geçiyorum)
def get_data(ticker):
    try:
        df = yf.download(ticker, period=DATA_PERIOD, interval="1d", progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        return df
    except: return None

def process_data_advanced(df):
    if df is None or len(df)<150: return None
    df = df.copy()
    df['kalman'] = df['close'].rolling(3).mean()
    df['log_ret'] = np.log(df['kalman']/df['kalman'].shift(1))
    df['ret'] = df['close'].pct_change()
    df['range'] = (df['high']-df['low'])/df['close']
    df['range_vol_delta'] = df['range'].pct_change(5)
    df['heuristic'] = (np.sign(df['close'].pct_change(5)) + np.sign(df['close'].pct_change(30)))/2.0
    
    df['avg_ret_5m'] = df['ret'].rolling(100).mean()*100
    df['avg_ret_3y'] = df['ret'].rolling(750).mean()*100
    avg_feats = df[['avg_ret_5m','avg_ret_3y']].fillna(0)
    df['historical_avg_score'] = StandardScaler().fit_transform(avg_feats).mean(axis=1)
    
    df['target'] = (df['close'].shift(-1)>df['close']).astype(int)
    df.dropna(subset=['target'], inplace=True)
    return df

def smart_impute(df, features):
    if len(df) < 50: return df.fillna(0)
    try: return pd.DataFrame(KNNImputer(n_neighbors=5).fit_transform(df[features]), columns=features, index=df.index)
    except: return df.fillna(0)

def estimate_garch_vol(returns):
    if len(returns) < 200: return 0.0
    try:
        am = arch_model(100*returns, vol='GARCH', p=1, o=0, q=1, dist='StudentsT')
        res = am.fit(disp='off')
        return float(np.sqrt(res.forecast(horizon=1).variance.iloc[-1,0])/100)
    except: return 0.0

class HedgeFundBrain:
    def __init__(self):
        self.meta_model = LogisticRegression(C=1.0)
    
    def train_predict_tournament(self, df):
        features = ['log_ret', 'range', 'heuristic', 'historical_avg_score', 'range_vol_delta']
        df = smart_impute(df, features)
        test_size = 60
        train = df.iloc[:-test_size]; test = df.iloc[-test_size:]
        X_tr = train[features]; y_tr = train['target']
        X_test = test[features]
        
        rf = RandomForestClassifier(n_estimators=100, max_depth=5).fit(X_tr, y_tr)
        etc = ExtraTreesClassifier(n_estimators=100, max_depth=5).fit(X_tr, y_tr)
        xgb_c = xgb.XGBClassifier(n_estimators=100, max_depth=3).fit(X_tr, y_tr)
        xgb_solo = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1).fit(X_tr, y_tr)
        
        scaler_hmm = StandardScaler()
        try:
            X_hmm = scaler_hmm.fit_transform(train[['log_ret', 'range_vol_delta']].fillna(0))
            hmm = GaussianHMM(n_components=3, covariance_type='diag', n_iter=50).fit(X_hmm)
            hmm_probs = hmm.predict_proba(X_hmm)
        except: hmm_probs = np.zeros((len(train), 3)); hmm = None
        
        meta_X_tr = pd.DataFrame({
            'RF': rf.predict_proba(X_tr)[:,1], 'ETC': etc.predict_proba(X_tr)[:,1],
            'XGB': xgb_c.predict_proba(X_tr)[:,1], 'Heuristic': train['heuristic'],
            'HMM_0': hmm_probs[:,0]
        }, index=train.index).fillna(0)
        self.meta_model.fit(meta_X_tr, y_tr)
        
        try:
            X_hmm_t = scaler_hmm.transform(test[['log_ret', 'range_vol_delta']].fillna(0))
            hmm_probs_t = hmm.predict_proba(X_hmm_t) if hmm else np.zeros((len(test),3))
        except: hmm_probs_t = np.zeros((len(test),3))
        
        meta_X_test = pd.DataFrame({
            'RF': rf.predict_proba(X_test)[:,1], 'ETC': etc.predict_proba(X_test)[:,1],
            'XGB': xgb_c.predict_proba(X_test)[:,1], 'Heuristic': test['heuristic'],
            'HMM_0': hmm_probs_t[:,0]
        }, index=test.index).fillna(0)
        
        p_ens = self.meta_model.predict_proba(meta_X_test)[:,1]
        p_solo = xgb_solo.predict_proba(X_test)[:,1]
        
        sim_ens = 100.0; sim_solo = 100.0; rets = test['close'].pct_change().fillna(0).values
        eq_ens = [100.0]; eq_solo = [100.0]
        for i in range(len(test)):
            if p_ens[i] > 0.55: sim_ens *= (1+rets[i])
            if p_solo[i] > 0.55: sim_solo *= (1+rets[i])
            eq_ens.append(sim_ens); eq_solo.append(sim_solo)
            
        garch = estimate_garch_vol(df['log_ret'].dropna().iloc[-200:])
        if sim_solo > sim_ens: return {'prob': p_solo[-1], 'winner': "Solo XGBoost", 'garch_vol': garch, 'winner_roi': sim_solo-100, 'eq_ens': eq_ens, 'eq_solo': eq_solo, 'dates': test.index}
        return {'prob': p_ens[-1], 'winner': "Ensemble", 'garch_vol': garch, 'winner_roi': sim_ens-100, 'eq_ens': eq_ens, 'eq_solo': eq_solo, 'dates': test.index}

# =============================================================================
# 4. EXECUTION & DASHBOARD
# =============================================================================
pf_df, sheet_pf = load_portfolio()
_, sheet_hist = connect_sheet_services() # Loglama için ikinci sayfa bağlantısı

tab1, tab2, tab3 = st.tabs(["📊 Dashboard & Kontrol", "📑 Canlı Sheet Verisi", "📜 Geçmiş İşlem Kayıtları"])

if not pf_df.empty:
    # --- TAB 1: ANA KONTROL ---
    with tab1:
        with st.sidebar:
            st.header("💼 Portföy Durumu")
            total_val = pf_df['Nakit_Bakiye_USD'].sum() + pf_df[pf_df['Durum']=='COIN']['Kaydedilen_Deger_USD'].sum()
            st.metric("Toplam Varlık", f"${total_val:.2f}")
            st.dataframe(pf_df[pf_df['Durum']=='COIN'][['Ticker', 'Miktar']], hide_index=True)

        if st.button("🏆 TURNUVAYI BAŞLAT VE ANALİZ ET", type="primary"):
            updated_pf = pf_df.copy()
            
            # --- BAKİYE YÖNETİMİ ---
            pool_cash = updated_pf['Nakit_Bakiye_USD'].sum()
            updated_pf['Nakit_Bakiye_USD'] = 0.0 # Mükerrer harcamayı önle
            
            buy_orders = []
            session_log = []
            brain = HedgeFundBrain()
            prog = st.progress(0)
            now_str = datetime.now(pytz.timezone('Turkey')).strftime('%Y-%m-%d %H:%M')
            
            for i, (idx, row) in enumerate(updated_pf.iterrows()):
                ticker = row['Ticker']
                df = get_data(ticker)
                
                if df is not None:
                    df = process_data_advanced(df)
                    res = brain.train_predict_tournament(df)
                    prob = res['prob']; winner = res['winner']
                    
                    decision = "HOLD"
                    if prob > 0.55: decision = "BUY"
                    elif prob < 0.45: decision = "SELL"
                    
                    with st.expander(f"{ticker} | {decision} | {winner} (ROI: %{res['winner_roi']:.1f})"):
                        c1, c2 = st.columns(2)
                        c1.metric("Güven", f"%{prob*100:.1f}")
                        c2.metric("Volatilite", f"%{res['garch_vol']*100:.2f}")
                        st.line_chart(pd.DataFrame({'Ensemble': res['eq_ens'], 'Solo XGB': res['eq_solo']}))
                    
                    current_p = df['close'].iloc[-1]
                    
                    # --- SATIŞ ---
                    if row['Durum'] == 'COIN' and decision == "SELL":
                        val = float(row['Miktar']) * current_p
                        pool_cash += val
                        updated_pf.at[idx, 'Durum'] = 'CASH'
                        updated_pf.at[idx, 'Miktar'] = 0.0
                        updated_pf.at[idx, 'Son_Islem_Log'] = f"SAT ({winner})"
                        updated_pf.at[idx, 'Son_Islem_Tarihi'] = now_str
                        
                        log_transaction(ticker, "SAT", row['Miktar'], current_p, winner, sheet_hist)
                        st.toast(f"🛑 {ticker} Satıldı!")
                        session_log.append({'Ticker': ticker, 'İşlem': 'SAT', 'Fiyat': current_p})
                        
                    # --- ALIM ADAYI ---
                    elif row['Durum'] == 'CASH' and decision == "BUY":
                        pos_scale = 0.5 if res['garch_vol'] > 0.05 else 1.0
                        buy_orders.append({'idx': idx, 'ticker': ticker, 'price': current_p, 'weight': prob * pos_scale, 'winner': winner})
                
                prog.progress((i+1)/len(updated_pf))
                
            # --- ALIM İŞLEMİ ---
            if buy_orders and pool_cash > 10:
                total_w = sum([b['weight'] for b in buy_orders])
                for b in buy_orders:
                    share = (b['weight'] / total_w) * pool_cash
                    amt = share / b['price']
                    
                    updated_pf.at[b['idx'], 'Durum'] = 'COIN'
                    updated_pf.at[b['idx'], 'Miktar'] = amt
                    updated_pf.at[b['idx'], 'Nakit_Bakiye_USD'] = 0.0
                    updated_pf.at[b['idx'], 'Son_Islem_Fiyati'] = b['price']
                    updated_pf.at[b['idx'], 'Son_Islem_Log'] = f"AL ({b['winner']})"
                    updated_pf.at[b['idx'], 'Son_Islem_Tarihi'] = now_str
                    
                    log_transaction(b['ticker'], "AL", amt, b['price'], b['winner'], sheet_hist)
                    st.toast(f"✅ {b['ticker']} Alındı!")
                    session_log.append({'Ticker': b['ticker'], 'İşlem': 'AL', 'Fiyat': b['price']})
            
            elif pool_cash > 0:
                fidx = updated_pf.index[0]
                updated_pf.at[fidx, 'Nakit_Bakiye_USD'] = float(updated_pf.at[fidx, 'Nakit_Bakiye_USD']) + pool_cash
            
            # Değerleme
            for idx, row in updated_pf.iterrows():
                if row['Durum'] == 'COIN':
                    try:
                        p = yf.download(row['Ticker'], period="1d", progress=False)['Close'].iloc[-1]
                        updated_pf.at[idx, 'Kaydedilen_Deger_USD'] = float(row['Miktar']) * float(p)
                    except: pass
                else: updated_pf.at[idx, 'Kaydedilen_Deger_USD'] = row['Nakit_Bakiye_USD']
                
            save_portfolio(updated_pf, sheet_pf)
            st.success("Analiz tamamlandı, Sheets güncellendi.")
            if session_log: st.table(pd.DataFrame(session_log))

    # --- TAB 2: DETAYLI SHEET GÖRÜNÜMÜ ---
    with tab2:
        st.subheader("📑 Google Sheets: Canlı Veri (Raw Data)")
        st.write("Google Sheets'teki anlık durumun birebir kopyasıdır.")
        st.dataframe(pf_df, use_container_width=True)
        st.caption("Not: Bu tablo sadece okuma amaçlıdır, değişiklikler bot çalıştığında güncellenir.")

    # --- TAB 3: İŞLEM GEÇMİŞİ (LOGLAR) ---
    with tab3:
        st.subheader("📜 Tüm İşlem Geçmişi (Logs)")
        hist_df = load_history_logs()
        if not hist_df.empty:
            # En son işlem en üstte görünsün diye ters çeviriyoruz
            st.dataframe(hist_df.iloc[::-1], use_container_width=True)
        else:
            st.info("Henüz kaydedilmiş geçmiş işlem yok veya 'Gecmis' sayfası boş.")
            st.markdown("⚠️ **Önemli:** Google Sheets'te `Gecmis` adında bir sayfa olduğundan emin olun.")
