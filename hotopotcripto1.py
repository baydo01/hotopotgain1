import streamlit as st
import pandas as pd
import numpy as np
import gspread
import plotly.express as px
import plotly.graph_objects as go
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime, timedelta
import time
import os
import yfinance as yf

# --- SCIENTIFIC LIBS ---
from sklearn.preprocessing import RobustScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

st.set_page_config(page_title="Model Audit Dashboard", layout="wide", page_icon="🏦")

# --- STYLING ---
st.markdown("""
<style>
    .metric-card {background-color: #1e2130; padding: 15px; border-radius: 10px; border-left: 5px solid #4caf50;}
    .audit-header {font-size: 24px; font-weight: bold; color: #ffffff; border-bottom: 2px solid #555;}
    .stDataFrame {font-size: 12px;}
</style>
""", unsafe_allow_html=True)

# --- 1. DATA CONNECTION LAYER ---
def connect_sheet_services():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = None
    if "gcp_service_account" in st.secrets:
        try: creds = ServiceAccountCredentials.from_json_keyfile_dict(dict(st.secrets["gcp_service_account"]), scope)
        except: pass
    if creds is None and os.path.exists("service_account.json"):
        try: creds = ServiceAccountCredentials.from_json_keyfile_name("service_account.json", scope)
        except: pass
    if creds is None: return None, None, None
    try:
        client = gspread.authorize(creds)
        SHEET_ID = "16zjLeps0t1P26OF3o7XQ-djEKKZtZX6t5lFxLmnsvpE"
        spreadsheet = client.open_by_key(SHEET_ID)
        try: hist = spreadsheet.worksheet("Gecmis")
        except: hist = spreadsheet.add_worksheet("Gecmis", 1000, 6)
        return spreadsheet.sheet1, hist, client
    except: return None, None, None

def load_data():
    pf_sheet, hist_sheet, sheet_obj = connect_sheet_services()
    if pf_sheet is None: return pd.DataFrame(), pd.DataFrame(), None
    
    # Portföy Verisi
    pf_data = pf_sheet.get_all_records()
    df_pf = pd.DataFrame(pf_data)
    cols = ["Miktar", "Son_Islem_Fiyati", "Nakit_Bakiye_USD", "Baslangic_USD", "Kaydedilen_Deger_USD"]
    for c in cols:
        if c in df_pf.columns: df_pf[c] = pd.to_numeric(df_pf[c].astype(str).str.replace(',', '.'), errors='coerce').fillna(0.0)
    
    # Geçmiş Verisi (Hata Korumalı)
    try: 
        hist_data = hist_sheet.get_all_values()
        if hist_data:
            expected_cols = ["Tarih", "Ticker", "Action", "Miktar", "Price", "Model"]
            # Sütun sayısı uyuşmazlığını önlemek için slice alıyoruz
            df_hist = pd.DataFrame(hist_data, columns=expected_cols[:len(hist_data[0])]) 
        else:
            df_hist = pd.DataFrame(columns=["Tarih", "Ticker", "Action", "Miktar", "Price", "Model"])
    except: 
        df_hist = pd.DataFrame(columns=["Tarih", "Ticker", "Action", "Miktar", "Price", "Model"])
    
    return df_pf, df_hist, pf_sheet

# --- 2. AUDIT ENGINE ---
class AuditBrain:
    def get_market_data(self, ticker):
        try:
            # Multi-level column hatasını önlemek için sağlamlaştırma
            df = yf.download(ticker, period="1y", interval="1d", progress=False)
            if df.empty: return None
            if isinstance(df.columns, pd.MultiIndex): 
                df.columns = df.columns.get_level_values(0)
            df.columns = [c.lower() for c in df.columns]
            return df
        except: return None

    def calculate_risk_metrics(self, df):
        # Eğer veri yoksa veya 30 günden azsa BOŞ sözlük döner
        if df is None or len(df) < 30: return {}
        try:
            ret = df['close'].pct_change().dropna()
            volatility = ret.std() * np.sqrt(252)
            var_95 = np.percentile(ret, 5)
            drawdown = (df['close'] / df['close'].cummax()) - 1
            return {"Volatilite (Yıllık)": volatility, "VaR (%95)": var_95, "Max Drawdown": drawdown.min()}
        except:
            return {}

    def simulate_models(self, df):
        if df is None or len(df) < 100: return []
        try:
            data = df.copy()
            data['rsi'] = 100 - (100 / (1 + data['close'].diff().clip(lower=0).rolling(14).mean() / data['close'].diff().clip(upper=0).abs().rolling(14).mean()))
            data['sma'] = data['close'].rolling(20).mean()
            data['target'] = (data['close'].shift(-1) > data['close']).astype(int)
            data = data.dropna()
            
            if len(data) < 50: return []
            
            X = data[['rsi', 'sma']].values; y = data['target'].values
            split = int(len(X)*0.8)
            X_tr, X_te = X[:split], X[split:]
            y_tr, y_te = y[:split], y[split:]
            scaler = RobustScaler()
            X_tr = scaler.fit_transform(X_tr); X_te = scaler.transform(X_te)
            
            m_xgb = xgb.XGBClassifier(n_estimators=50, max_depth=3).fit(X_tr, y_tr)
            p_xgb = m_xgb.predict_proba(X_te[-1].reshape(1,-1))[0][1]
            
            m_rf = RandomForestClassifier(n_estimators=50, max_depth=5).fit(X_tr, y_tr)
            p_rf = m_rf.predict_proba(X_te[-1].reshape(1,-1))[0][1]
            
            m_nn = MLPClassifier(hidden_layer_sizes=(32,16), max_iter=500).fit(X_tr, y_tr)
            p_nn = m_nn.predict_proba(X_te[-1].reshape(1,-1))[0][1]
            
            return [
                {"Model": "XGBoost", "Olasılık": p_xgb, "Karar": "AL" if p_xgb>0.55 else "SAT" if p_xgb<0.45 else "NÖTR"},
                {"Model": "Random Forest", "Olasılık": p_rf, "Karar": "AL" if p_rf>0.55 else "SAT" if p_rf<0.45 else "NÖTR"},
                {"Model": "Neural Network", "Olasılık": p_nn, "Karar": "AL" if p_nn>0.55 else "SAT" if p_nn<0.45 else "NÖTR"}
            ]
        except: return []

# --- 3. UI LAYOUT ---
df_pf, df_hist, pf_sheet_obj = load_data()
brain = AuditBrain()

with st.sidebar:
    st.title("🛡️ Model Denetim")
    st.markdown("---")
    if not df_pf.empty:
        # Hata korumalı veri okuma
        last = df_pf['Bot_Son_Kontrol'].iloc[0] if 'Bot_Son_Kontrol' in df_pf.columns else "---"
        status = df_pf['Bot_Durum'].iloc[0] if 'Bot_Durum' in df_pf.columns else "---"
        st.info(f"Güncelleme: {str(last).split(' ')[-1]}")
        if "Hazır" in str(status): st.success(status)
        else: st.warning(status)
        
        if st.button("🚨 Acil Durum Taraması", type="primary"):
            if pf_sheet_obj:
                df_pf['Bot_Trigger'] = "TRUE"
                try:
                    pf_sheet_obj.clear()
                    pf_sheet_obj.update([df_pf.columns.values.tolist()] + df_pf.astype(str).values.tolist())
                    st.toast("Sinyal Gönderildi!")
                    time.sleep(2)
                    st.rerun()
                except: st.error("Güncelleme Hatası")

st.markdown("<div class='audit-header'>🏦 Kurumsal Portföy Yönetim Paneli</div><br>", unsafe_allow_html=True)

if df_pf.empty:
    st.error("Veri yok. Secrets ayarlarını kontrol et.")
else:
    val_col = 'Kaydedilen_Deger_USD' if 'Kaydedilen_Deger_USD' in df_pf.columns else 'Baslangic_USD'
    total = df_pf['Nakit_Bakiye_USD'].sum() + df_pf[df_pf['Durum']=='COIN'][val_col].sum()
    cash_r = (df_pf['Nakit_Bakiye_USD'].sum() / total) * 100 if total > 0 else 0
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("NAV (Toplam)", f"${total:.2f}")
    k2.metric("Nakit Oranı", f"%{cash_r:.1f}")
    k3.metric("Model Mimarisi", "Ensemble (XGB+RF+NN)")
    k4.metric("Veri", "Live Feed", delta_color="off")

    t1, t2, t3 = st.tabs(["📊 Özet", "🔍 Derin Analiz", "📜 Kayıtlar"])

    with t1:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("Varlık Dağılımı")
            cd = df_pf.copy()
            cd['Val'] = np.where(cd['Durum']=='COIN', cd[val_col], cd['Nakit_Bakiye_USD'])
            if cd['Val'].sum() > 0:
                fig = px.pie(cd[cd['Val']>0], values='Val', names='Ticker', hole=0.4)
                st.plotly_chart(fig, use_container_width=True)
            else: st.info("Bakiye 0")
        with c2:
            st.subheader("Son İşlemler")
            if not df_hist.empty and 'Ticker' in df_hist.columns:
                st.dataframe(df_hist.tail(10)[['Ticker','Action','Price','Model']], use_container_width=True, hide_index=True)
            else:
                st.write("İşlem yok.")

    with t2:
        tk = st.selectbox("Varlık Seç:", df_pf['Ticker'].unique())
        if st.button("Analiz Et"):
            with st.spinner("Modeller Çalışıyor..."):
                md = brain.get_market_data(tk)
                
                # --- BURASI DÜZELTİLDİ (HATA KORUMASI) ---
                if md is not None and len(md) > 50:
                    # Grafik
                    md['SMA'] = md['close'].rolling(20).mean()
                    md['U'] = md['SMA'] + (md['close'].rolling(20).std()*2)
                    md['L'] = md['SMA'] - (md['close'].rolling(20).std()*2)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=md.index, y=md['close'], name='Fiyat'))
                    fig.add_trace(go.Scatter(x=md.index, y=md['U'], name='Üst Bant', line=dict(dash='dot', color='red')))
                    fig.add_trace(go.Scatter(x=md.index, y=md['L'], name='Alt Bant', line=dict(dash='dot', color='green')))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Risk Metrikleri
                    risk = brain.calculate_risk_metrics(md)
                    
                    # Eğer risk hesaplanabildiyse (Boş değilse)
                    if risk:
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Volatilite", f"%{risk['Volatilite (Yıllık)']*100:.2f}")
                        c2.metric("VaR (%95)", f"%{risk['VaR (%95)']*100:.2f}")
                        c3.metric("Max Drawdown", f"%{risk['Max Drawdown']*100:.2f}")
                    else:
                        st.warning("⚠️ Yetersiz veri nedeniyle risk metrikleri hesaplanamadı.")
                    
                    # Model Simülasyonu
                    sims = brain.simulate_models(md)
                    if sims:
                        st.table(pd.DataFrame(sims))
                    else:
                        st.warning("⚠️ Yetersiz veri nedeniyle model simülasyonu yapılamadı.")
                else:
                    st.error("⚠️ Veri çekilemedi veya veri seti çok kısa.")

    with t3:
        st.dataframe(df_pf, use_container_width=True)

    if "İşleniyor" in str(df_pf['Bot_Durum'].iloc[0]):
        time.sleep(10)
        st.rerun()
