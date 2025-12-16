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
from sklearn.neural_network import MLPClassifier # Neural Network
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb

st.set_page_config(page_title="Model Audit Dashboard", layout="wide", page_icon="🏦")

# --- STYLING (BANKACI MODU) ---
st.markdown("""
<style>
    .metric-card {background-color: #1e2130; padding: 15px; border-radius: 10px; border-left: 5px solid #4caf50;}
    .risk-card {background-color: #1e2130; padding: 15px; border-radius: 10px; border-left: 5px solid #ff9800;}
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
    
    pf_data = pf_sheet.get_all_records()
    df_pf = pd.DataFrame(pf_data)
    cols = ["Miktar", "Son_Islem_Fiyati", "Nakit_Bakiye_USD", "Baslangic_USD", "Kaydedilen_Deger_USD"]
    for c in cols:
        if c in df_pf.columns: df_pf[c] = pd.to_numeric(df_pf[c].astype(str).str.replace(',', '.'), errors='coerce').fillna(0.0)
    
    try: df_hist = pd.DataFrame(hist_sheet.get_all_records())
    except: df_hist = pd.DataFrame()
    return df_pf, df_hist, sheet_obj

# --- 2. AUDIT ENGINE (LIVE ANALYSIS) ---
# Bankacılar için anlık model simülasyonu yapan motor
class AuditBrain:
    def get_market_data(self, ticker):
        try:
            df = yf.download(ticker, period="1y", interval="1d", progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df.columns = [c.lower() for c in df.columns]
            return df
        except: return None

    def calculate_risk_metrics(self, df):
        if df is None or len(df) < 30: return {}
        ret = df['close'].pct_change().dropna()
        volatility = ret.std() * np.sqrt(252) # Yıllık Volatilite
        var_95 = np.percentile(ret, 5) # %95 Value at Risk
        drawdown = (df['close'] / df['close'].cummax()) - 1
        max_dd = drawdown.min()
        return {"Volatilite (Yıllık)": volatility, "VaR (%95)": var_95, "Max Drawdown": max_dd}

    def simulate_models(self, df):
        # Basitleştirilmiş Feature Engineering
        data = df.copy()
        data['rsi'] = 100 - (100 / (1 + data['close'].diff().clip(lower=0).rolling(14).mean() / data['close'].diff().clip(upper=0).abs().rolling(14).mean()))
        data['sma'] = data['close'].rolling(20).mean()
        data['target'] = (data['close'].shift(-1) > data['close']).astype(int)
        data = data.dropna()
        
        if len(data) < 100: return []
        
        X = data[['rsi', 'sma']].values
        y = data['target'].values
        split = int(len(X)*0.8)
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]
        scaler = RobustScaler()
        X_tr = scaler.fit_transform(X_tr); X_te = scaler.transform(X_te)
        
        # 1. XGBoost
        m_xgb = xgb.XGBClassifier(n_estimators=50, max_depth=3, eval_metric='logloss').fit(X_tr, y_tr)
        p_xgb = m_xgb.predict_proba(X_te[-1].reshape(1,-1))[0][1]
        
        # 2. Random Forest
        m_rf = RandomForestClassifier(n_estimators=50, max_depth=5).fit(X_tr, y_tr)
        p_rf = m_rf.predict_proba(X_te[-1].reshape(1,-1))[0][1]
        
        # 3. Neural Network (MLP)
        m_nn = MLPClassifier(hidden_layer_sizes=(32,16), max_iter=500, random_state=42).fit(X_tr, y_tr)
        p_nn = m_nn.predict_proba(X_te[-1].reshape(1,-1))[0][1]
        
        return [
            {"Model": "XGBoost (Gradient Boosting)", "Olasılık": p_xgb, "Karar": "AL" if p_xgb>0.55 else "SAT" if p_xgb<0.45 else "NÖTR"},
            {"Model": "Random Forest (Bagging)", "Olasılık": p_rf, "Karar": "AL" if p_rf>0.55 else "SAT" if p_rf<0.45 else "NÖTR"},
            {"Model": "Neural Network (MLP/Deep)", "Olasılık": p_nn, "Karar": "AL" if p_nn>0.55 else "SAT" if p_nn<0.45 else "NÖTR"}
        ]

# --- 3. UI LAYOUT ---
df_pf, df_hist, sheet_obj = load_data()
brain = AuditBrain()

# SIDEBAR
with st.sidebar:
    st.title("🛡️ Model Denetim")
    st.markdown("---")
    if not df_pf.empty:
        last_update = str(df_pf['Bot_Son_Kontrol'].iloc[0]).split(' ')[1]
        status = str(df_pf['Bot_Durum'].iloc[0])
        
        st.info(f"Son Güncelleme: {last_update}")
        if "Hazır" in status: st.success(f"Sistem Durumu: {status}")
        else: st.warning(f"Sistem Durumu: {status}")
        
        st.markdown("### ⚙️ Kontrol Paneli")
        if st.button("🚨 Acil Durum Taraması Başlat", type="primary"):
            if sheet_obj:
                df_pf['Bot_Trigger'] = "TRUE"
                sheet_obj.update([df_pf.columns.values.tolist()] + df_pf.astype(str).values.tolist())
                st.toast("Tetikleyici gönderildi. Cloud servisi bekleniyor...")
    
    st.markdown("---")
    st.caption("v18.0.2 Audit Edition | Powered by XGBoost & Neural Nets")

# MAIN PAGE
st.markdown("<div class='audit-header'>🏦 Kurumsal Portföy Yönetim Paneli</div><br>", unsafe_allow_html=True)

if df_pf.empty:
    st.error("Veri bağlantısı kurulamadı. Lütfen 'Secrets' ayarlarını kontrol edin.")
else:
    # KPI ROW
    total_equity = df_pf['Nakit_Bakiye_USD'].sum() + df_pf[df_pf['Durum']=='COIN']['Kaydedilen_Deger_USD'].sum()
    cash_ratio = (df_pf['Nakit_Bakiye_USD'].sum() / total_equity) * 100
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Toplam Varlık (NAV)", f"${total_equity:.2f}", delta_color="normal")
    k2.metric("Nakit Oranı", f"%{cash_ratio:.1f}")
    k3.metric("Aktif Model Sayısı", "3 (XGB+RF+NN)")
    k4.metric("Veri Kaynağı", "Google Sheets Live", delta_color="off")

    # TABS
    tab1, tab2, tab3 = st.tabs(["📊 Yönetici Özeti", "🔍 Model Denetimi (Deep Dive)", "📜 Ham Veriler"])

    with tab1:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("Varlık Dağılımı")
            chart_data = df_pf.copy()
            chart_data['Değer'] = np.where(chart_data['Durum']=='COIN', chart_data['Kaydedilen_Deger_USD'], chart_data['Nakit_Bakiye_USD'])
            fig = px.pie(chart_data[chart_data['Değer']>0], values='Değer', names='Ticker', hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig, use_container_width=True)
        
        with c2:
            st.subheader("Son Aksiyonlar")
            if not df_hist.empty:
                st.dataframe(df_hist.tail(5)[['Ticker','Action','Price','Model']], use_container_width=True, hide_index=True)

    with tab2:
        st.info("Bu modül, seçilen varlık için anlık olarak yapay zeka modellerini çalıştırır ve risk analizi yapar.")
        selected_ticker = st.selectbox("Denetlenecek Varlık Seçin:", df_pf['Ticker'].unique())
        
        if st.button("🔍 Detaylı Analiz Başlat"):
            with st.spinner(f"{selected_ticker} için Neural Network ve XGBoost modelleri çalıştırılıyor..."):
                market_data = brain.get_market_data(selected_ticker)
                
                if market_data is not None:
                    # 1. PRICE CHART with Risk Bands
                    st.subheader("Fiyat ve Volatilite Bandı")
                    market_data['SMA'] = market_data['close'].rolling(20).mean()
                    market_data['Upper'] = market_data['SMA'] + (market_data['close'].rolling(20).std() * 2)
                    market_data['Lower'] = market_data['SMA'] - (market_data['close'].rolling(20).std() * 2)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=market_data.index, y=market_data['close'], name='Fiyat', line=dict(color='white')))
                    fig.add_trace(go.Scatter(x=market_data.index, y=market_data['Upper'], name='Risk Üst', line=dict(dash='dot', color='red')))
                    fig.add_trace(go.Scatter(x=market_data.index, y=market_data['Lower'], name='Risk Alt', line=dict(dash='dot', color='green')))
                    fig.update_layout(template="plotly_dark", height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 2. RISK METRICS
                    risk = brain.calculate_risk_metrics(market_data)
                    r1, r2, r3 = st.columns(3)
                    r1.metric("Yıllık Volatilite (Risk)", f"%{risk['Volatilite (Yıllık)']*100:.2f}")
                    r2.metric("VaR (%95 Güven Aralığı)", f"%{risk['VaR (%95)']*100:.2f}")
                    r3.metric("Max Drawdown (Çöküş)", f"%{risk['Max Drawdown']*100:.2f}")
                    
                    # 3. MODEL CONSENSUS
                    st.subheader("🤖 Yapay Zeka Konsensüsü")
                    sim_results = brain.simulate_models(market_data)
                    sim_df = pd.DataFrame(sim_results)
                    
                    # Renkli Tablo
                    def color_decision(val):
                        color = '#4caf50' if val == 'AL' else '#f44336' if val == 'SAT' else '#ff9800'
                        return f'color: {color}; font-weight: bold'
                    
                    st.table(sim_df.style.applymap(color_decision, subset=['Karar']))
                    
                    st.caption("*Not: Bu veriler anlık simülasyon sonucudur. Botun ana kararı Sheets'te kayıtlıdır.*")
                    
                else:
                    st.error("Veri çekilemedi.")

    with tab3:
        st.subheader("Veritabanı Kayıtları (Google Sheets)")
        st.dataframe(df_pf, use_container_width=True)
        st.download_button("Excel Olarak İndir", df_pf.to_csv(), "portfolio_audit.csv")

    # Auto-Refresh Mechanism
    if "İşleniyor" in str(df_pf['Bot_Durum'].iloc[0]):
        time.sleep(10)
        st.rerun()
