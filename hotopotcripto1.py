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
from scipy.stats import norm # İstatistiksel PD hesabı için

# --- SCIENTIFIC LIBS ---
from sklearn.preprocessing import RobustScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

st.set_page_config(page_title="Model Audit & Risk Dashboard", layout="wide", page_icon="🏦")

# --- STYLING (BANKACI MODU) ---
st.markdown("""
<style>
    .metric-card {background-color: #1e2130; padding: 15px; border-radius: 10px; border-left: 5px solid #4caf50;}
    .risk-card {background-color: #262730; padding: 10px; border-radius: 5px; border-left: 4px solid #d32f2f;}
    .audit-header {font-size: 24px; font-weight: bold; color: #ffffff; border-bottom: 2px solid #555; padding-bottom: 10px;}
    .stDataFrame {font-size: 12px;}
</style>
""", unsafe_allow_html=True)

# --- 1. DATA CONNECTION ---
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

# --- 2. AUDIT & RISK ENGINE ---
class AuditBrain:
    def get_market_data(self, ticker):
        try:
            df = yf.download(ticker, period="1y", interval="1d", progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df.columns = [c.lower() for c in df.columns]
            return df
        except: return None

    # --- YENİ EKLENEN KREDİ RİSKİ MODÜLÜ (IFRS 9 / BASEL) ---
    def calculate_credit_risk_metrics(self, df, exposure_usd):
        """
        Merton Modeli benzeri yaklaşımla PD, LGD, ECL hesaplar.
        Varsayım: Varlık fiyatı %20 düşerse 'Temerrüt' (Default) sayılır.
        """
        if df is None or len(df) < 30: return {}
        
        current_price = df['close'].iloc[-1]
        returns = df['close'].pct_change().dropna()
        
        # 1. PD (Probability of Default)
        # Mevcut fiyatın %20 altına düşme olasılığı (1 Aylık ufukta)
        threshold_price = current_price * 0.80 # %20 Drop barrier
        volatility_daily = returns.std()
        volatility_monthly = volatility_daily * np.sqrt(21)
        
        # Z-Score hesabı (Distance to Default)
        ln_returns = np.log(current_price / threshold_price)
        d2 = ln_returns / volatility_monthly
        pd_value = 1 - norm.cdf(d2) # Normal dağılımdan olasılık
        
        # 2. EAD (Exposure at Default)
        ead_value = exposure_usd # Şu anki risk tutarı
        
        # 3. LGD (Loss Given Default)
        # Tarihsel olarak en kötü aylık düşüşü LGD olarak kabul edelim (Muhafazakar yaklaşım)
        rolling_max = df['close'].rolling(30).max()
        drawdown = (df['close'] / rolling_max) - 1
        lgd_value = abs(drawdown.min()) # En kötü düşüş oranı (örn. 0.45)
        if lgd_value < 0.2: lgd_value = 0.45 # Basel standartlarına yakın bir taban değer
        
        # 4. ECL (Expected Credit Loss) - Beklenen Zarar Karşılığı
        ecl_value = pd_value * ead_value * lgd_value
        
        return {
            "PD (%)": pd_value * 100,
            "EAD ($)": ead_value,
            "LGD (%)": lgd_value * 100,
            "ECL ($)": ecl_value,
            "Risk Skoru": "Yüksek" if pd_value > 0.10 else "Orta" if pd_value > 0.05 else "Düşük"
        }

    def simulate_models(self, df):
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
        
        m_xgb = xgb.XGBClassifier(n_estimators=50, max_depth=3, eval_metric='logloss').fit(X_tr, y_tr)
        p_xgb = m_xgb.predict_proba(X_te[-1].reshape(1,-1))[0][1]
        
        m_rf = RandomForestClassifier(n_estimators=50, max_depth=5).fit(X_tr, y_tr)
        p_rf = m_rf.predict_proba(X_te[-1].reshape(1,-1))[0][1]
        
        m_nn = MLPClassifier(hidden_layer_sizes=(32,16), max_iter=500, random_state=42).fit(X_tr, y_tr)
        p_nn = m_nn.predict_proba(X_te[-1].reshape(1,-1))[0][1]
        
        return [
            {"Model": "XGBoost", "Olasılık": p_xgb, "Karar": "AL" if p_xgb>0.55 else "SAT" if p_xgb<0.45 else "NÖTR"},
            {"Model": "Random Forest", "Olasılık": p_rf, "Karar": "AL" if p_rf>0.55 else "SAT" if p_rf<0.45 else "NÖTR"},
            {"Model": "Neural Network", "Olasılık": p_nn, "Karar": "AL" if p_nn>0.55 else "SAT" if p_nn<0.45 else "NÖTR"}
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
        if "Hazır" in status: st.success(f"{status}")
        else: st.warning(f"{status}")
        
        if st.button("🚨 Acil Durum Taraması"):
            if sheet_obj:
                df_pf['Bot_Trigger'] = "TRUE"
                sheet_obj.update([df_pf.columns.values.tolist()] + df_pf.astype(str).values.tolist())
                st.toast("Sinyal gönderildi...")
    st.caption("v19.0 Credit Risk Edition")

# MAIN PAGE
st.markdown("<div class='audit-header'>🏦 Kurumsal Risk Yönetim Paneli (Basel III / IFRS 9)</div><br>", unsafe_allow_html=True)

if df_pf.empty:
    st.error("Veri bağlantısı kurulamadı. Lütfen 'Secrets' ayarlarını kontrol edin.")
else:
    # KPI ROW
    total_equity = df_pf['Nakit_Bakiye_USD'].sum() + df_pf[df_pf['Durum']=='COIN']['Kaydedilen_Deger_USD'].sum()
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Toplam Portföy (EAD)", f"${total_equity:.2f}", delta_color="normal")
    k2.metric("Aktif Varlıklar", f"{len(df_pf[df_pf['Durum']=='COIN'])}")
    k3.metric("Model Mimarisi", "Ensemble (XGB+NN)")
    k4.metric("Risk Metodolojisi", "Merton Model / VaR")

    # TABS
    tab1, tab2, tab3 = st.tabs(["📊 Basel III Risk Parametreleri", "🧠 Model Denetimi (AI)", "📜 Ham Veriler"])

    with tab1:
        st.info("💡 Bu bölüm, portföydeki varlıkları 'Kredi Riski' perspektifiyle analiz eder (PD, LGD, ECL).")
        
        # Sadece COIN olanları veya Cash olmayanları seçelim, yoksa sembolik seçelim
        tickers = df_pf['Ticker'].unique()
        selected_ticker_risk = st.selectbox("Risk Analizi İçin Varlık Seçin:", tickers, key="risk_select")
        
        if st.button("Risk Parametrelerini Hesapla (PD/LGD/ECL)"):
            with st.spinner("Merton Modeli çalıştırılıyor..."):
                market_data = brain.get_market_data(selected_ticker_risk)
                
                # EAD hesapla (O anki miktar * fiyat veya nakit)
                row = df_pf[df_pf['Ticker']==selected_ticker_risk].iloc[0]
                exposure = float(row['Kaydedilen_Deger_USD']) if row['Durum']=='COIN' else float(row['Nakit_Bakiye_USD'])
                if exposure == 0: exposure = 100.0 # Demo amaçlı varsayılan maruziyet
                
                risk_metrics = brain.calculate_credit_risk_metrics(market_data, exposure)
                
                if risk_metrics:
                    # RİSK KARTLARI
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.markdown('<div class="risk-card"><h5>PD (Probability of Default)</h5><h3>%{:.2f}</h3><p>İflas Olasılığı (1 Ay)</p></div>'.format(risk_metrics['PD (%)']), unsafe_allow_html=True)
                    with c2:
                        st.markdown(f'<div class="risk-card"><h5>LGD (Loss Given Default)</h5><h3>%{risk_metrics["LGD (%)"]:.2f}</h3><p>Temerrüt Kayıp Oranı</p></div>', unsafe_allow_html=True)
                    with c3:
                        st.markdown(f'<div class="risk-card"><h5>EAD (Exposure)</h5><h3>${risk_metrics["EAD ($)"]:.2f}</h3><p>Maruz Kalınan Tutar</p></div>', unsafe_allow_html=True)
                    with c4:
                        st.markdown(f'<div class="risk-card"><h5>ECL (Expected Loss)</h5><h3>${risk_metrics["ECL ($)"]:.4f}</h3><p>Beklenen Kredi Zararı</p></div>', unsafe_allow_html=True)
                    
                    st.divider()
                    
                    # Risk Gauge Chart
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = risk_metrics['PD (%)'],
                        title = {'text': "Risk Skoru (PD)"},
                        gauge = {
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 5], 'color': "green"},
                                {'range': [5, 20], 'color': "orange"},
                                {'range': [20, 100], 'color': "red"}],
                        }))
                    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.warning(f"📝 **Denetim Notu:** Bu varlık için hesaplanan Beklenen Kredi Zararı (ECL) **${risk_metrics['ECL ($)'].2f}** seviyesindedir. IFRS 9 standardına göre bu tutar kadar karşılık ayrılması önerilir.")

    with tab2:
        st.subheader("Yapay Zeka Karar Destek Mekanizması")
        sel_model_ticker = st.selectbox("Model Simülasyonu İçin Varlık:", df_pf['Ticker'].unique(), key="model_select")
        
        if st.button("Modelleri Çalıştır (XGBoost vs NeuralNet)"):
             with st.spinner("Algoritmalar yarışıyor..."):
                m_data = brain.get_market_data(sel_model_ticker)
                sim_res = brain.simulate_models(m_data)
                sim_df = pd.DataFrame(sim_res)
                
                # Stilize Tablo
                def color_decision(val):
                    color = '#4caf50' if val == 'AL' else '#f44336' if val == 'SAT' else '#ff9800'
                    return f'color: {color}; font-weight: bold'
                
                st.table(sim_df.style.applymap(color_decision, subset=['Karar']))
                
                # Fiyat Grafiği
                fig = px.line(m_data, x=m_data.index, y='close', title=f"{sel_model_ticker} Fiyat Analizi")
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.dataframe(df_pf, use_container_width=True)

    # Auto-Refresh
    if "İşleniyor" in str(df_pf['Bot_Durum'].iloc[0]):
        time.sleep(10)
        st.rerun()
