import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import gspread
import os
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import pytz

# --- ML LIBS ---
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb
import plotly.express as px

warnings.filterwarnings("ignore")

# UI CONFIG
st.set_page_config(page_title="Hedge Fund AI: V14", layout="wide", page_icon="🌐")
st.markdown("""
<style>
    .main {background-color: #0E1117;}
    .header-box {background: linear-gradient(135deg, #000428 0%, #004e92 100%); padding: 25px; border-radius: 12px; border-left: 5px solid #00ff88; margin-bottom: 25px;}
    .header-title {font-size: 32px; font-weight: 700; color: #fff; margin:0;}
    .header-sub {font-size: 14px; color: #b0b0b0; margin-top: 5px;}
</style>
<div class="header-box">
    <div class="header-title">🌐 Hedge Fund AI: V14 (System Monitor)</div>
    <div class="header-sub">Bot Health Check • Transaction Logs • Real-time Analysis</div>
</div>
""", unsafe_allow_html=True)

SHEET_ID = "16zjLeps0t1P26OF3o7XQ-djEKKZtZX6t5lFxLmnsvpE"
CREDENTIALS_FILE = "service_account.json"
DATA_PERIOD = "730d"

# --- CONNECT ---
def connect_sheet_services():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = None
    if "gcp_service_account" in st.secrets:
        try: creds = ServiceAccountCredentials.from_json_keyfile_dict(dict(st.secrets["gcp_service_account"]), scope)
        except: pass
    elif os.path.exists(CREDENTIALS_FILE):
        creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, scope)
    if not creds: return None, None, None
    try:
        client = gspread.authorize(creds)
        spreadsheet = client.open_by_key(SHEET_ID)
        try: hist = spreadsheet.worksheet("Gecmis")
        except: hist = spreadsheet.add_worksheet("Gecmis", 1000, 6)
        return spreadsheet.sheet1, hist, client
    except: return None, None, None

def load_data():
    pf_sheet, hist_sheet, _ = connect_sheet_services()
    if pf_sheet is None: return pd.DataFrame(), pd.DataFrame()
    
    pf_data = pf_sheet.get_all_records()
    df_pf = pd.DataFrame(pf_data)
    cols = ["Miktar", "Son_Islem_Fiyati", "Nakit_Bakiye_USD", "Baslangic_USD", "Kaydedilen_Deger_USD"]
    for c in cols:
        if c in df_pf.columns: df_pf[c] = pd.to_numeric(df_pf[c].astype(str).str.replace(',', '.'), errors='coerce').fillna(0.0)
    
    try:
        hist_data = hist_sheet.get_all_records()
        df_hist = pd.DataFrame(hist_data)
    except: df_hist = pd.DataFrame()
    return df_pf, df_hist

# --- UI LOGIC ---
df_pf, df_hist = load_data()

# 1. SIDEBAR STATUS
with st.sidebar:
    st.header("🔍 Sistem Durumu")
    if not df_pf.empty and 'Bot_Son_Kontrol' in df_pf.columns:
        last_check_str = str(df_pf['Bot_Son_Kontrol'].iloc[0])
        st.write(f"Son Sinyal: **{last_check_str}**")
        st.success("🟢 Bot Aktif (Data Connected)")
    else:
        st.error("🔴 Bot Verisi Yok")
        
    st.divider()
    st.write("Varlık Dağılımı")
    if not df_pf.empty:
        st.dataframe(df_pf[df_pf['Durum']=='COIN'][['Ticker', 'Miktar']], hide_index=True)

# 2. MAIN TABS
tab1, tab2, tab3 = st.tabs(["📊 Analiz Masası", "📜 İşlem Geçmişi (Logs)", "💰 Portföy Detayı"])

with tab1:
    if not df_pf.empty:
        total_val = df_pf['Nakit_Bakiye_USD'].sum() + df_pf[df_pf['Durum']=='COIN']['Kaydedilen_Deger_USD'].sum()
        st.metric("Canlı Portföy Değeri", f"${total_val:.2f}")
        st.info("Bu ekran sadece izleme amaçlıdır. Bot arka planda otomatik çalışır.")

with tab2:
    st.subheader("📜 Bot İşlem Kayıtları (Server Logs)")
    if not df_hist.empty:
        st.dataframe(df_hist.iloc[::-1], use_container_width=True)
    else:
        st.info("Henüz kaydedilmiş işlem geçmişi yok.")

with tab3:
    st.subheader("Google Sheets Ham Verisi")
    st.dataframe(df_pf, use_container_width=True)
