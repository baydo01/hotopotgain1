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
st.set_page_config(page_title="Hedge Fund AI: V15 Control", layout="wide", page_icon="🎮")
st.markdown("""
<style>
    .main {background-color: #0E1117;}
    .header-box {background: linear-gradient(135deg, #1f4037 0%, #99f2c8 100%); padding: 25px; border-radius: 12px; border-left: 5px solid #fff; margin-bottom: 25px;}
    .header-title {font-size: 32px; font-weight: 700; color: #fff; margin:0;}
    .header-sub {font-size: 14px; color: #f0f0f0; margin-top: 5px;}
    .stButton>button {width: 100%; border-radius: 5px; font-weight: bold;}
</style>
<div class="header-box">
    <div class="header-title">🎮 Hedge Fund AI: V15 (Remote Control)</div>
    <div class="header-sub">Manual Trigger • Live Monitor • Auto-Pilot Status</div>
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
    if pf_sheet is None: return pd.DataFrame(), pd.DataFrame(), None
    
    pf_data = pf_sheet.get_all_records()
    df_pf = pd.DataFrame(pf_data)
    cols = ["Miktar", "Son_Islem_Fiyati", "Nakit_Bakiye_USD", "Baslangic_USD", "Kaydedilen_Deger_USD"]
    for c in cols:
        if c in df_pf.columns: df_pf[c] = pd.to_numeric(df_pf[c].astype(str).str.replace(',', '.'), errors='coerce').fillna(0.0)
    
    try:
        hist_data = hist_sheet.get_all_records()
        df_hist = pd.DataFrame(hist_data)
    except: df_hist = pd.DataFrame()
    
    return df_pf, df_hist, pf_sheet

# --- ACTIONS ---
def trigger_bot(sheet, df):
    if sheet is not None and not df.empty:
        # Tüm satırlara (veya sadece ilkine) Trigger = TRUE yaz
        # Pandas üzerinde güncelle
        df['Bot_Trigger'] = "TRUE"
        
        # Sheet'e geri yükle
        df_exp = df.copy().astype(str)
        sheet.clear()
        sheet.update([df_exp.columns.values.tolist()] + df_exp.values.tolist())
        return True
    return False

# --- UI LOGIC ---
df_pf, df_hist, pf_sheet_obj = load_data()

# 1. SIDEBAR (KUMANDA)
with st.sidebar:
    st.header("🎮 Komuta Merkezi")
    
    # DURUM
    if not df_pf.empty and 'Bot_Son_Kontrol' in df_pf.columns:
        last_check = str(df_pf['Bot_Son_Kontrol'].iloc[0])
        st.info(f"🕒 Son Sinyal: {last_check}")
        
        # Trigger Durumu
        trig_state = str(df_pf['Bot_Trigger'].iloc[0]) if 'Bot_Trigger' in df_pf.columns else "FALSE"
        if trig_state == "TRUE":
            st.warning("⚠️ Sinyal Gönderildi! Bot bekleniyor...")
        else:
            st.success("🟢 Bot Beklemede (Ready)")
    else:
        st.error("🔴 Bağlantı Yok")
        
    st.divider()
    
    # BUTON
    if st.button("🚨 BOTU ZORLA ÇALIŞTIR", type="primary"):
        if trigger_bot(pf_sheet_obj, df_pf):
            st.toast("Sinyal Gönderildi! Bot 30sn içinde çalışacak.")
            st.rerun()
        else:
            st.error("Hata: Sheet güncellenemedi.")
            
    st.caption("Not: Butona basınca Google Sheets'e sinyal gider. Bot bunu görür ve çalışır.")

# 2. MAIN TABS
tab1, tab2, tab3 = st.tabs(["💰 Portföy Özeti", "📜 İşlem Kayıtları", "📊 Detaylı Veri"])

with tab1:
    if not df_pf.empty:
        total = df_pf['Nakit_Bakiye_USD'].sum() + df_pf[df_pf['Durum']=='COIN']['Kaydedilen_Deger_USD'].sum()
        c1, c2 = st.columns(2)
        c1.metric("Toplam Varlık", f"${total:.2f}")
        c1.caption("Otomatik güncellenir.")
        
        # Basit Pasta Grafik
        coin_val = df_pf[df_pf['Durum']=='COIN']['Kaydedilen_Deger_USD'].sum()
        cash_val = df_pf['Nakit_Bakiye_USD'].sum()
        
        fig = px.pie(names=['COIN', 'CASH'], values=[coin_val, cash_val], title="Varlık Dağılımı", hole=0.4)
        c2.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("İşlem Geçmişi")
    if not df_hist.empty:
        st.dataframe(df_hist.iloc[::-1], use_container_width=True)
    else:
        st.info("Kayıt yok.")

with tab3:
    st.dataframe(df_pf, use_container_width=True)
