import streamlit as st
import pandas as pd
import numpy as np
import gspread
import plotly.express as px
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import time

st.set_page_config(page_title="Hedge Fund AI: V16", layout="wide", page_icon="📡")

# --- CONNECT ---
def load_data():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("service_account.json", scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_key("16zjLeps0t1P26OF3o7XQ-djEKKZtZX6t5lFxLmnsvpE")
    
    pf = pd.DataFrame(sheet.sheet1.get_all_records())
    try: hist = pd.DataFrame(sheet.worksheet("Gecmis").get_all_records())
    except: hist = pd.DataFrame()
    
    # Numeric conversion
    cols = ["Miktar", "Nakit_Bakiye_USD", "Kaydedilen_Deger_USD"]
    for c in cols:
        if c in pf.columns: pf[c] = pd.to_numeric(pf[c].astype(str).str.replace(',', '.'), errors='coerce').fillna(0.0)
        
    return pf, hist, sheet.sheet1

# --- UI ---
pf, hist, sheet_obj = load_data()

st.title("📡 Hedge Fund AI: V16 Live Monitor")

# TOP METRICS
total_val = pf['Nakit_Bakiye_USD'].sum() + pf[pf['Durum']=='COIN']['Kaydedilen_Deger_USD'].sum()
last_update = pf['Bot_Son_Kontrol'].iloc[0] if 'Bot_Son_Kontrol' in pf.columns else "N/A"
status = pf['Bot_Durum'].iloc[0] if 'Bot_Durum' in pf.columns else "Bilinmiyor"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Toplam Varlık", f"${total_val:.2f}")
c2.metric("Son Güncelleme", last_update.split(' ')[1] if ' ' in str(last_update) else last_update)

# Durum Rozeti
if "İşleniyor" in status:
    c3.warning(f"⚙️ {status}")
elif "Hazır" in status:
    c3.success(f"🟢 {status}")
else:
    c3.info(f"ℹ️ {status}")

# MANUEL TETİKLEME
if c4.button("🚨 ŞİMDİ ÇALIŞTIR"):
    pf['Bot_Trigger'] = "TRUE"
    # Sadece Trigger sütununu güncelle (Hız için)
    sheet_obj.update([pf.columns.values.tolist()] + pf.astype(str).values.tolist())
    st.toast("Sinyal gönderildi! Bot 15sn içinde başlayacak...")
    time.sleep(2)
    st.rerun()

# --- MAIN VIEW ---
t1, t2 = st.tabs(["📊 Portföy", "📜 Geçmiş"])

with t1:
    st.dataframe(pf, use_container_width=True)
    
    # Grafik
    if not pf.empty:
        chart_data = pf.copy()
        chart_data['Değer'] = np.where(chart_data['Durum']=='COIN', chart_data['Kaydedilen_Deger_USD'], chart_data['Nakit_Bakiye_USD'])
        fig = px.pie(chart_data, values='Değer', names='Ticker', title="Portföy Dağılımı", hole=0.4)
        st.plotly_chart(fig)

with t2:
    if not hist.empty:
        st.dataframe(hist.iloc[::-1], use_container_width=True)
    else:
        st.write("İşlem geçmişi yok.")

# Auto-Refresh (Her 30 saniyede bir sayfayı yeniler ki durumu gör)
if "İşleniyor" in status:
    time.sleep(10)
    st.rerun()
