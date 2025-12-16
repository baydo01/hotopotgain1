import streamlit as st
import pandas as pd
import numpy as np
import gspread
import plotly.express as px
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import time
import os
import yfinance as yf

st.set_page_config(page_title="Hedge Fund AI: V18", layout="wide", page_icon="📡")

# --- CONNECT (HİBRİT) ---
def connect_sheet_services():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = None
    
    # 1. Cloud Secrets
    if "gcp_service_account" in st.secrets:
        try:
            creds = ServiceAccountCredentials.from_json_keyfile_dict(dict(st.secrets["gcp_service_account"]), scope)
        except Exception as e:
            st.error(f"Cloud Secrets Hatası: {e}")

    # 2. Yerel Dosya
    if creds is None and os.path.exists("service_account.json"):
        try:
            creds = ServiceAccountCredentials.from_json_keyfile_name("service_account.json", scope)
        except Exception as e:
            st.error(f"Yerel Dosya Hatası: {e}")

    if creds is None:
        st.error("⚠️ BAĞLANTI YOK! 'service_account.json' dosyası veya Secrets bulunamadı.")
        return None, None, None
    
    try:
        client = gspread.authorize(creds)
        SHEET_ID = "16zjLeps0t1P26OF3o7XQ-djEKKZtZX6t5lFxLmnsvpE"
        spreadsheet = client.open_by_key(SHEET_ID)
        try: hist = spreadsheet.worksheet("Gecmis")
        except: hist = spreadsheet.add_worksheet("Gecmis", 1000, 6)
        
        # DÖNÜŞ: (Portfolio Sayfası, Geçmiş Sayfası, Client)
        return spreadsheet.sheet1, hist, client
    except Exception as e:
        st.error(f"Google Sheets Bağlantı Hatası: {e}")
        return None, None, None

def load_data():
    pf_sheet, hist_sheet, _ = connect_sheet_services()
    
    if pf_sheet is None: return pd.DataFrame(), pd.DataFrame(), None
    
    pf_data = pf_sheet.get_all_records()
    df_pf = pd.DataFrame(pf_data)
    
    # Sayısal Dönüşüm
    cols = ["Miktar", "Son_Islem_Fiyati", "Nakit_Bakiye_USD", "Baslangic_USD", "Kaydedilen_Deger_USD", "Volatilite"]
    for c in cols:
        if c in df_pf.columns: df_pf[c] = pd.to_numeric(df_pf[c].astype(str).str.replace(',', '.'), errors='coerce').fillna(0.0)
    
    if 'Volatilite' not in df_pf.columns: df_pf['Volatilite'] = 0.0

    try:
        hist_data = hist_sheet.get_all_records()
        df_hist = pd.DataFrame(hist_data)
    except: df_hist = pd.DataFrame()
    
    return df_pf, df_hist, pf_sheet 

# --- VOLATİLİTE HESAPLAMA (YENİ FONKSİYON) ---
def calculate_and_update_volatility(sheet, df):
    """
    Tüm tickerlar için 1 aylık volatiliteyi hesaplar ve Sheets'e yazar.
    """
    status_bar = st.progress(0)
    status_text = st.empty()
    
    new_vols = []
    tickers = df['Ticker'].tolist()
    
    for i, ticker in enumerate(tickers):
        status_text.text(f"Analiz ediliyor: {ticker}...")
        try:
            if "USD" not in ticker and len(ticker) < 6:
                new_vols.append(0.0)
                continue
                
            # Son 2 ayın verisini çekip 20 günlük volatiliteye bakıyoruz
            hist = yf.download(ticker, period="2mo", progress=False)
            if len(hist) > 20:
                # Log Return
                hist['Log_Ret'] = np.log(hist['Close'] / hist['Close'].shift(1))
                # Volatilite (Standart Sapma)
                vol = hist['Log_Ret'].rolling(window=20).std().iloc[-1]
                new_vols.append(float(vol) if not pd.isna(vol) else 0.0)
            else:
                new_vols.append(0.0)
        except Exception as e:
            new_vols.append(0.0)
        
        status_bar.progress((i + 1) / len(tickers))
        
    df['Volatilite'] = new_vols
    
    # Sheets'e Yazma
    try:
        # Eğer Volatilite sütunu yoksa oluşturulmasını garanti edelim ama update ile tüm tabloyu yazmak daha güvenli
        sheet.update([df.columns.values.tolist()] + df.astype(str).values.tolist())
        st.success("✅ Volatilite değerleri Sheets'e işlendi!")
    except Exception as e:
        st.error(f"Sheets Güncelleme Hatası: {e}")
        
    status_text.empty()
    status_bar.empty()
    return df

# --- UI LOGIC ---
df_pf, df_hist, sheet_obj = load_data()

st.title("📡 Hedge Fund AI: V18 Live Monitor")

if not df_pf.empty:
    # TOP METRICS
    total_val = df_pf['Nakit_Bakiye_USD'].sum() + df_pf[df_pf['Durum']=='COIN']['Kaydedilen_Deger_USD'].sum()
    
    last_update = df_pf['Bot_Son_Kontrol'].iloc[0] if 'Bot_Son_Kontrol' in df_pf.columns else "N/A"
    status = df_pf['Bot_Durum'].iloc[0] if 'Bot_Durum' in df_pf.columns else "Bilinmiyor"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Toplam Varlık", f"${total_val:.2f}")
    c2.metric("Son Güncelleme", str(last_update).split(' ')[1] if ' ' in str(last_update) else str(last_update))

    if "İşleniyor" in status:
        c3.warning(f"⚙️ {status}")
    elif "Hazır" in status or "Bitti" in status:
        c3.success(f"🟢 {status}")
    else:
        c3.info(f"ℹ️ {status}")

    # MANUEL TETİKLEME ALANI
    with c4:
        st.write("Kontrol Paneli")
        if st.button("📊 1. Volatiliteyi Güncelle"):
            if sheet_obj:
                df_pf = calculate_and_update_volatility(sheet_obj, df_pf)
                st.rerun()
            else:
                st.error("Sheet bağlantısı yok.")
                
        if st.button("🚨 2. BOTU ÇALIŞTIR"):
            if sheet_obj:
                df_pf['Bot_Trigger'] = "TRUE"
                try:
                    sheet_obj.update([df_pf.columns.values.tolist()] + df_pf.astype(str).values.tolist())
                    st.toast("Sinyal gönderildi! Bot bekleniyor...")
                    time.sleep(2)
                    st.rerun()
                except Exception as e:
                    st.error(f"Güncelleme Hatası: {e}")
            else:
                st.error("Sheet bağlantısı kopuk.")

    # --- TABS ---
    t1, t2 = st.tabs(["📊 Portföy & Risk", "📜 Geçmiş"])

    with t1:
        # Volatiliteyi renklendirerek gösterelim
        st.dataframe(df_pf.style.background_gradient(subset=['Volatilite'], cmap='Reds'), use_container_width=True)
        
        if not df_pf.empty:
            c_chart1, c_chart2 = st.columns(2)
            with c_chart1:
                chart_data = df_pf.copy()
                chart_data['Değer'] = np.where(chart_data['Durum']=='COIN', chart_data['Kaydedilen_Deger_USD'], chart_data['Nakit_Bakiye_USD'])
                chart_data = chart_data[chart_data['Değer'] > 0]
                if not chart_data.empty:
                    fig = px.pie(chart_data, values='Değer', names='Ticker', title="Varlık Dağılımı", hole=0.4)
                    st.plotly_chart(fig, use_container_width=True)
            
            with c_chart2:
                # Volatilite Bar Chart 
                if 'Volatilite' in df_pf.columns:
                    fig_vol = px.bar(df_pf[df_pf['Volatilite']>0], x='Ticker', y='Volatilite', title="Risk (Volatilite) Analizi", color='Volatilite')
                    st.plotly_chart(fig_vol, use_container_width=True)

    with t2:
        if not df_hist.empty:
            st.dataframe(df_hist.iloc[::-1], use_container_width=True)
        else:
            st.write("İşlem geçmişi yok.")

    # Auto-Refresh
    if "İşleniyor" in status:
        time.sleep(10)
        st.rerun()
else:
    st.warning("Veri yüklenemedi. Bağlantı ayarlarını kontrol et.")
