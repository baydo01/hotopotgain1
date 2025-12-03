import streamlit as st
import pandas as pd
import numpy as np
import gspread
import plotly.express as px
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import time
import os

st.set_page_config(page_title="Hedge Fund AI: V16", layout="wide", page_icon="📡")

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
    # BURASI DÜZELTİLDİ: Değişkenleri doğru sırada alıyoruz
    pf_sheet, hist_sheet, _ = connect_sheet_services()
    
    if pf_sheet is None: return pd.DataFrame(), pd.DataFrame(), None
    
    pf_data = pf_sheet.get_all_records()
    df_pf = pd.DataFrame(pf_data)
    
    # Sayısal Dönüşüm
    cols = ["Miktar", "Son_Islem_Fiyati", "Nakit_Bakiye_USD", "Baslangic_USD", "Kaydedilen_Deger_USD"]
    for c in cols:
        if c in df_pf.columns: df_pf[c] = pd.to_numeric(df_pf[c].astype(str).str.replace(',', '.'), errors='coerce').fillna(0.0)
    
    try:
        hist_data = hist_sheet.get_all_records()
        df_hist = pd.DataFrame(hist_data)
    except: df_hist = pd.DataFrame()
    
    # KRİTİK DÜZELTME: update işlemini yapacak olan 'pf_sheet' nesnesini döndürüyoruz
    return df_pf, df_hist, pf_sheet 

# --- UI LOGIC ---
df_pf, df_hist, sheet_obj = load_data()

st.title("📡 Hedge Fund AI: V16 Live Monitor")

if not df_pf.empty:
    # TOP METRICS
    total_val = df_pf['Nakit_Bakiye_USD'].sum() + df_pf[df_pf['Durum']=='COIN']['Kaydedilen_Deger_USD'].sum()
    
    # Sütun kontrolü (Hata almamak için)
    last_update = df_pf['Bot_Son_Kontrol'].iloc[0] if 'Bot_Son_Kontrol' in df_pf.columns else "N/A"
    status = df_pf['Bot_Durum'].iloc[0] if 'Bot_Durum' in df_pf.columns else "Bilinmiyor"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Toplam Varlık", f"${total_val:.2f}")
    c2.metric("Son Güncelleme", str(last_update).split(' ')[1] if ' ' in str(last_update) else str(last_update))

    if "İşleniyor" in status:
        c3.warning(f"⚙️ {status}")
    elif "Hazır" in status:
        c3.success(f"🟢 {status}")
    else:
        c3.info(f"ℹ️ {status}")

    # MANUEL TETİKLEME
    if c4.button("🚨 ŞİMDİ ÇALIŞTIR"):
        if sheet_obj:
            df_pf['Bot_Trigger'] = "TRUE"
            # Güncelleme satırı (Artık doğru nesneyi kullanıyor)
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
    t1, t2 = st.tabs(["📊 Portföy", "📜 Geçmiş"])

    with t1:
        st.dataframe(df_pf, use_container_width=True)
        # Grafik
        if not df_pf.empty:
            chart_data = df_pf.copy()
            # Basit bir değer hesaplaması (Coin değeri + Nakit)
            chart_data['Değer'] = np.where(chart_data['Durum']=='COIN', chart_data['Kaydedilen_Deger_USD'], chart_data['Nakit_Bakiye_USD'])
            # Sadece değeri 0'dan büyük olanları göster
            chart_data = chart_data[chart_data['Değer'] > 0]
            
            if not chart_data.empty:
                fig = px.pie(chart_data, values='Değer', names='Ticker', title="Varlık Dağılımı", hole=0.4)
                st.plotly_chart(fig)
            else:
                st.info("Gösterilecek varlık verisi yok (Tüm bakiyeler 0).")

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
