import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import pytz
import time
import warnings
import json
import os

warnings.filterwarnings("ignore")

# ==============================
#   AYARLAR
# ==============================
SHEET_ID = "16zjLeps0t1P26OF3o7XQ-djEKKZtZX6t5lFxLmnsvpE"
SHEET_NAME = "hedge_fund_portfolio"
START_DATE = "2018-01-01"

COIN_LIST = [
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD",
    "XRP-USD", "DOGE-USD", "ADA-USD", "AVAX-USD",
    "DOT-USD", "LINK-USD", "ICP-USD"
]

# ==============================
#  GOOGLE SHEETS BAĞLANTISI
# ==============================
def connect_sheet():
    scope = ["https://spreadsheets.google.com/feeds",
             "https://www.googleapis.com/auth/drive"]

    creds_json_str = os.environ.get("GCP_CREDS")
    if not creds_json_str:
        st.error("❌ Google Sheet bağlantısı için GCP_CREDS eksik.")
        return None

    creds_dict = json.loads(creds_json_str)
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)

    try:
        sheet = client.open_by_key(SHEET_ID).worksheet(SHEET_NAME)
        return sheet
    except:
        # Sheet yoksa oluştur
        sh = client.open_by_key(SHEET_ID)
        sheet = sh.add_worksheet(SHEET_NAME, rows="200", cols="20")
        sheet.append_row([
            "Ticker", "Durum", "Miktar", "Son_Islem_Fiyati",
            "Nakit_Bakiye_USD", "Baslangic_USD",
            "Kaydedilen_Deger_USD", "Son_Islem_Log",
            "Son_Islem_Zamani"
        ])
        return sheet

def load_portfolio():
    sheet = connect_sheet()
    records = sheet.get_all_records()
    df = pd.DataFrame(records)

    if df.empty:
        df = pd.DataFrame({
            "Ticker": COIN_LIST,
            "Durum": ["CASH"] * len(COIN_LIST),
            "Miktar": [0.0] * len(COIN_LIST),
            "Son_Islem_Fiyati": [0.0] * len(COIN_LIST),
            "Nakit_Bakiye_USD": [10.0] * len(COIN_LIST),
            "Baslangic_USD": [10.0] * len(COIN_LIST),
            "Kaydedilen_Deger_USD": [10.0] * len(COIN_LIST),
            "Son_Islem_Log": ["KURULUM"] * len(COIN_LIST),
            "Son_Islem_Zamani": ["-"] * len(COIN_LIST)
        })
        save_portfolio(df, sheet)

    numeric_cols = [
        "Miktar", "Son_Islem_Fiyati", "Nakit_Bakiye_USD",
        "Baslangic_USD", "Kaydedilen_Deger_USD"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df, sheet

def save_portfolio(df, sheet):
    sheet.clear()
    sheet.append_row(df.columns.tolist())
    for row in df.values.tolist():
        sheet.append_row(row)

# ==============================
#  PUANLAMA MOTORU (7 KRİTER)
# ==============================
def calculate_custom_score(df):
    if len(df) < 366:
        return pd.Series(0, index=df.index)

    daily_steps = np.sign(df["close"].diff()).fillna(0)

    s1 = np.where(daily_steps.rolling(5).sum() > 0, 1, -1)
    s2 = np.where(daily_steps.rolling(35).sum() > 0, 1, -1)
    s3 = np.where(daily_steps.rolling(150).sum() < 0, 1, -1)
    ma = df["close"].rolling(365).mean()
    s4 = np.where(ma > ma.shift(1), 1, -1)
    vol = df["close"].pct_change().rolling(10).std()
    s5 = np.where(vol < vol.shift(1), 1, -1)
    s6 = np.where(df["volume"] > df["volume"].rolling(20).mean(), 1, 0)
    s7 = np.where(df["close"] > df["open"], 1, -1)

    return s1 + s2 + s3 + s4 + s5 + s6 + s7

# ==============================
#   HMM + PUAN TURNUVASI
# ==============================
def run_tournament_logic(ticker):
    try:
        df = yf.download(ticker, start=START_DATE, progress=False)

        if df.empty:
            return "BEKLE", 0.0

        df.columns = [c.lower() for c in df.columns]
        if "close" not in df.columns and "adj close" in df.columns:
            df["close"] = df["adj close"]

        df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
        df["range"] = (df["high"] - df["low"]) / df["close"]
        df["custom_score"] = calculate_custom_score(df)
        df.dropna(inplace=True)

        if len(df) < 50:
            return "BEKLE", df["close"].iloc[-1]

        X = df[["log_ret", "range"]].values
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        hmm = GaussianHMM(n_components=3, covariance_type="full", n_iter=200)
        hmm.fit(Xs)
        df["state"] = hmm.predict(Xs)

        bull = df.groupby("state")["log_ret"].mean().idxmax()
        bear = df.groupby("state")["log_ret"].mean().idxmin()

        last = df.iloc[-1]
        hmm_sig = 1 if last["state"] == bull else -1
        score_sig = 1 if last["custom_score"] >= 3 else -1

        decision_val = 0.7 * hmm_sig + 0.3 * score_sig

        if decision_val > 0.25:
            return "AL", last["close"]
        elif decision_val < -0.25:
            return "SAT", last["close"]
        return "BEKLE", last["close"]

    except:
        return "BEKLE", 0.0

# ==============================
#   PORTFÖY GÜNCELLEYİCİ
# ==============================
def update_portfolio(df, sheet):
    tz = pytz.timezone("Europe/Istanbul")
    time_str = datetime.now(tz).strftime("%d-%m %H:%M")

    pf = df.copy()
    for i, row in pf.iterrows():
        ticker = row["Ticker"]
        decision, price = run_tournament_logic(ticker)

        if decision == "AL" and row["Durum"] == "CASH":
            cash = row["Nakit_Bakiye_USD"]
            amount = cash / price
            pf.at[i, "Durum"] = "COIN"
            pf.at[i, "Miktar"] = amount
            pf.at[i, "Nakit_Bakiye_USD"] = 0
            pf.at[i, "Son_Islem_Log"] = "AL (HMM)"
            pf.at[i, "Son_Islem_Fiyati"] = price
            pf.at[i, "Son_Islem_Zamani"] = time_str

        elif decision == "SAT" and row["Durum"] == "COIN":
            coin_value = row["Miktar"] * price
            pf.at[i, "Durum"] = "CASH"
            pf.at[i, "Nakit_Bakiye_USD"] = coin_value
            pf.at[i, "Miktar"] = 0
            pf.at[i, "Son_Islem_Log"] = "SAT (HMM)"
            pf.at[i, "Son_Islem_Fiyati"] = price
            pf.at[i, "Son_Islem_Zamani"] = time_str

        # Değer güncelle
        if pf.at[i, "Durum"] == "COIN":
            pf.at[i, "Kaydedilen_Deger_USD"] = pf.at[i, "Miktar"] * price
        else:
            pf.at[i, "Kaydedilen_Deger_USD"] = pf.at[i, "Nakit_Bakiye_USD"]

    save_portfolio(pf, sheet)
    return pf

# ==============================
#   STREAMLIT ARAYÜZÜ
# ==============================
st.set_page_config(page_title="AI Crypto Fund", page_icon="💰", layout="wide")
st.title("💰 AI Hedge Fund — FULL AUTO TRADING BOT")

df, sheet = load_portfolio()

st.subheader("📌 Portföy Durumu")
st.dataframe(df)

if st.button("🚀 Güncelle / Çalıştır"):
    df = update_portfolio(df, sheet)
    st.success("Portföy güncellendi!")
    st.dataframe(df)

st.markdown("---")
st.info("Bu bot 7 kriterli puanlama + 3 durumlu HMM turnuva motoru kullanmaktadır. Tamamen otomatik çalışır.")
