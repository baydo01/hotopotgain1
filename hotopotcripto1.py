import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import warnings
import gspread
import os
import json
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime, timedelta
import pytz

# --- ML & İstatistik ---
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# =============================================================================
# 1. AYARLAR & STİL (HEDGE FUND UI)
# =============================================================================
st.set_page_config(page_title="Hedge Fund AI: Chameleon V3", layout="wide", page_icon="🦎")

# Özel CSS (Bloomberg / Glassmorphism Style)
st.markdown("""
<style>
    .main {background-color: #0E1117;}
    .stApp {background-color: #0E1117;}
    
    /* Header Stili */
    .header-box {
        background: linear-gradient(90deg, #1f4037, #99f2c8); /* Hunter Green */
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        color: #000;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .header-title {font-size: 28px; font-weight: bold; margin: 0;}
    .header-subtitle {font-size: 16px; opacity: 0.9;}
    
    /* Metrik Kartları */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        backdrop-filter: blur(5px);
    }
    .metric-val {font-size: 24px; font-weight: bold; color: #00CC96;}
    .metric-label {font-size: 14px; color: #aaa;}
    
    /* Rejim Etiketleri */
    .regime-hunter {color: #00CC96; font-weight: bold; border: 1px solid #00CC96; padding: 2px 8px; border-radius: 4px;}
    .regime-harvester {color: #FFAA00; font-weight: bold; border: 1px solid #FFAA00; padding: 2px 8px; border-radius: 4px;}
    .regime-bunker {color: #EF553B; font-weight: bold; border: 1px solid #EF553B; padding: 2px 8px; border-radius: 4px;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-box">
    <div class="header-title">🦎 Hedge Fund AI: Chameleon V3</div>
    <div class="header-subtitle">Regime Switching • Dynamic Risk • Walk-Forward Simulation</div>
</div>
""", unsafe_allow_html=True)

SHEET_ID = "16zjLeps0t1P26OF3o7XQ-djEKKZtZX6t5lFxLmnsvpE"
CREDENTIALS_FILE = "service_account.json"
TARGET_COINS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "DOGE-USD"]

# =============================================================================
# 2. ALTYAPI & BAĞLANTI
# =============================================================================
def connect_sheet():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = None
    try:
        if "gcp_service_account" in st.secrets:
            creds = ServiceAccountCredentials.from_json_keyfile_dict(dict(st.secrets["gcp_service_account"]), scope)
        elif os.path.exists(CREDENTIALS_FILE):
            creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, scope)
        if not creds: return None
        client = gspread.authorize(creds)
        return client.open_by_key(SHEET_ID).sheet1
    except: return None

def load_portfolio():
    sheet = connect_sheet()
    if sheet is None: return pd.DataFrame(), None
    try:
        headers = sheet.row_values(1)
        required_cols = ["Ticker", "Durum", "Miktar", "Son_Islem_Fiyati", "Nakit_Bakiye_USD", "Baslangic_USD", "Kaydedilen_Deger_USD", "Son_Islem_Log"]
        if not headers or headers[0] != "Ticker":
            sheet.clear(); sheet.append_row(required_cols)
            defaults = []
            for t in TARGET_COINS: defaults.append([t, "CASH", 0.0, 0.0, 100.0, 100.0, 100.0, "KURULUM"])
            for d in defaults: sheet.append_row(d)
            time.sleep(1)
        
        data = sheet.get_all_records()
        df = pd.DataFrame(data)
        numeric_cols = ["Miktar", "Son_Islem_Fiyati", "Nakit_Bakiye_USD", "Baslangic_USD", "Kaydedilen_Deger_USD"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce').fillna(0.0)
        return df, sheet
    except: return pd.DataFrame(), None

def save_portfolio(df, sheet):
    if sheet is None: return
    try:
        df_export = df.copy().astype(str)
        sheet.update([df_export.columns.values.tolist()] + df_export.values.tolist())
    except: pass

# =============================================================================
# 3. İLERİ SEVİYE MATEMATİK & TEKNİK ANALİZ
# =============================================================================
def calculate_adx(df, period=14):
    """ADX (Trend Gücü) Hesaplama - Manuel Implementasyon"""
    df = df.copy()
    df['up_move'] = df['high'] - df['high'].shift(1)
    df['down_move'] = df['low'].shift(1) - df['low']
    df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0.0)
    df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0.0)
    
    df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))))
    
    df['atr'] = df['tr'].rolling(period).mean()
    
    df['plus_di'] = 100 * (df['plus_dm'].rolling(period).mean() / df['atr'])
    df['minus_di'] = 100 * (df['minus_dm'].rolling(period).mean() / df['atr'])
    
    df['dx'] = 100 * abs((df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di']))
    df['adx'] = df['dx'].rolling(period).mean()
    return df['adx'].fillna(0), df['atr'].fillna(0)

def get_data_with_features(ticker):
    try:
        # 1.5 Yıllık Veri (Daha geniş perspektif)
        df = yf.download(ticker, period="730d", interval="1d", progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        
        # Temel Göstergeler
        df['sma50'] = df['close'].rolling(50).mean()
        df['sma200'] = df['close'].rolling(200).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # ADX ve ATR (Volatilite)
        df['adx'], df['atr'] = calculate_adx(df)
        
        # Bollinger
        df['bb_mid'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_lower'] = df['bb_mid'] - (2 * df['bb_std'])
        df['bb_upper'] = df['bb_mid'] + (2 * df['bb_std'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
        
        # Log Return & Target
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        df.dropna(inplace=True)
        return df
    except: return None

# =============================================================================
# 4. THE CHAMELEON BRAIN (REJİM & STRATEJİ MOTORU)
# =============================================================================
class ChameleonBrain:
    def __init__(self):
        pass
        
    def detect_regime(self, row):
        """
        Piyasa rejimini belirler: HUNTER (Boğa), HARVESTER (Yatay), BUNKER (Ayı)
        """
        price = row['close']
        sma200 = row['sma200']
        adx = row['adx']
        
        # Kural 1: Fiyat 200 günlüğün altındaysa tehlike çanları (Ayı)
        # Ancak ADX çok düşükse (trend yoksa) hala Harvester olabilir.
        if price < sma200:
            if adx < 20: 
                return "HARVESTER" # Düşük volatilitede yatay seyir
            else:
                return "BUNKER" # Güçlü düşüş trendi
        
        # Kural 2: Fiyat 200 günlüğün üzerinde (Boğa Potansiyeli)
        else:
            if adx > 25:
                return "HUNTER" # Güçlü Trend
            else:
                return "HARVESTER" # Trend zayıf, yatay
            
    def get_strategy_signal(self, regime, row, model_prob):
        """
        Rejime göre AL/SAT/BEKLE kararı ve Risk Parametreleri üretir.
        Döndürür: (Action, StopLoss_ATR_Mult, TakeProfit_ATR_Mult, Position_Size_Pct)
        Action: 1 (AL), 0 (BEKLE), -1 (SAT)
        """
        rsi = row['rsi']
        
        if regime == "HUNTER":
            # --- STRATEJİ: TREND FOLLOWER ---
            # Model onayıyla al, trend bitene kadar tut.
            # Stop Loss Geniş, Take Profit Yok (Trailing)
            if model_prob > 0.60 and rsi > 50:
                return 1, 3.0, None, 1.0 # %100 Pozisyon, Geniş Stop
            elif model_prob < 0.40:
                return -1, 0, 0, 0
                
        elif regime == "HARVESTER":
            # --- STRATEJİ: MEAN REVERSION (OSİLATÖR) ---
            # Dipten al, tepeden sat. Model sinyali ikinci planda.
            # Stop Loss Dar, Take Profit Hızlı
            if rsi < 35: # Aşırı Satım
                return 1, 1.5, 3.0, 0.4 # %40 Pozisyon (Daha az risk)
            elif rsi > 65: # Aşırı Alım
                return -1, 0, 0, 0
                
        elif regime == "BUNKER":
            # --- STRATEJİ: CAPITAL PRESERVATION ---
            # Sadece "Deep Dip" (Ölü kedi) alımı. Yoksa Nakit.
            if rsi < 20: # Fiyat çok çok düştü, tepki gelebilir
                return 1, 1.0, 2.0, 0.2 # %20 Pozisyon (Çok riskli)
            else:
                return -1, 0, 0, 0 # NAKİTE GEÇ
                
        return 0, 0, 0, 0 # İşlem yok

# =============================================================================
# 5. WALK-FORWARD SİMÜLASYON MOTORU
# =============================================================================
class WalkForwardEngine:
    def __init__(self, initial_capital=100.0):
        self.balance = initial_capital
        self.position_amt = 0.0
        self.entry_price = 0.0
        self.trades = []
        self.equity_curve = []
        self.brain = ChameleonBrain()
        self.in_position = False
        
        # Anlık Trade Parametreleri
        self.current_sl_price = 0.0
        self.current_tp_price = 0.0
        self.regime_history = []
        
    def run(self, df, model_probs):
        dates = df.index
        opens = df['open'].values # İşlem bir sonraki barın açılışında
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        atrs = df['atr'].values
        
        # Simülasyon son 90 gün için (Hız için)
        sim_start_idx = 0 
        
        for i in range(sim_start_idx, len(df)-1):
            row = df.iloc[i]
            current_date = dates[i]
            prob = model_probs[i]
            
            # 1. Rejimi Tespit Et
            regime = self.brain.detect_regime(row)
            self.regime_history.append(regime)
            
            # 2. İşlem Fiyatı (Bir sonraki açılış)
            next_open_price = opens[i+1]
            
            # --- POZİSYON YÖNETİMİ ---
            if self.in_position:
                # Stop Loss Kontrol
                if lows[i+1] < self.current_sl_price:
                    exit_price = self.current_sl_price # Slippage ihmal
                    self._close_position(exit_price, current_date, "SL")
                    
                # Take Profit Kontrol (Varsa)
                elif self.current_tp_price and highs[i+1] > self.current_tp_price:
                    exit_price = self.current_tp_price
                    self._close_position(exit_price, current_date, "TP")
                
                # Sinyal ile Çıkış
                else:
                    action, _, _, _ = self.brain.get_strategy_signal(regime, row, prob)
                    if action == -1: # SAT Sinyali
                        self._close_position(next_open_price, current_date, "SIGNAL")
                    
                    # Trailing Stop (Sadece Hunter Modunda)
                    elif regime == "HUNTER":
                        new_sl = closes[i] - (atrs[i] * 2.5)
                        if new_sl > self.current_sl_price:
                            self.current_sl_price = new_sl

            # --- ALIM YÖNETİMİ ---
            else:
                action, sl_mult, tp_mult, pos_size = self.brain.get_strategy_signal(regime, row, prob)
                
                if action == 1:
                    # Risk Yönetimi: Bakiyenin %'si kadar al
                    invest_amt = self.balance * pos_size
                    self.position_amt = invest_amt / next_open_price
                    self.balance -= invest_amt
                    self.entry_price = next_open_price
                    self.in_position = True
                    
                    # SL / TP Belirle
                    self.current_sl_price = next_open_price - (atrs[i] * sl_mult)
                    self.current_tp_price = (next_open_price + (atrs[i] * tp_mult)) if tp_mult else None
                    
                    self.trades.append({
                        'Date': current_date, 'Type': 'BUY', 'Price': next_open_price, 
                        'Regime': regime, 'Balance': self._get_equity(closes[i])
                    })
            
            # Equity Kayıt
            self.equity_curve.append(self._get_equity(closes[i]))
            
        return self.equity_curve, self.trades, self.regime_history

    def _close_position(self, price, date, reason):
        val = self.position_amt * price
        self.balance += val
        self.position_amt = 0.0
        self.in_position = False
        self.trades.append({
            'Date': date, 'Type': f'SELL ({reason})', 'Price': price, 
            'Regime': 'EXIT', 'Balance': self.balance
        })

    def _get_equity(self, current_price):
        return self.balance + (self.position_amt * current_price)

# =============================================================================
# 6. MODEL EĞİTİMİ (EXPANDING WINDOW)
# =============================================================================
def get_ml_predictions(df):
    """
    Basit bir XGBoost modeli ile olasılıkları üretir.
    Gerçek bir Walk-Forward için bu döngü içinde olmalıydı ama performans için
    tek seferde tahmin alıp simülasyon motoruna besleyeceğiz.
    """
    features = ['log_ret', 'rsi', 'adx', 'bb_width', 'sma50']
    X = df[features].fillna(0)
    y = df['target']
    
    # Son 90 günü test et
    test_len = 90
    train_end = len(df) - test_len
    
    model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, eval_metric='logloss')
    model.fit(X.iloc[:train_end], y.iloc[:train_end])
    
    # Tüm veriye tahmin üret (Simülasyon motoru seçecek)
    probs = model.predict_proba(X)[:, 1]
    
    # Sadece test kısmı için veriyi kesip döndürelim
    return probs[-test_len:], df.iloc[-test_len:]

# =============================================================================
# 7. UI HELPER & MAIN APP
# =============================================================================
pf_df, sheet = load_portfolio()

if not pf_df.empty:
    with st.sidebar:
        st.write("## 💼 Portföy Özeti")
        total_usd = pf_df['Nakit_Bakiye_USD'].sum() + pf_df[pf_df['Durum']=='COIN']['Kaydedilen_Deger_USD'].sum()
        st.metric("Toplam Varlık", f"${total_usd:.2f}")
        st.divider()
        st.write("Pozisyonlar:")
        st.dataframe(pf_df[pf_df['Durum']=='COIN'][['Ticker','Miktar']], hide_index=True)
    
    if st.button("🦎 CHAMELEON ANALİZİNİ BAŞLAT", type="primary", use_container_width=True):
        
        updated_pf = pf_df.copy()
        pool_cash = updated_pf['Nakit_Bakiye_USD'].sum()
        buy_candidates = []
        
        progress_bar = st.progress(0)
        
        for i, (idx, row) in enumerate(updated_pf.iterrows()):
            ticker = row['Ticker']
            df = get_data_with_features(ticker)
            
            if df is not None:
                # 1. Tahminleri Al
                probs, df_sim = get_ml_predictions(df)
                
                # 2. Simülasyonu Çalıştır
                engine = WalkForwardEngine(initial_capital=100)
                equity, trades, regimes = engine.run(df_sim, probs)
                
                # Son Durumlar
                last_regime = regimes[-1]
                last_roi = equity[-1] - 100
                current_price = df_sim['close'].iloc[-1]
                last_prob = probs[-1]
                
                # --- DASHBOARD KARTI ---
                with st.expander(f"{ticker} | Rejim: {last_regime} | ROI: %{last_roi:.1f}", expanded=False):
                    
                    # Üst Metrikler
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Son Fiyat", f"${current_price:.2f}")
                    c2.metric("ROI (90 Gün)", f"%{last_roi:.1f}", delta_color="normal")
                    
                    # Rejim Rozeti
                    regime_html = ""
                    if last_regime == "HUNTER": regime_html = "<span class='regime-hunter'>🏹 HUNTER (BOĞA)</span>"
                    elif last_regime == "HARVESTER": regime_html = "<span class='regime-harvester'>🦀 HARVESTER (YATAY)</span>"
                    else: regime_html = "<span class='regime-bunker'>🛡️ BUNKER (AYI)</span>"
                    
                    c3.markdown(f"<div style='text-align:center'><small>Piyasa Rejimi</small><br>{regime_html}</div>", unsafe_allow_html=True)
                    c4.metric("AI Sinyali", f"{last_prob:.2f}")
                    
                    # Grafikler
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
                    
                    # Fiyat Grafiği
                    fig.add_trace(go.Candlestick(x=df_sim.index, open=df_sim['open'], high=df_sim['high'], low=df_sim['low'], close=df_sim['close'], name='Fiyat'), row=1, col=1)
                    
                    # Buy/Sell İşaretleri
                    if trades:
                        buys = [t for t in trades if t['Type']=='BUY']
                        sells = [t for t in trades if 'SELL' in t['Type']]
                        fig.add_trace(go.Scatter(x=[t['Date'] for t in buys], y=[t['Price'] for t in buys], mode='markers', marker=dict(symbol='triangle-up', color='#00CC96', size=12), name='AL'), row=1, col=1)
                        fig.add_trace(go.Scatter(x=[t['Date'] for t in sells], y=[t['Price'] for t in sells], mode='markers', marker=dict(symbol='triangle-down', color='#EF553B', size=12), name='SAT'), row=1, col=1)
                        
                    # Equity Curve
                    fig.add_trace(go.Scatter(x=df_sim.index, y=equity, line=dict(color='#FFAA00', width=2), name='Strateji Bakiye'), row=2, col=1)
                    
                    fig.update_layout(height=500, margin=dict(t=0, b=0, l=0, r=0), template="plotly_dark", showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                # --- CANLI İŞLEM KARAR MEKANİZMASI ---
                # Mevcut Durum Kontrolü
                brain = ChameleonBrain()
                action, _, _, pos_size = brain.get_strategy_signal(last_regime, df_sim.iloc[-1], last_prob)
                
                # SATIŞ (Eğer eldekini satmamız gerekiyorsa)
                if row['Durum'] == 'COIN':
                    if action == -1: # Brain "SAT" diyor
                        sale_val = float(row['Miktar']) * current_price
                        pool_cash += sale_val
                        updated_pf.at[idx, 'Durum'] = 'CASH'
                        updated_pf.at[idx, 'Miktar'] = 0.0
                        updated_pf.at[idx, 'Nakit_Bakiye_USD'] = 0.0
                        updated_pf.at[idx, 'Son_Islem_Log'] = f"SAT ({last_regime})"
                        st.toast(f"🛑 {ticker} Satıldı ({last_regime} Modu)")
                        
                # ALIM LİSTESİ
                elif row['Durum'] == 'CASH':
                    if action == 1: # Brain "AL" diyor
                        buy_candidates.append({
                            'idx': idx, 'ticker': ticker, 'price': current_price,
                            'weight': pos_size, # Rejime göre ağırlık (Hunter: 1.0, Bunker: 0.2)
                            'regime': last_regime
                        })

            progress_bar.progress((i+1)/len(updated_pf))
            
        # --- ALIMLARI GERÇEKLEŞTİR ---
        if buy_candidates and pool_cash > 10:
            total_weight = sum([c['weight'] for c in buy_candidates])
            
            for c in buy_candidates:
                # Ağırlıklı Paylaştırma
                share_pct = c['weight'] / total_weight
                usd_amount = pool_cash * share_pct
                
                # İşlem
                amount = usd_amount / c['price']
                updated_pf.at[c['idx'], 'Durum'] = 'COIN'
                updated_pf.at[c['idx'], 'Miktar'] = amount
                updated_pf.at[c['idx'], 'Nakit_Bakiye_USD'] = 0.0
                updated_pf.at[c['idx'], 'Son_Islem_Fiyati'] = c['price']
                updated_pf.at[c['idx'], 'Son_Islem_Log'] = f"AL ({c['regime']})"
                st.toast(f"✅ {c['ticker']} Alındı (${usd_amount:.1f})")
                
        elif not buy_candidates and pool_cash > 0:
            # Nakit Koruma
            fidx = updated_pf.index[0]
            updated_pf.at[fidx, 'Nakit_Bakiye_USD'] = pool_cash
            for xi in updated_pf.index:
                if xi != fidx and updated_pf.at[xi, 'Durum'] == 'CASH':
                    updated_pf.at[xi, 'Nakit_Bakiye_USD'] = 0.0

        # Değerleme
        for idx, row in updated_pf.iterrows():
            if row['Durum'] == 'COIN':
                try:
                    p = yf.download(row['Ticker'], period="1d", progress=False)['Close'].iloc[-1]
                    updated_pf.at[idx, 'Kaydedilen_Deger_USD'] = float(row['Miktar']) * float(p)
                except: pass
            else:
                updated_pf.at[idx, 'Kaydedilen_Deger_USD'] = row['Nakit_Bakiye_USD']

        save_portfolio(updated_pf, sheet)
        st.success("🦎 Bukalemun Analizi Tamamlandı! Portföy Rejime Göre Uyarlandı.")
        st.balloons()
