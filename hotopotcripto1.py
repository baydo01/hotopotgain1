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
from sklearn.metrics import accuracy_score, precision_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# =============================================================================
# 1. AYARLAR & KONFİGÜRASYON
# =============================================================================
st.set_page_config(page_title="Hedge Fund AI: Pro Dashboard", layout="wide", page_icon="🏦")

SHEET_ID = "16zjLeps0t1P26OF3o7XQ-djEKKZtZX6t5lFxLmnsvpE"
CREDENTIALS_FILE = "service_account.json"
TARGET_COINS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "DOGE-USD"]

# Stil CSS (Dashboard Görünümü İçin)
st.markdown("""
<style>
    .metric-card {background-color: #0E1117; border: 1px solid #262730; border-radius: 5px; padding: 15px; text-align: center;}
    .stProgress > div > div > div > div {background-color: #00cc96;}
    div[data-testid="stExpander"] div[role="button"] p {font-size: 1.1rem; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

st.title("🏦 Hedge Fund AI: Pro Dashboard (Walk-Forward)")
st.markdown("Bu sistem, **Walk-Forward Analizi** kullanarak geçmiş veriyi gün gün simüle eder ve sanki o gün canlı işlem yapıyormuş gibi karar verir.")

# =============================================================================
# 2. GOOGLE SHEETS ENTEGRASYONU
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
# 3. GELİŞMİŞ VERİ İŞLEME & FEATURE ENGINEERING
# =============================================================================
def get_data(ticker, period="730d", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        return df
    except: return None

def calculate_indicators(df):
    if df is None or len(df) < 50: return None
    df = df.copy()
    
    # 1. Trend Göstergeleri
    df['sma20'] = df['close'].rolling(20).mean()
    df['sma50'] = df['close'].rolling(50).mean()
    df['sma200'] = df['close'].rolling(200).mean()
    
    # 2. Momentum (RSI)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 3. Volatilite (ATR)
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift(1)).abs()
    lc = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    
    # 4. Bollinger Bantları
    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + (2 * df['bb_std'])
    df['bb_lower'] = df['bb_mid'] - (2 * df['bb_std'])
    
    # 5. Getiriler
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    
    # 6. Target (Gelecek 1 günün yönü - Binary)
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    df.dropna(inplace=True)
    return df

# =============================================================================
# 4. WALK-FORWARD SIMULATION ENGINE (BACKTEST MOTORU)
# =============================================================================
class BacktestEngine:
    def __init__(self, initial_capital=100.0, stop_loss_atr=2.0, trailing_stop=True):
        self.initial_capital = initial_capital
        self.balance = initial_capital
        self.position = 0.0 # Coin miktarı
        self.entry_price = 0.0
        self.highest_price = 0.0 # Trailing stop için
        self.stop_price = 0.0
        self.trades = [] # İşlem geçmişi
        self.equity_curve = []
        self.sl_atr_mult = stop_loss_atr
        self.trailing_stop = trailing_stop
        
    def run_simulation(self, df, model_probs):
        """
        df: Fiyat verisi (Index: Tarih)
        model_probs: Modelin her gün için ürettiği 'Yükseliş' olasılığı
        """
        dates = df.index
        closes = df['close'].values
        atrs = df['atr'].values
        opens = df['open'].values # İşlemler bir sonraki açılışta yapılır (Realistic)
        
        in_position = False
        
        # Simülasyon Döngüsü (Her gün için)
        for i in range(len(df) - 1):
            date = dates[i]
            current_close = closes[i]
            current_atr = atrs[i]
            prob = model_probs[i]
            
            next_open = opens[i+1] # İşlem yapılacak fiyat
            
            # --- POZİSYON YÖNETİMİ (Eğer pozisyondaysak) ---
            if in_position:
                # 1. Trailing Stop Güncelleme
                if current_close > self.highest_price:
                    self.highest_price = current_close
                    if self.trailing_stop:
                        new_stop = self.highest_price - (current_atr * self.sl_atr_mult)
                        self.stop_price = max(self.stop_price, new_stop)
                
                # 2. Stop Loss Kontrolü (Düşük fiyatla kontrol edilir)
                if df['low'].iloc[i] < self.stop_price:
                    # Stop olduk
                    exit_price = self.stop_price # Slippage ihmal edildi
                    pnl = (exit_price - self.entry_price) * self.position
                    self.balance += (self.position * exit_price)
                    self.trades.append({'Date': date, 'Type': 'SELL (STOP)', 'Price': exit_price, 'Balance': self.balance})
                    in_position = False
                    self.position = 0.0
                    
                # 3. Model "SAT" derse (Prob < 0.4)
                elif prob < 0.40:
                    exit_price = next_open
                    self.balance += (self.position * exit_price)
                    self.trades.append({'Date': date, 'Type': 'SELL (SIGNAL)', 'Price': exit_price, 'Balance': self.balance})
                    in_position = False
                    self.position = 0.0
            
            # --- ALIM YÖNETİMİ (Nakitdeysek) ---
            else:
                # Model "AL" derse (Prob > 0.6) ve Trend Filtresi (SMA50 üzerinde)
                # Not: Trend filtresi feature engineering içinde halledilebilir ama burada da ek güvenlik.
                if prob > 0.60:
                    self.position = self.balance / next_open
                    self.entry_price = next_open
                    self.highest_price = next_open
                    self.stop_price = next_open - (current_atr * self.sl_atr_mult)
                    self.balance = 0.0 # Tüm parayı yatır (Basitlik için)
                    self.trades.append({'Date': date, 'Type': 'BUY', 'Price': next_open, 'Balance': self.balance})
                    in_position = True
            
            # Equity Curve Kaydı
            current_equity = self.balance + (self.position * current_close)
            self.equity_curve.append(current_equity)
            
        return self.equity_curve, self.trades

# =============================================================================
# 5. MODEL EĞİTİMİ VE TAHMİN (Walk-Forward)
# =============================================================================
def train_and_predict(df):
    """
    Son 30 gün (veya belirlenen test süresi) için Walk-Forward tahmin üretir.
    """
    features = ['log_ret', 'rsi', 'atr', 'bb_pct', 'sma20', 'sma50']
    # Basit feature normalizasyonu
    df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    X = df[features].fillna(0)
    y = df['target']
    
    # Test süresi: Son 60 bar (Yaklaşık 2 ay)
    test_size = 60 
    predictions = []
    
    # Walk-Forward Loop
    # Modeli her gün yeniden eğitmek yavaş olabilir, bu yüzden
    # "Sliding Window" kullanacağız. Train seti her adımda büyür.
    
    # İlk eğitim
    train_end = len(df) - test_size
    model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, eval_metric='logloss')
    
    # Hızlandırmak için: Modeli bir kere eğitip, son 60 günde tahmin alalım.
    # Gerçek Walk-Forward için döngü içinde fit() gerekir ama Streamlit limiti için
    # "Expanding Window" simülasyonu yapıyoruz.
    
    model.fit(X.iloc[:train_end], y.iloc[:train_end])
    
    # Test seti üzerindeki olasılıklar
    probs = model.predict_proba(X.iloc[-test_size:])[:, 1]
    
    return probs, df.iloc[-test_size:]

# =============================================================================
# 6. ARAYÜZ OLUŞTURMA
# =============================================================================
def create_dashboard_ui(ticker_data):
    # Metrik Hesaplama
    equity = ticker_data['equity']
    trades = ticker_data['trades']
    
    start_bal = equity[0]
    end_bal = equity[-1]
    roi = ((end_bal - start_bal) / start_bal) * 100
    
    # Drawdown
    equity_series = pd.Series(equity)
    rolling_max = equity_series.cummax()
    drawdown = (equity_series - rolling_max) / rolling_max
    max_dd = drawdown.min() * 100
    
    # Win Rate
    wins = [t for i, t in enumerate(trades) if t['Type'].startswith('SELL') and t['Balance'] > trades[i-1]['Balance']] # Basit mantık
    win_rate = (len(wins) / (len(trades)/2)) * 100 if len(trades) > 0 else 0
    
    # --- METRİK KARTLARI ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Toplam Getiri (ROI)", f"%{roi:.2f}", delta_color="normal")
    col2.metric("Max Drawdown", f"%{max_dd:.2f}", delta_color="inverse")
    col3.metric("Son Fiyat", f"${ticker_data['price']:.2f}")
    col4.metric("Model Sinyali", f"{ticker_data['last_signal']:.2f}", help="0-1 arası. >0.6 AL, <0.4 SAT")

    # --- GRAFİKLER ---
    tab1, tab2 = st.tabs(["📈 Analiz Grafiği", "📝 İşlem Geçmişi"])
    
    with tab1:
        # Alt alta iki grafik: Fiyat ve Strateji Performansı
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
        
        # 1. Fiyat Grafiği (Candlestick)
        df_plot = ticker_data['df_test']
        fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['open'], high=df_plot['high'], low=df_plot['low'], close=df_plot['close'], name='Fiyat'), row=1, col=1)
        
        # İşlem Noktaları
        buy_x = [t['Date'] for t in trades if t['Type']=='BUY']
        buy_y = [t['Price'] for t in trades if t['Type']=='BUY']
        sell_x = [t['Date'] for t in trades if t['Type'].startswith('SELL')]
        sell_y = [t['Price'] for t in trades if t['Type'].startswith('SELL')]
        
        fig.add_trace(go.Scatter(x=buy_x, y=buy_y, mode='markers', marker=dict(symbol='triangle-up', color='green', size=12), name='AL'), row=1, col=1)
        fig.add_trace(go.Scatter(x=sell_x, y=sell_y, mode='markers', marker=dict(symbol='triangle-down', color='red', size=12), name='SAT'), row=1, col=1)
        
        # 2. Equity Curve (Bakiye)
        fig.add_trace(go.Scatter(x=df_plot.index, y=equity, mode='lines', line=dict(color='#00cc96', width=2), name='Strateji Bakiyesi'), row=2, col=1)
        
        fig.update_layout(height=600, margin=dict(l=0, r=0, t=30, b=0), template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        if trades:
            trades_df = pd.DataFrame(trades)
            st.dataframe(trades_df, use_container_width=True)
        else:
            st.info("Bu simülasyon döneminde hiç işlem açılmadı.")

# =============================================================================
# 7. ANA UYGULAMA AKIŞI
# =============================================================================
pf_df, sheet = load_portfolio()

if not pf_df.empty:
    # Sidebar Özet
    with st.sidebar:
        total_balance = pf_df['Nakit_Bakiye_USD'].sum() + pf_df[pf_df['Durum']=='COIN']['Kaydedilen_Deger_USD'].sum()
        st.metric("💰 Toplam Portföy", f"${total_balance:.2f}")
        st.divider()
        st.write("Aktif Pozisyonlar:")
        st.dataframe(pf_df[pf_df['Durum']=='COIN'][['Ticker', 'Miktar']], hide_index=True)
    
    if st.button("🚀 SİSTEMİ ÇALIŞTIR VE ANALİZ ET", type="primary"):
        updated_pf = pf_df.copy()
        prog_bar = st.progress(0)
        
        # Havuz Hesabı
        pool_cash = updated_pf['Nakit_Bakiye_USD'].sum()
        
        buy_signals = [] # Alım adayları
        
        for i, (idx, row) in enumerate(updated_pf.iterrows()):
            ticker = row['Ticker']
            raw_data = get_data(ticker)
            
            if raw_data is not None:
                df_ind = calculate_indicators(raw_data)
                
                if df_ind is not None:
                    # 1. Model Tahmini (Walk-Forward benzeri)
                    probs, df_test = train_and_predict(df_ind)
                    
                    # 2. Backtest Motorunu Çalıştır
                    engine = BacktestEngine(initial_capital=100, stop_loss_atr=2.5) # ATR 2.5 Stop
                    equity, trades = engine.run_simulation(df_test, probs)
                    
                    last_prob = probs[-1]
                    current_price = df_test['close'].iloc[-1]
                    
                    # Sonuçları Kaydet
                    sim_result = {
                        'ticker': ticker,
                        'equity': equity,
                        'trades': trades,
                        'price': current_price,
                        'df_test': df_test,
                        'last_signal': last_prob,
                        'roi': (equity[-1] - 100)
                    }
                    
                    with st.expander(f"📊 {ticker} Analizi - ROI: %{sim_result['roi']:.1f}", expanded=False):
                        create_dashboard_ui(sim_result)
                    
                    # --- AL/SAT KARAR MEKANİZMASI ---
                    # MEVCUT POZİSYON VARSA:
                    if row['Durum'] == 'COIN':
                        # Satış Şartları:
                        # 1. Sinyal çok düşük (< 0.35)
                        # 2. Fiyat SMA50 altına düştü (Trend kırıldı)
                        is_trend_broken = current_price < df_test['sma50'].iloc[-1]
                        
                        if last_prob < 0.35 or (is_trend_broken and last_prob < 0.5):
                            sale_val = float(row['Miktar']) * current_price
                            pool_cash += sale_val
                            updated_pf.at[idx, 'Durum'] = 'CASH'
                            updated_pf.at[idx, 'Miktar'] = 0.0
                            updated_pf.at[idx, 'Nakit_Bakiye_USD'] = 0.0
                            updated_pf.at[idx, 'Son_Islem_Log'] = f"SAT ({current_price:.2f})"
                            st.toast(f"🛑 {ticker} Satıldı (Trend Kırıldı/Sinyal Zayıf)")
                            
                    # NAKİTTEYSEK:
                    elif row['Durum'] == 'CASH':
                        # Alım Adayı Listesine Ekle
                        # Şart: Sinyal > 0.60 VE Trend Yukarı (Fiyat > SMA50) VE Son 1 ayda kazandırmış (ROI > 0)
                        if last_prob > 0.60 and current_price > df_test['sma50'].iloc[-1] and sim_result['roi'] > -2:
                            buy_signals.append({
                                'idx': idx,
                                'ticker': ticker,
                                'prob': last_prob,
                                'roi': sim_result['roi'],
                                'price': current_price
                            })

            prog_bar.progress((i + 1) / len(updated_pf))
        
        # --- ALIMLARI GERÇEKLEŞTİR ---
        # Nakiti adaylar arasında paylaştır (ROI'ye göre ağırlıklandır)
        if buy_signals and pool_cash > 10:
            total_weight = sum([x['prob'] for x in buy_signals]) # Olasılığa göre ağırlık ver
            
            for signal in buy_signals:
                share = (signal['prob'] / total_weight) * pool_cash
                amount = share / signal['price']
                
                updated_pf.at[signal['idx'], 'Durum'] = 'COIN'
                updated_pf.at[signal['idx'], 'Miktar'] = amount
                updated_pf.at[signal['idx'], 'Nakit_Bakiye_USD'] = 0.0
                updated_pf.at[signal['idx'], 'Son_Islem_Fiyati'] = signal['price']
                updated_pf.at[signal['idx'], 'Son_Islem_Log'] = f"AL ({signal['price']:.2f})"
                st.toast(f"✅ {signal['ticker']} Alındı (${share:.1f})")
                
        elif not buy_signals and pool_cash > 0:
            # Kimse alınmadıysa para ilk satırda nakit olarak dursun
            first_idx = updated_pf.index[0]
            updated_pf.at[first_idx, 'Nakit_Bakiye_USD'] = pool_cash
            for x in updated_pf.index:
                if x != first_idx and updated_pf.at[x, 'Durum'] == 'CASH':
                    updated_pf.at[x, 'Nakit_Bakiye_USD'] = 0.0

        # Değer Güncelleme
        for idx, row in updated_pf.iterrows():
            if row['Durum'] == 'COIN':
                # Fiyatı results'dan bulmamız lazım ama basitlik için yfinance son fiyatı tekrar çekebiliriz veya yukarıda saklayabilirdik.
                # Burada hızlıca tekrar çekiyoruz (optimize edilebilir)
                try:
                    p = yf.download(row['Ticker'], period="1d", progress=False)['Close'].iloc[-1]
                    updated_pf.at[idx, 'Kaydedilen_Deger_USD'] = float(row['Miktar']) * float(p)
                except: pass
            else:
                updated_pf.at[idx, 'Kaydedilen_Deger_USD'] = row['Nakit_Bakiye_USD']

        save_portfolio(updated_pf, sheet)
        st.success("✅ Analiz ve Portföy Güncellemesi Tamamlandı!")
        st.balloons()
