import ccxt
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import time
import datetime
import warnings
import math
import os # Ortam değişkenlerini okumak için

warnings.filterwarnings("ignore")

# --- GENEL AYARLAR VE GÜVENLİK (Gerçek API Bilgilerinizi Buraya Yazın) ---

# GÜVENLİK TAVSİYESİ: API anahtarlarını ortam değişkenlerinden çekin.
api_key = os.environ.get("BINANCE_API_KEY", "YOUR_BINANCE_API_KEY")
api_secret = os.environ.get("BINANCE_API_SECRET", "YOUR_BINANCE_API_SECRET")

exchange = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True,
})

# --- BOT PARAMETRELERİ ---
tickers = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"] 
initial_capital = 10000 
commission = 0.001
n_states = 3
validation_days = 21 
DECISION_THRESHOLD = 0.25 # Karar Eşiği (AL/SAT)

# --- RİSK PARAMETRELERİ (Yeni Eklenenler) ---
STOP_LOSS_PCT = 0.02      # %2 Zarar Durdurma
MIN_HOLD_HOURS = 24       # Minimum pozisyon tutma süresi (Saat)
MAX_ALLOCATION_PER_COIN = 0.50 # Tek bir coine maksimum kasanın %50'si
MTF_CONSENSUS_REQUIRED = 2 # İşlem için gerekli minimum Timeframe onayı (D, W, M)

# --- ZAMAN DİLİMLERİ ---
TIME_FRAMES = {'DAILY': '1d', 'WEEKLY': '1w', 'MONTHLY': '1M'}
WEIGHT_CANDIDATES = np.linspace(0.1, 0.9, 9)

# --- GLOBAL DURUM KAYDI (Simülasyon İçin) ---
PORTFOLIO_STATE = {'USDT': initial_capital, 'total_value': initial_capital}
POSITIONS = {t: {'qty': 0, 'entry_price': 0, 'entry_time': None} for t in tickers}
TRADE_LOG = [] # Tüm işlemleri kaydeder
HOURLY_UPDATE_LOG = [] # Portföy değeri takibi için saatlik log

# --- YARDIMCI FONKSİYONLAR ---

def calculate_custom_score(df):
    # Veri yeterliliği kontrolü (En uzun periyot 365 gün)
    if len(df) < 366: return pd.Series(0, index=df.index)

    s1 = np.where(df['close'] > df['close'].shift(5), 1, -1)
    s2 = np.where(df['close'] > df['close'].shift(35), 1, -1)
    s3 = np.where(df['close'] > df['close'].shift(150), 1, -1)
    s4 = np.where(df['close'] > df['close'].shift(365), 1, -1)
    vol = df['close'].pct_change().rolling(5).std()
    s5 = np.where(vol < vol.shift(5), 1, -1)
    s6 = np.where(df['volume'] > df['volume'].rolling(5).mean(), 1, -1) if 'volume' in df.columns else 0
    s7 = np.where(df['close'] > df['open'], 1, -1) if 'open' in df.columns else 0
    return s1+s2+s3+s4+s5+s6+s7

def get_ohlcv(ticker, timeframe, limit=1000):
    """Borsa API'den OHLCV verisini çeker."""
    # Varsayılan başlangıç tarihi (ccxt için)
    since = exchange.parse8601('2018-01-01T00:00:00Z') 
    
    ohlcv = exchange.fetch_ohlcv(ticker, timeframe=timeframe, limit=limit, since=since)
    df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.dropna(inplace=True)
    return df

def optimize_dynamic_weights(df):
    """
    Son 21 günlük validation verisi üzerinde en iyi HMM/Puan ağırlığını bulur.
    """
    df = df.copy()
    df['log_ret'] = np.log(df['close']/df['close'].shift(1))
    df['range'] = (df['high']-df['low'])/df['close']
    df['custom_score'] = calculate_custom_score(df)
    df.dropna(inplace=True)
    
    if len(df)<validation_days+5: return (0.7,0.3)
    
    train_df = df.iloc[:-validation_days]
    test_df = df.iloc[-validation_days:]
    
    X = train_df[['log_ret','range']].values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = GaussianHMM(n_components=n_states, covariance_type='full', n_iter=100, random_state=42)
    model.fit(X_s)
    
    state_stats = train_df.groupby(model.predict(X_s))['log_ret'].mean()
    bull_state = state_stats.idxmax()
    bear_state = state_stats.idxmin()
    
    best_roi = -np.inf
    best_w = (0.5,0.5) 
    
    # Grid search HMM ağırlığı
    for w_hmm in WEIGHT_CANDIDATES:
        w_score = 1-w_hmm
        cash_sim = initial_capital 
        coin_amt_sim = 0
        
        for idx,row in test_df.iterrows():
            X_test = scaler.transform([[row['log_ret'], row['range']]])
            hmm_signal = 1 if model.predict(X_test)[0]==bull_state else (-1 if model.predict(X_test)[0]==bear_state else 0)
            score_signal = 1 if row['custom_score']>=3 else (-1 if row['custom_score']<=-3 else 0)
            decision = w_hmm*hmm_signal + w_score*score_signal
            price = row['close']
            
            # Not: Optimizasyonda karar eşiği kullanılmaz, sadece ROI'ye bakılır
            if decision > 0.25: coin_amt_sim=cash_sim/price; cash_sim=0
            elif decision < -0.25: cash_sim=coin_amt_sim*price; coin_amt_sim=0
            
        final_val = cash_sim + coin_amt_sim*test_df['close'].iloc[-1]
        roi = (final_val-initial_capital)/initial_capital
        
        if roi>best_roi: best_roi=roi; best_w=(w_hmm,w_score)
        
    return best_w

def place_order_sim(ticker, side, amount_usd, price, time_now):
    """Gerçek emir göndermek yerine pozisyonu güncelleyen simülasyon fonksiyonu"""
    global PORTFOLIO_STATE
    
    if side == 'buy':
        qty = (amount_usd / price) * (1 - commission)
        
        # Risk Kontrolü: Tekrar AL sinyali gelirse ortalama maliyet hesapla
        if POSITIONS[ticker]['qty'] > 0:
            old_qty = POSITIONS[ticker]['qty']
            old_usd_value = old_qty * POSITIONS[ticker]['entry_price']
            
            new_qty = old_qty + qty
            new_usd_value = old_usd_value + amount_usd
            new_entry_price = new_usd_value / new_qty
            
            POSITIONS[ticker]['entry_price'] = new_entry_price
            POSITIONS[ticker]['qty'] = new_qty
            
        else:
            POSITIONS[ticker]['qty'] = qty
            POSITIONS[ticker]['entry_price'] = price
            POSITIONS[ticker]['entry_time'] = time_now
            
        PORTFOLIO_STATE['USDT'] -= amount_usd
        
        TRADE_LOG.append({
            'Time': time_now, 'Ticker': ticker, 'Action': 'BUY', 'Qty': qty,
            'Price': price, 'Fee': amount_usd * commission, 'Reason': 'Signal',
            'Timeframe': 'MTF'
        })
        print(f"🟢 BUY {ticker} @ {price:.2f} | Kasa Kalan: {PORTFOLIO_STATE['USDT']:.2f}")

    elif side == 'sell':
        qty = POSITIONS[ticker]['qty']
        if qty == 0: return

        revenue = qty * price
        revenue_after_fee = revenue * (1 - commission)
        
        PORTFOLIO_STATE['USDT'] += revenue_after_fee
        
        TRADE_LOG.append({
            'Time': time_now, 'Ticker': ticker, 'Action': 'SELL', 'Qty': qty,
            'Price': price, 'Fee': revenue * commission, 'Reason': 'Signal',
            'Pnl_Pct': (price / POSITIONS[ticker]['entry_price'] - 1) * 100,
            'Timeframe': 'MTF'
        })
        
        POSITIONS[ticker]['qty'] = 0
        POSITIONS[ticker]['entry_price'] = 0
        POSITIONS[ticker]['entry_time'] = None
        print(f"🔴 SELL {ticker} @ {price:.2f} | Gelir: {revenue_after_fee:.2f} | Kasa: {PORTFOLIO_STATE['USDT']:.2f}")

    elif side == 'stop_loss':
        qty = POSITIONS[ticker]['qty']
        if qty == 0: return
        
        revenue = qty * price
        revenue_after_fee = revenue * (1 - commission)
        
        PORTFOLIO_STATE['USDT'] += revenue_after_fee
        
        TRADE_LOG.append({
            'Time': time_now, 'Ticker': ticker, 'Action': 'STOP_LOSS', 'Qty': qty,
            'Price': price, 'Fee': revenue * commission, 'Reason': 'SL',
            'Pnl_Pct': (price / POSITIONS[ticker]['entry_price'] - 1) * 100,
            'Timeframe': 'Risk Control'
        })
        
        POSITIONS[ticker]['qty'] = 0
        POSITIONS[ticker]['entry_price'] = 0
        POSITIONS[ticker]['entry_time'] = None
        print(f"🚨 STOP_LOSS {ticker} @ {price:.2f} | Pozisyon Kapatıldı. Kasa: {PORTFOLIO_STATE['USDT']:.2f}")


def check_stop_loss(ticker, price, time_now):
    """Pozisyonda %STOP_LOSS_PCT zarar var mı kontrol eder."""
    pos = POSITIONS[ticker]
    if pos['qty'] > 0 and pos['entry_price'] > 0:
        loss_pct = (pos['entry_price'] - price) / pos['entry_price']
        if loss_pct >= STOP_LOSS_PCT:
            place_order_sim(ticker, 'stop_loss', 0, price, time_now) # 0 miktarı sadece stop_loss sinyalini tetikler

def check_minimum_hold_time(ticker, time_now):
    """Minimum tutma süresi doldu mu kontrol eder."""
    pos = POSITIONS[ticker]
    if pos['entry_time'] is None: return True
    
    elapsed_time = time_now - pos['entry_time']
    if elapsed_time.total_seconds() >= MIN_HOLD_HOURS * 3600:
        return True
    return False

def update_portfolio_value(time_now):
    """Anlık portföy değerini hesaplar ve loglar."""
    global PORTFOLIO_STATE
    
    total_value = PORTFOLIO_STATE['USDT']
    
    for ticker in tickers:
        qty = POSITIONS[ticker]['qty']
        if qty > 0:
            try:
                # Gerçek zamanlı fiyat çek
                ticker_data = exchange.fetch_ticker(ticker)
                price = ticker_data['close']
                total_value += qty * price
            except Exception:
                # Eğer fiyat çekilemezse, pozisyonu son entry fiyatından tut
                total_value += qty * POSITIONS[ticker]['entry_price']
    
    PORTFOLIO_STATE['total_value'] = total_value
    
    HOURLY_UPDATE_LOG.append({
        'Time': time_now,
        'Total_Value': total_value,
        'USDT_Balance': PORTFOLIO_STATE['USDT'],
        'Positions': {t: POSITIONS[t]['qty'] for t in tickers}
    })
    
    print(f"\n[PORTFÖY GÜNCELLEME] {time_now.strftime('%Y-%m-%d %H:%M:%S')} | Toplam Değer: {total_value:.2f} USDT")

# --- ANA BOT DÖNGÜSÜ ---
def run_live_bot():
    """Tüm analiz, risk kontrolü ve işlem mantığını içeren ana döngü."""
    global PORTFOLIO_STATE
    
    # Simülasyonun Başlangıç Ayarı
    capital_per_coin = initial_capital / len(tickers)
    
    # ⚠️ Güvenlik Kontrolü: API Anahtarı eksikse durdur
    if api_key == "YOUR_BINANCE_API_KEY" or api_secret == "YOUR_BINANCE_API_SECRET":
        print("\nFATAL HATA: Lütfen API anahtarlarınızı güncelleyin.")
        return

    # Başlangıç logu
    print("\n--- DYNAMIC V9 BOT BAŞLATILIYOR ---")
    print(f"Başlangıç Sermayesi: {initial_capital} USDT. | Risk Eşiği: {DECISION_THRESHOLD}")
    print(f"Stop Loss: {STOP_LOSS_PCT*100}%. | Min Hold: {MIN_HOLD_HOURS} saat.")
    print("-----------------------------------\n")

    # Hafta sonu takibini 168 saat (7 gün) veya manuel olarak durdurana kadar yap
    start_time = datetime.datetime.now()
    # 7 gün (168 saat) boyunca çalışsın
    end_time_limit = start_time + datetime.timedelta(hours=168) 
    
    current_time = start_time
    
    while current_time < end_time_limit:
        
        time_now = datetime.datetime.now()
        print(f"=== {time_now.strftime('%Y-%m-%d %H:%M:%S')} ===")
        
        update_portfolio_value(time_now) # Portföyü başta ve her saat güncelle

        for ticker in tickers:
            try:
                # 1. SL Kontrolü: Pozisyon varsa Stop Loss kontrolü yap
                if POSITIONS[ticker]['qty'] > 0:
                    current_price = exchange.fetch_ticker(ticker)['close']
                    check_stop_loss(ticker, current_price, time_now)
                    # SL tetiklenirse POSITIONS[ticker] sıfırlanmıştır

                # 2. Ağırlık Optimizasyonu
                # Sadece DAILY (1d) verisi ile (en uzun geçmişi çeker) optimize et
                df_long = get_ohlcv(ticker, timeframe='1d', limit=1000)
                w_hmm, w_score = optimize_dynamic_weights(df_long)
                
                # 3. MTF Sinyal Üretimi
                consensus_score = 0
                
                for tf_name, tf_code in TIME_FRAMES.items():
                    df_tf = get_ohlcv(ticker, timeframe=tf_code, limit=500)
                    
                    df_tf['log_ret'] = df_tf['close'].pct_change().apply(lambda x: np.log(1+x))
                    df_tf['range'] = (df_tf['high']-df_tf['low'])/df_tf['close']
                    df_tf['custom_score'] = calculate_custom_score(df_tf)
                    df_tf.dropna(inplace=True)
                    
                    if len(df_tf) < 50: continue

                    X = df_tf[['log_ret','range']].values
                    scaler = StandardScaler()
                    X_s = scaler.fit_transform(X)
                    
                    model = GaussianHMM(n_components=n_states, covariance_type='full', n_iter=100, random_state=42)
                    model.fit(X_s)

                    state_stats = df_tf.groupby(model.predict(X_s))['log_ret'].mean()
                    bull_state = state_stats.idxmax()
                    bear_state = state_stats.idxmin()
                    
                    last_row = df_tf.iloc[-1]
                    hmm_signal = 1 if model.predict(scaler.transform([[last_row['log_ret'], last_row['range']]]))[0]==bull_state else (-1 if model.predict(scaler.transform([[last_row['log_ret'], last_row['range']]]))[0]==bear_state else 0)
                    score_signal = 1 if last_row['custom_score']>=3 else (-1 if last_row['custom_score']<=-3 else 0)
                    
                    decision = w_hmm*hmm_signal + w_score*score_signal
                    
                    if decision > DECISION_THRESHOLD: consensus_score += 1
                    elif decision < -DECISION_THRESHOLD: consensus_score -= 1

                # 4. İşlem Kararı (Konsensüs Kontrolü)
                current_price = exchange.fetch_ticker(ticker)['close']
                position_qty = POSITIONS[ticker]['qty']
                
                # Minimum tutma süresi kontrolü
                can_trade = check_minimum_hold_time(ticker, time_now) 
                
                if consensus_score >= MTF_CONSENSUS_REQUIRED and position_qty == 0 and can_trade:
                    # AL sinyali ve MTF onayı var
                    amount_usd_to_buy = min(PORTFOLIO_STATE['USDT'] / (len(tickers) - sum(1 for p in POSITIONS.values() if p['qty'] > 0)), capital_per_coin)
                    
                    if amount_usd_to_buy > 10: # Minimum işlem limiti
                        # Gerçek Alım Simülasyonu
                        place_order_sim(ticker, 'buy', amount_usd_to_buy, current_price, time_now)
                
                elif consensus_score <= -MTF_CONSENSUS_REQUIRED and position_qty > 0 and can_trade:
                    # SAT sinyali ve MTF onayı var
                    # Gerçek Satış Simülasyonu
                    place_order_sim(ticker, 'sell', 0, current_price, time_now) # Miktar 0, fonksiyon içinden çekilir
                
                else:
                    print(f"⚪ {ticker}: HOLD (Konsensüs: {consensus_score}/{MTF_CONSENSUS_REQUIRED}. Pozisyon: {position_qty:.2f})")
            
            except Exception as e:
                print(f"🚨 {ticker} GENEL HATA (Döngü İçi): {e}")

        # 1 saat bekle
        time.sleep(3600) 
        current_time = datetime.datetime.now()

    # Bot durduktan sonra sonuçları yazdırma
    print("\n--- BOT SİMÜLASYONU SONUÇLANDI ---")
    final_value = PORTFOLIO_STATE['total_value']
    roi = (final_value - initial_capital) / initial_capital * 100
    print(f"Final Bakiye: {final_value:.2f} USDT | ROI: {roi:.2f}%")
    
    # Logları DataFrame'e çevirip yazdırma
    df_hourly = pd.DataFrame(HOURLY_UPDATE_LOG)
    df_trades = pd.DataFrame(TRADE_LOG)
    
    print("\n--- SAATLİK PORTFÖY DEĞERİ TAKİP TABLOSU ---")
    print(df_hourly.tail(24))
    print("\n--- İŞLEM LOGU (SON 10 İŞLEM) ---")
    print(df_trades.tail(10))

# --- BOTU ÇALIŞTIR ---
if __name__ == "__main__":
    run_live_bot()