# %%
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime as dt
import datetime, pytz
from sklearn.neighbors import KNeighborsClassifier
import pickle
import MetaTrader5 as mt5
from collections import namedtuple
import json
import sys

from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.momentum import StochasticOscillator
from ta.volatility import BollingerBands
from ta.volume import AccDistIndexIndicator
from ta.volatility import AverageTrueRange

# %% [markdown]
# # Classes

# %% [markdown]
# ## MT5 Bot
# (Not available for Mac/Linux)

# %%
class MT5Bot:
    def __init__(self):
        if not mt5.initialize():
            print("initialize() failed, error code =",mt5.last_error())
            print()
            self.shutdown()
            quit()

    def login(self,account,server,password):
        if mt5.login(login=account, server=server,password=password):
            print('Succefully logged in')
            self.acc_info = mt5.account_info()._asdict()
            for key,value in self.acc_info.items():
                print(f'    {key}: {value}')
        else:
            print('Failed to login')
        print()

    def shutdown(self):
        mt5.shutdown()
    
    def parse_position(self):
        mt5_pos = mt5.positions_get()
        pos = pd.DataFrame(columns=['ticket',
                                    'time',
                                    'time_update',
                                    'time_update_msc',
                                    'type',
                                    'magic',
                                    'identifier',
                                    'reason',
                                    'volume',
                                    'price_open',
                                    'sl', 
                                    'tp', 
                                    'price_current', 
                                    'swap', 
                                    'profit', 
                                    'symbol', 
                                    'comment', 
                                    'external_id'])
        for r in range(len(mt5_pos)):
            trade_pos = mt5_pos[r]
            for col in pos.columns:
                pos.loc[r,col] = eval(f'trade_pos.{col}')
        pos['position'] = np.where(pos['type']==0,1,-1) * pos['volume']
        return pos
    
    def open_position(self,symbol,vol):
        print(f'Opening {symbol}...')
        action = mt5.TRADE_ACTION_DEAL
        if np.sign(vol) == 1:
            action_type = mt5.ORDER_TYPE_BUY
            print(f'Buy {abs(vol)} lots')
        elif np.sign(vol) == -1:
            action_type = mt5.ORDER_TYPE_SELL
            print(f'Sell {abs(vol)} lots')
        lot = abs(vol)
        request = {
            "action": action,
            "symbol": symbol,
            "volume": lot,
            "type": action_type,
            "comment": 'FX Bot open position',
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        result = mt5.order_send(request)
        print(result)

    def close_position(self,symbol,vol,ticket):
        print(f'Closing {symbol}...')
        action = mt5.TRADE_ACTION_DEAL
        if np.sign(vol) == 1:
            action_type = mt5.ORDER_TYPE_BUY
            print(f'Buy {abs(vol)} lots')
        elif np.sign(vol) == -1:
            action_type = mt5.ORDER_TYPE_SELL
            print(f'Sell {abs(vol)} lots')
        lot = abs(vol)
        request = {
            "action": action,
            "symbol": symbol,
            "volume": lot,
            "type": action_type,
            "position": ticket,
            "comment": "FX Bot close position",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        result = mt5.order_send(request)
        print(result)

    def close_all(self,symbol,curr_pos):
        print(f'Closing all {symbol}...')
        for r in range(len(curr_pos)):
            vol = curr_pos.loc[r,'position']
            ticket = curr_pos.loc[r,'ticket']
            self.close_position(symbol,-vol,ticket)
    
    def close_part(self,symbol,chg_vol,curr_pos):
        print(f'Closing part of {symbol}...')
        for r in range(len(curr_pos)):
            vol = curr_pos.loc[r,'position']
            ticket = curr_pos.loc[r,'ticket']
            self.close_position(symbol,chg_vol,ticket)
            chg_vol = chg_vol + vol
            if np.sign(chg_vol) != np.sign(vol):
                break
    
    def modify_position(self,symbol,new_vol):
        curr_pos = self.parse_position()
        curr_pos = curr_pos.loc[curr_pos['symbol']==symbol].sort_values('volume').reset_index(drop=True)
        tot_vol = curr_pos['position'].sum()
        direction = 'LONG' if np.sign(tot_vol) else 'SHORT'
        new_direction = 'LONG' if np.sign(new_vol) else 'SHORT'
        chg_vol = new_vol - tot_vol

        if chg_vol == 0:
            print(f'{symbol} position remains the same')
            pass

        elif tot_vol == 0:
            print(f'Open new {symbol} {direction} position')
            self.open_position(symbol,new_vol)

        elif new_vol == 0:
            print(f'Close all {symbol} {direction} position')
            self.close_all(symbol,curr_pos)

        elif np.sign(chg_vol) == np.sign(tot_vol):
            print(f'Adding {symbol} {direction} position')
            self.open_position(symbol,chg_vol)

        elif np.sign(chg_vol) != np.sign(tot_vol):
            print(f'Closing {symbol} {direction} position')
            if abs(chg_vol) > abs(tot_vol):
                self.close_all(symbol,curr_pos)
                print(f'Adding {symbol} {new_direction} position')
                self.open_position(symbol,chg_vol+tot_vol)
            else:
                self.close_part(symbol,chg_vol,curr_pos)
        
        print()

# %% [markdown]
# ## Signal Generator

# %%
class SignalGenerator:
    def __init__(self):
        pass
    
    def load_model(self,dir='models'):
        self.knns = list()
        for i in range(100):
            with open(f'{dir}/knn_{i}.pkl', 'rb') as f:
                knn = pickle.load(f)
                self.knns.append(knn)
        print(f'Loaded {i+1} KNNs')
        print()

    def set_symbols(self,symbols):
        self.symbols = symbols
        print(f'Symbols: {self.symbols}')
        print()

    def set_parameters(self,leverage,acc_size,lot_size):
        self.leverage = leverage
        self.acc_size = acc_size
        self.lot_size = lot_size

    def load_features(self,features):
        self.features = features.copy()
        print('Loaded features')
        print(self.features.keys())
        print()

    def load_rates(self,convert,rates):
        self.convert = convert.copy()
        print('Loaded convert table')
        print(self.convert)
        print()
        self.rates = rates.copy()
        print('Loaded rates')
        print(self.rates.tail()[::-1])
        print()

    def get_positions(self):
        print('Getting positions...')
        for symbol in self.symbols:
            features = self.features[symbol]
            pos = pd.DataFrame()
            for i,knn in enumerate(self.knns):
                pos[f'{i}'] = knn.predict(features.iloc[:,1:])
            features['Pred'] = pos.mean(axis=1).to_numpy()
            features['Pos'] = (np.around((features['Pred'].expanding().rank(pct=True)*2-1)*5))/5
        print()

    def get_lots(self):
        self.get_positions()
        print('Getting lots...')
        position = pd.DataFrame()
        for symbol in self.symbols:
            capital = self.leverage * self.acc_size / len(self.symbols)
            
            if symbol in self.convert.keys():
                lot_usd = self.lot_size / self.rates[self.convert[symbol]].iloc[-1]
            else:
                lot_usd = self.lot_size * 1
            
            pos_value = self.features[symbol]['Pos'] * capital

            position[symbol] = pos_value / lot_usd
        self.lots = np.around(position,2)[::-1]
        print()
        return self.lots

# %% [markdown]
# ## Data Downloader

# %%
class DataDownloader:
    def __init__(self):
        pass
    
    def set_symbols(self,symbols,dl_symbols):
        self.symbols = symbols
        self.dl_symbols = dl_symbols
        print(f'Symbols: {self.symbols}')
        print(f'Download symbols: {self.dl_symbols}')
        print()

    def download_yfinance(self):
        data = dict()
        for symbol in self.symbols:
            print(f'Downloading {symbol}...')
            data[symbol] = yf.download(self.dl_symbols[symbol])
        self.data = data
        print()

    def get_data(self):
        return self.data

    def get_ta_features(self):
        ta_features = dict()
        for symbol in self.symbols:
            print(f'Adding TA for {symbol}...')
            ta_data = self.add_ta_signals(self.data[symbol])
            ta_features[symbol] = ta_data.iloc[-252:]
        self.ta_features = ta_features
        print()
        return self.ta_features

    def add_ta_signals(self,df):
        df = df.copy()

        o,h,l,c,v = df['Open'],df['High'],df['Low'],df['Close'],df['Volume']
        
        df['Ret'] = np.log(c).diff().shift(-1)

        df['MACDD'] = MACD(c).macd_diff()
        df['MACDS'] = MACD(c).macd_signal()

        df['RSI'] = RSIIndicator(c).rsi()

        df['SO'] = StochasticOscillator(h,l,c).stoch()
        df['SOS'] = StochasticOscillator(h,l,c).stoch_signal()

        df['BBH'] = BollingerBands(c).bollinger_hband_indicator() - c
        df['BBL'] = BollingerBands(c).bollinger_lband_indicator() - c

        df['ADI'] = AccDistIndexIndicator(h,l,c,v).acc_dist_index()

        df['HLR'] = c / (h.rolling(252).max() - l.rolling(252).min()) - 1 / 2

        df['ATR'] = AverageTrueRange(h,l,c).average_true_range()

        df = df.drop(columns=['Open','High','Low','Close','Adj Close','Volume'])

        try: df.drop(columns=['Repaired?'],inplace=True)
        except: pass

        #df.iloc[:,1:-1] = df.iloc[:,1:-1].div(df.loc[:,'ATR'],axis=0)
        return df

# %% [markdown]
# # Main

# %%
utc_datetime = datetime.datetime.now(pytz.utc)
est_datetime = datetime.datetime.now(pytz.timezone('US/Eastern'))
print(utc_datetime.strftime('UTC: %Y-%m-%d %H:%M:%S'))
print(est_datetime.strftime('EST: %Y-%m-%d %H:%M:%S'))
print()
print('Using the nearest-day position at 20:00 HKT (00:00 UTC)')
print('Open the position at 19:55 since data will disappear at the following 2 hours')
print('Backtest is buying at today\'s close based on data generated at today\'s close, and predicting the return from today to tomorrow')
print('Trading 18x is optimal but 5x is conservative')
print()

# %%


print('==== Load Config =========================================================')
config_url = sys.argv[0]
with open(config_url) as f:
    config = json.load(f)
print(config)

print('==== Download Data =======================================================')
dl_symbols = {  'EURUSD': 'EURUSD=X',
                'USDJPY': 'JPY=X',
                'GBPUSD': 'GBPUSD=X',
                'USDCHF': 'CHF=X',
                'AUDUSD': 'AUDUSD=X',
                'USDCAD': 'CAD=X',
                'NZDUSD': 'NZDUSD=X'}
symbols = ['EURUSD','USDJPY','GBPUSD','USDCHF','AUDUSD','USDCAD','NZDUSD']
data_dl = DataDownloader()
data_dl.set_symbols(symbols,dl_symbols)
data_dl.download_yfinance()
features = data_dl.get_ta_features()

print('==== Download Convert Rates ==============================================')
base_to_usd = { 'EURUSD': 'EUR=X',
                'GBPUSD': 'GBP=X',
                'AUDUSD': 'AUD=X',
                'NZDUSD': 'NZD=X'}
prc_to_usd = yf.download(list(base_to_usd.values()))['Close']

print('==== Generate New Positions ==============================================')
signal_gen = SignalGenerator()
signal_gen.load_model('models')
signal_gen.set_symbols(symbols)
signal_gen.set_parameters(leverage=config['leverage'],
                          acc_size=config['acc_size'],
                          lot_size=config['lot_size'])
signal_gen.load_features(features)
signal_gen.load_rates(base_to_usd,prc_to_usd)
lots = signal_gen.get_lots()
new_pos = lots.iloc[0].sort_index()

print('==== Connect to MT5 ======================================================')
trader = MT5Bot()
trader.login(account=config['account'],
             server=config['server'],
             password=config['password'])
print('==== Current Positions ===================================================')
curr_pos = trader.parse_position().loc[:,['symbol','position']]
print(curr_pos.groupby('symbol').sum())
print('==== New Positions =======================================================')
print(new_pos)
print('==== Rebalance ===========================================================')
date = new_pos.name.date()
print(f'Rebalance the position at {date}? (y/n)')
confirm = input() if sys.argv[1] not in ('y','n') else sys.argv[1]
if confirm == 'y':
    for symbol,pos in zip(new_pos.index,new_pos):
        trader.modify_position(symbol,pos)
else:
    print('Halted rebalancing')
trader.shutdown()


