

# symbols = [
#   "EURUSD", "USDCAD", "USDJPY", "EURJPY", "GBPUSD", "XAUUSD",
#   "AUDUSD", "NZDUSD", "USDCHF", "CADJPY", "EURGBP",
# ]


symbols = [
  "EURUSD", 
]

general_config = {
  'base_candle_timeframe': [15, 30, 60, 120, 180, 240, 360, 720, 1440],
  
  'fe_ATR': {'timeframe': [240, 60],
  'window_size': [30, 14, 7],
  'base_columns': ['HIGH', 'CLOSE', 'LOW']},


  'fe_RSTD': {'timeframe': [240],
  'window_size': [30, 14, 7],
  'base_columns': ['CLOSE']},


  'fe_WIN': {'timeframe': [5],
  'window_size': [5, 10, 48, 288, 480],
  'base_columns': ['CLOSE']},


  'fe_cndl': [5, 15, 30, 60, 240, 1440],


  'fe_EMA': {'timeframe': [5],
  'window_size': [7, 60, 336, 1440],
  'base_columns': ['CLOSE']},

  'fe_SMA': {'base_columns': ['CLOSE'],
  'timeframe': [5],
  'window_size': [240, 360, 720]},

  'fe_RSI': {'timeframe': [5, 60, 240],
  'window_size': [30, 14, 7],
  'base_columns': ['CLOSE']},


  'fe_cndl_shift': {'columns': ['OPEN', 'HIGH', 'LOW', 'CLOSE'],
  'shift_configs': [
    {'timeframe': 5, 'shift_sizes': [1]},
    {'timeframe': 15, 'shift_sizes': [1]},
    {'timeframe': 30, 'shift_sizes': [1]},
    {'timeframe': 60, 'shift_sizes': [1]},
    {'timeframe': 240, 'shift_sizes': [1]},
    {'timeframe': 1440, 'shift_sizes': [1]}]},


  'fe_ratio': {'ATR': {'timeframe': [60, 240],
    'window_size': [(7, 14), (7, 30)]},

  'EMA': {'timeframe': [5], 'window_size': [
    (7, 60),
    (60, 336),
    (60, 1440),
    ]},

  'RSI': {'timeframe': [5, 60, 240],
    'window_size': [(7, 14), (7, 30)]},

  'RSTD': {'timeframe': [240],
    'window_size': [(7, 14), (7, 30)]},

  'SMA': {'timeframe': [5], 'window_size': [
    (5 * 48, 15 * 48),
    (10 * 48, 15 * 48),
    ]}},

}

def generate_general_config(symbols=symbols,general_config=general_config):
    config_dict = {}
    for sym in symbols:
        config_dict[sym] = general_config
    
    return config_dict