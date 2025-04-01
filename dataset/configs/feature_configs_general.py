##? asset classes
US_INDICES = ["US30", "US100"]
FOREX = ["EURUSD", "GBPUSD", "USDJPY"]
CRYPTO = ["BTCUSD"]

##? in EST time zone
time_sessions = {
    "US Indices": {
        "New_York": {
            "hour": 15,
            "minute": 55
        }
    },
    "XAUUSD": {
        "New_York": {
            "hour": 16,
            "minute": 55
        },
        "London": {
            "hour": 10,
            "minute": 55,
        },
        "Asia": {
            "hour": 2,
            "minute": 55,
        },
    },
    "FOREX": {
        "New_York": {
            "hour": 16,
            "minute": 55,
        },
        "London": {
            "hour": 11,
            "minute": 55,
        },
        "Tokyo": {
            "hour": 3,
            "minute": 55,
        },
        "Sydney": {
            "hour": 23,
            "minute": 55,
        },
    }
}

##? in EST time zone
sessions_trade_times = {
    "US Indices": {
        "New_York": (9.5, 16),
    },
    "XAUUSD": {
        "New_York": (8, 17),
        "London": (3, 11),
        "Asia": (18, 3),
    },
    "FOREX": {
        "New_York": (8, 17),
        "London": (3, 12),
        "Tokyo": (19, 4),
        "Sydney": (15, 0),
    }
}

fe_leg_config = {
    "XAUUSD": {
        'percentage': 0.003,
        'timeframe': [5, 15, 60],
        'window_size': [19, 24, 30],
        'exponents': {
            '15': (2, 3),
            '60': (4, 6)
        }
    },
    "EURUSD": {
        'percentage': 0.001,
        'timeframe': [5, 15, 60],
        'window_size': [800, 1100, 1400],
        'exponents': {
            '15': (1.5, 1.75),
            '60': (3, 4.5)
        }
    },
    "GBPUSD": {
        'percentage': 0.0015,
        'timeframe': [5, 15, 60],
        'window_size': [1240, 1600, 2400],
        'exponents': {
            '15': (1, 1.5),
            '60': (4, 7)
        }
    },
    "USDJPY": {
        'percentage': 0.0015,
        'timeframe': [5, 15, 60],
        'window_size': [13, 18, 28],
        'exponents': {
            '15': (2, 3),
            '60': (3, 4.5)
        }
    },
    "US100": {
        'percentage': 0.0025,
        'timeframe': [5, 15, 60],
        'window_size': [19, 24, 34],
        'exponents': {
            '15': (2, 2.5),
            '60': (4, 6)
        }
    },
    "US30": {
        'percentage': 0.0017,
        'timeframe': [5, 15, 60],
        'window_size': [16, 23, 40],
        'exponents': {
            '15': (2, 2.5),
            '60': (5, 6)
        }
    },
    "BTCUSD": {
        'percentage': 0.01,
        'timeframe': [5, 15],
        'window_size': [49, 60, 80],
        'exponents': {
            '15': (2, 3)
        }
    },
}

general_config = {
    'base_candle_timeframe': [15, 30, 60, 120, 180, 240, 360, 720, 1380],


    'fe_GMA': {
        'timeframe': [5, 240],
        'window_size': [5, 7, 9],
        'base_columns': ['CLOSE', 'HIGH', 'LOW'],
        'devs': [1, 1.3, 1.5, 1.7, 1.9, 2]
    },


    'fe_FFD': {
        'timeframe': [5],  # always 5
        'window_size': [0],  # to prevent causing bugs in the framework
        'base_columns': ['CLOSE'],
        'n_splits': [5]
    },


    'fe_OL': {
        'timeframe': [5, 60, 240],
        'window_size': [1, 7, 21, 33],
        'window_size_SMA': [5, 15, 21, 55],
        'base_columns': ['HIGH', 'LOW']
    },


    'fe_supertrend': {
        'timeframe': [5, 15],
        'window_size': [14],
        'multipliers': [1.0, 3.0],
        'base_columns': ['HIGH', 'CLOSE', 'LOW']
    },


    'fe_leg': {
        'base_columns': ['CLOSE']
    },


    'fe_cndl_shape_n_cntxt': {
        'timeframe': [5, 15, 60],
        'window_size': [12],
        'base_columns': ['OPEN', 'HIGH', 'LOW', 'CLOSE']
    },


    'fe_ATR': {
        'timeframe': [5, 60, 240],
        'window_size': [7, 14, 30],
        'base_columns': ['HIGH', 'CLOSE', 'LOW']
    },


    'fe_RSTD': {
        'timeframe': [240],
        'window_size': [7, 14, 30],
        'base_columns': ['CLOSE']
    },


    # Support & Resistance Levels
    'fe_WIN': {
        'timeframe': [5],
        'window_size': [6, 12, 48, 276, 480],
        'base_columns': ['CLOSE']
    },


    'fe_WIN_FREQ': {
        'timeframe': [5],
        'window_size': [240, 480, 1380],
        'base_columns': ['CLOSE']
    },


    'fe_cndl': [5, 15, 30, 60, 240, 1380],


    'fe_EMA': {
        'timeframe': [5],
        'window_size': [7, 60, 336, 1380],
        'base_columns': ['CLOSE']
    },


    'fe_SMA': {
        'timeframe': [5],
        'window_size': [20, 60, 240, 360, 720],
        'base_columns': ['CLOSE']
    },


    'fe_RSI': {
        'timeframe': [5, 60, 240],
        'window_size': [7, 14, 30],
        'base_columns': ['CLOSE']
    },


    'fe_cndl_shift': {
        'columns': ['OPEN', 'HIGH', 'LOW', 'CLOSE'],
        'shift_configs': [
            {'timeframe': 5, 'shift_sizes': [1]},
            {'timeframe': 15, 'shift_sizes': [1]},
            {'timeframe': 30, 'shift_sizes': [1]},
            {'timeframe': 60, 'shift_sizes': [1]},
            {'timeframe': 240, 'shift_sizes': [1]},
            {'timeframe': 1380, 'shift_sizes': [1]}
        ]
    },


    'fe_ratio': {
        'ATR': {
            'timeframe': [60, 240],
            'window_size': [
                (7, 14),
                (7, 30),
            ]
        },

        'EMA': {
            'timeframe': [5],
            'window_size': [
                (7, 60),
                (60, 336),
                (60, 1380),
            ]
        },

        'RSI': {
            'timeframe': [5, 60, 240],
            'window_size': [
                (7, 14),
                (7, 30),
            ]
        },

        'RSTD': {
            'timeframe': [240],
            'window_size': [
                (7, 14),
                (7, 30),
            ]
        },

        'SMA': {
            'timeframe': [5],
            'window_size': [
                (240, 720),
                (360, 720),
            ]
        }
    },
}

def generate_general_config(symbols, general_config=general_config):
    config_dict = {}
    for sym in symbols:
        config_dict[sym] = general_config
    
    return config_dict
