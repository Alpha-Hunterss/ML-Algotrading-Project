symbols_dict = {
    # ? majors
    "EURUSD": {
        "decimal_divide": 1e5,
        "pip_size": 0.0001,
        "metatrader_id": "EURUSD",
        "dukascopy_id": "EURUSD",
        "swap_rate":{"long": -3.25, "short": 1.56}
    },
    "AUDUSD": {
        "decimal_divide": 1e5,
        "pip_size": 0.0001,
        "metatrader_id": "AUDUSD",
        "dukascopy_id": "AUDUSD",
        "swap_rate":{"long": -1.99, "short": -0.37}
    },
    "GBPUSD": {
        "decimal_divide": 1e5,
        "pip_size": 0.0001,
        "metatrader_id": "GBPUSD",
        "dukascopy_id": "GBPUSD",
        "swap_rate":{"long": -1.99, "short": -0.37}
    },
    "NZDUSD": {
        "decimal_divide": 1e5,
        "pip_size": 0.0001,
        "metatrader_id": "NZDUSD",
        "dukascopy_id": "NZDUSD",
        "swap_rate":{"long": 0.3, "short": -4}
    },
    "USDCAD": {
        "decimal_divide": 1e5,
        "pip_size": 0.0001,
        "metatrader_id": "USDCAD",
        "dukascopy_id": "USDCAD",
        "swap_rate":{"long": 0.2, "short": -3}
    },
    "USDCHF": {
        "decimal_divide": 1e5,
        "pip_size": 0.0001,
        "metatrader_id": "USDCHF",
        "dukascopy_id": "USDCHF",
        "swap_rate":{"long": -4, "short": 0.5}
    },
    "USDJPY": {
        "decimal_divide": 1e3,
        "pip_size": 0.01,
        "metatrader_id": "USDJPY",
        "dukascopy_id": "USDJPY",
        "swap_rate":{"long": 1.46, "short": -2.92}
    },
    # ? US indices
    "US30": {
        "decimal_divide": 1e2,
        "pip_size": 0.01,
        "metatrader_id": "US30",
        "swap_rate":{"long": -7.65, "short": 2.03}
    },
    "US100": {
        "decimal_divide": 1e2,
        "pip_size": 0.01,
        "metatrader_id": "USTEC",
        "swap_rate":{"long": -3.31, "short": 0.83}
    },
    # ? metals
    "XAUUSD": {
        "decimal_divide": 1e2,
        "pip_size": 0.01,
        "yahoo_finance": ["GC=F"],
        "metatrader_id": "XAUUSD",
        "dukascopy_id": "XAUUSD",
        "swap_rate":{"long": -3.84, "short": 2.78}
    },  # Spot gold
    # ? crypto
    "BTCUSD": {
        "decimal_divide": 1e2,
        "pip_size": 0.01,
        "metatrader_id": "BTCUSD",
        "swap_rate":{"long": -20.0, "short": 0.0}
    },
    # ? crosses
    "EURJPY": {
        "decimal_divide": 1e3,
        "pip_size": 0.01,
        "metatrader_id": "EURJPY",
        "dukascopy_id": "EURJPY",
        "swap_rate":{"long": 0.5, "short": -5}
    },
    "CADJPY": {
        "decimal_divide": 1e3,
        "pip_size": 0.01,
        "metatrader_id": "CADJPY",
        "dukascopy_id": "CADJPY",
        "swap_rate":{"long": 0.4, "short": -5}
    },
    "EURGBP": {
        "decimal_divide": 1e5,
        "pip_size": 0.0001,
        "metatrader_id": "EURGBP",
        "dukascopy_id": "EURGBP",
        "swap_rate":{"long": -4, "short": +0.2}
    },
}
