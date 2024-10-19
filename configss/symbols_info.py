


symbols_dict = {
    # ? Majers
    "EURUSD": {
        "decimal_divide": 1e5,
        "pip_size": 0.0001,
        "metatrader_id": "EURUSD",
        "dukascopy_id": "EURUSD",
        "swap_rate":{"long": -2, "short": +0.1}
    },
    "AUDUSD": {
        "decimal_divide": 1e5,
        "pip_size": 0.0001,
        "metatrader_id": "AUDUSD",
        "dukascopy_id": "AUDUSD",
        "swap_rate":{"long": 0.4, "short": -3}
    },
    "GBPUSD": {
        "decimal_divide": 1e5,
        "pip_size": 0.0001,
        "metatrader_id": "GBPUSD",
        "dukascopy_id": "GBPUSD",
        "swap_rate":{"long": 0.1, "short": -3}
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
        "swap_rate":{"long": 0.8, "short": -6}
    },
    # ? metals
    "XAUUSD": {
        "decimal_divide": 1e3,
        "pip_size": 0.1,
        "yahoo_finance": ["GC=F"],
        "metatrader_id": "XAUUSD",
        "dukascopy_id": "XAUUSD",
        "swap_rate":{"long": -8, "short": +1}
    },  # Spot gold
    # ? Crosses
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
