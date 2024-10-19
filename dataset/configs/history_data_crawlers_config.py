from datetime import datetime
from pathlib import Path
import os

start_date_str = "2023/01/01"
# stop_date_str = "2024/07/01"
start_date = datetime.strptime(start_date_str, "%Y/%m/%d")
# stop_date = datetime.strptime(stop_date_str, "%Y/%m/%d")
stop_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)



metatrader_number_of_days = 340
root_path = str(os.path.dirname(os.path.abspath(__file__))).replace("configs", "")
data_folder = f"{root_path}/data/raw_data/"
Path(data_folder).mkdir(parents=True, exist_ok=True)

symbols_dict = {
    # ? Majers
    "EURUSD": {
        "decimal_divide": 1e5,
        "pip_size": 0.0001,
        "metatrader_id": "EURUSD",
        "dukascopy_id": "EURUSD",
    },
    "AUDUSD": {
        "decimal_divide": 1e5,
        "pip_size": 0.0001,
        "metatrader_id": "AUDUSD",
        "dukascopy_id": "AUDUSD",
    },
    "GBPUSD": {
        "decimal_divide": 1e5,
        "pip_size": 0.0001,
        "metatrader_id": "GBPUSD",
        "dukascopy_id": "GBPUSD",
    },
    "NZDUSD": {
        "decimal_divide": 1e5,
        "pip_size": 0.0001,
        "metatrader_id": "NZDUSD",
        "dukascopy_id": "NZDUSD",
    },
    "USDCAD": {
        "decimal_divide": 1e5,
        "pip_size": 0.0001,
        "metatrader_id": "USDCAD",
        "dukascopy_id": "USDCAD",
    },
    "USDCHF": {
        "decimal_divide": 1e5,
        "pip_size": 0.0001,
        "metatrader_id": "USDCHF",
        "dukascopy_id": "USDCHF",
    },
    "USDJPY": {
        "decimal_divide": 1e3,
        "pip_size": 0.01,
        "metatrader_id": "USDJPY",
        "dukascopy_id": "USDJPY",
    },
    # ? metals
    # "XAGUSD":{"decimal_divide":1e+3,"pip_size":0.1,"metatrader_id":"XAGUSD","dukascopy_id":"XAGUSD"}, # Spot silver
    "XAUUSD": {
        "decimal_divide": 1e3,
        "pip_size": 0.1,
        "yahoo_finance": ["GC=F"],
        "metatrader_id": "XAUUSD",
        "dukascopy_id": "XAUUSD",
    },  # Spot gold
    # ? Crosses
    "EURJPY": {
        "decimal_divide": 1e3,
        "pip_size": 0.01,
        "metatrader_id": "EURJPY",
        "dukascopy_id": "EURJPY",
    },
    # "AUDCHF":{"decimal_divide":1e+5,"pip_size":0.0001,"metatrader_id":"AUDCHF","dukascopy_id":"AUDCHF"},
    # "AUDJPY":{"decimal_divide":1e+3,"pip_size":0.01,"metatrader_id":"AUDJPY","dukascopy_id":"AUDJPY"},
    "CADJPY": {
        "decimal_divide": 1e3,
        "pip_size": 0.01,
        "metatrader_id": "CADJPY",
        "dukascopy_id": "CADJPY",
    },
    # "CADCHF":{"decimal_divide":1e+5,"pip_size":0.0001,"metatrader_id":"CADCHF","dukascopy_id":"CADCHF"},
    # "CHFJPY":{"decimal_divide":1e+3,"pip_size":0.01,"metatrader_id":"CHFJPY","dukascopy_id":"CHFJPY"},
    # "EURAUD":{"decimal_divide":1e+5,"pip_size":0.0001,"metatrader_id":"EURAUD","dukascopy_id":"EURAUD"},
    # "EURCAD":{"decimal_divide":1e+5,"pip_size":0.0001,"metatrader_id":"EURCAD","dukascopy_id":"EURCAD"},
    # "EURCHF":{"decimal_divide":1e+5,"pip_size":0.0001,"metatrader_id":"EURCHF","dukascopy_id":"EURCHF"},
    "EURGBP": {
        "decimal_divide": 1e5,
        "pip_size": 0.0001,
        "metatrader_id": "EURGBP",
        "dukascopy_id": "EURGBP",
    },
    # #? indies:
    # "DOLLARIDXUSD":{"decimal_divide":1e+3,"pip_size":0.0001,"metatrader_id":"","dukascopy_id":"DOLLARIDXUSD"},
    # "USA30IDXUSD":{"decimal_divide":1e+3,"pip_size":0.0001,"metatrader_id":"","dukascopy_id":"USA30IDXUSD"},
    # "USATECHIDXUSD":{"decimal_divide":1e+3,"pip_size":0.0001,"metatrader_id":"","dukascopy_id":"USATECHIDXUSD"},
    # "USA500IDXUSD":{"decimal_divide":1e+3,"pip_size":0.0001,"metatrader_id":"SP500.r","dukascopy_id":"USA500IDXUSD"},
    # "USSC2000IDXUSD":{"decimal_divide":1e+3,"pip_size":0.0001,"metatrader_id":"","dukascopy_id":"USSC2000IDXUSD"},
    # "VOLIDXUSD":{"decimal_divide":1e+3,"pip_size":0.0001,"metatrader_id":"","dukascopy_id":"VOLIDXUSD"},
    # #? energy:
    # "DIESELCMDUSD":{"decimal_divide":1e+3,"pip_size":0.0001,"metatrader_id":"","dukascopy_id":"DIESELCMDUSD"},
    # "BRENTCMDUSD":{"decimal_divide":1e+3,"pip_size":0.0001,"metatrader_id":"","dukascopy_id":"BRENTCMDUSD"},
    # "LIGHTCMDUSD":{"decimal_divide":1e+3,"pip_size":0.0001,"metatrader_id":"","dukascopy_id":"LIGHTCMDUSD"},
    # "GASCMDUSD":{"decimal_divide":1e+3,"pip_size":0.0001,"metatrader_id":"NG-Cr","dukascopy_id":"GASCMDUSD"},
}
