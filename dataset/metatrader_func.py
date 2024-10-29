import pandas as pd
import pytz
import MetaTrader5 as mt5
from dataset.configs.history_data_crawlers_config import metatrader_number_of_days
from dataset.utils.df_utils import ffill_df_to_true_time_steps
from dataset.logging_tools import default_logger
from dotenv import load_dotenv
from pathlib import Path
from dataset.configs.history_data_crawlers_config import (
    data_folder,
    symbols_dict,
)
import os

load_dotenv()
broker_path = os.environ.get("BROKER_PATH")
timezone = pytz.timezone("Etc/UTC")


def initialize_login_metatrader(logger=default_logger, broker_path = broker_path):
    # establish MetaTrader 5 connection to a specified trading account
    if not mt5.initialize(broker_path):
        print("initialize() failed, error code =",mt5.last_error())
        error_msg="!!! ERROR in initializing metatrader."
        logger.error(f"--> {error_msg}")
        raise ValueError(error_msg)
    # mt5 = MetaTrader5(
    #     host=os.getenv('METATRADER_HOST'),
    #     port=os.getenv('METATRADER_PORT'),
    # )
    logger.info(f"--> metatrader connected.")



def crawl_data_from_metatrader(
    mt5, symbol, timeframe, number_of_days, forward_fill=False,logger=default_logger
):
    """
    crawl data to pandas dataframe.
    min date is = 2023-01-01
    """

    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 288 * number_of_days)
    rates_df = pd.DataFrame(rates)
    if rates_df.shape[0] == 0:
        logger.warning(f"!!! metatrader: {symbol} no new data to crawl.")
        rates_df["_time"] = None
        return rates_df

    rates_df.rename(columns={"time": "_time"}, inplace=True)
    rates_df["_time"] = pd.to_datetime(rates_df["_time"], unit="s")

    if forward_fill and rates_df.shape[0] > 0:
        rates_df = ffill_df_to_true_time_steps(rates_df)

    return rates_df


def get_symbols_info(mt5,logger=default_logger):
    """
    Get all financial instruments info from the MetaTrader 5 terminal.

    """
    # ? list if vilable symbols:
    symbols_dict = mt5.symbols_get()
    symbols_dict[0]

    symbols_info = {}
    for item in symbols_dict:
        temp_dict = {
            "description": item.description,
            "name": item.name,
            "path": item.path,
            "currency_base": item.currency_base,
            "currency_profit": item.currency_profit,
            "currency_margin": item.currency_margin,
            "digits": item.digits,
        }
        symbols_info[item.name] = temp_dict

    logger.info(f"--> number of all symbols: {len(symbols_info)}")
    return symbols_info

def crawl_OHLCV_data_metatrader_one_symbol(
    mt5, symbol, number_of_days, forward_fill=False
):
    """
    get data for realtime loop.

    """
    timeframe = mt5.TIMEFRAME_M5

    df = (
        crawl_data_from_metatrader(
            mt5, symbol, timeframe, number_of_days, forward_fill=forward_fill
        )
        .sort_values("_time")
        .reset_index(drop=True)
    )
    return df


def crawl_OHLCV_data_metatrader(
    feature_config: dict, logger=default_logger, number_of_days: int =metatrader_number_of_days, forward_fill :bool =False,
):
    logger.info(f"= " * 25)
    logger.info(f"--> start crawl_OHLCV_data_metatrader fumc:")
    # key_name = "rawdata_metatrader"
    # table_name = databese_tables[key_name]["table_name"]
    # overlap_days = databese_tables[key_name]["overlap_days"]

    initialize_login_metatrader()
    data_source = "metatrader"
    folder_name = f"{data_folder}/{data_source}/"
    Path(folder_name).mkdir(parents=True, exist_ok=True)
    timeframe = mt5.TIMEFRAME_M5

    for symbol in list(feature_config.keys()):
        logger.info(f"--> symbol: {symbol}")
        file_name = f"{folder_name}/{symbol}_{data_source}.parquet"
        symbol = symbols_dict[symbol]["metatrader_id"]
        df = crawl_data_from_metatrader(
            mt5,symbol, timeframe, number_of_days, forward_fill=forward_fill
        )
        if df.shape[0] == 0:
            logger.info(f"!!! no data for {symbol} | skip this item")
            continue

        df = df.sort_values("_time").reset_index(drop=True)
        df["data_source"] = data_source
        df["symbol"] = symbol
        df.to_parquet(file_name, index=False)

    logger.info(f"--> crawl_OHLCV_data_metatrader run successfully.")
    return True


if __name__ == "__main__":
    from configs.feature_configs_general import generate_general_config
    config_general = generate_general_config()
    crawl_OHLCV_data_metatrader(config_general)
    default_logger.info(f"--> crawl_OHLCV_data_metatrader DONE.")
