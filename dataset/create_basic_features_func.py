import polars as pl
import pandas as pd
from dataset.configs.history_data_crawlers_config import root_path, symbols_dict
from datetime import timedelta
from dataset.logging_tools import default_logger
from pathlib import Path
from feature_engine.datetime import DatetimeFeatures


def add_candle_features(
    df, symbol: str, tf_list=[5, 15, 60, 120, 240, 1440], fe_prefix="fe_cndl"
):
    org_columns = set(df.columns)
    org_columns.remove("_time")
    for tf in tf_list:
        df = df.with_columns(
            [
                (
                    (pl.col(f"M{tf}_CLOSE") - pl.col(f"M{tf}_OPEN"))
                    / symbols_dict[symbol]["pip_size"]
                ).alias(f"{fe_prefix}_M{tf}_CLOSE_to_OPEN"),
                (
                    (pl.col(f"M{tf}_HIGH") - pl.col(f"M{tf}_LOW"))
                    / symbols_dict[symbol]["pip_size"]
                ).alias(f"{fe_prefix}_M{tf}_HIGH_to_LOW"),
                (
                    (pl.col(f"M{tf}_HIGH") - pl.col(f"M{tf}_CLOSE"))
                    / symbols_dict[symbol]["pip_size"]
                ).alias(f"{fe_prefix}_M{tf}_HIGH_to_CLOSE"),
                (
                    (pl.col(f"M{tf}_HIGH") - pl.col(f"M{tf}_OPEN"))
                    / symbols_dict[symbol]["pip_size"]
                ).alias(f"{fe_prefix}_M{tf}_HIGH_to_OPEN"),
                (
                    (pl.col(f"M{tf}_OPEN") - pl.col(f"M{tf}_LOW"))
                    / symbols_dict[symbol]["pip_size"]
                ).alias(f"{fe_prefix}_M{tf}_OPEN_to_LOW"),
                (
                    (pl.col(f"M{tf}_CLOSE") - pl.col(f"M{tf}_LOW"))
                    / symbols_dict[symbol]["pip_size"]
                ).alias(f"{fe_prefix}_M{tf}_CLOSE_to_LOW"),
            ]
        )

    df = df.drop(org_columns)

    return df

def add_shifted_candle_features(
    df, tf, shift_sizes=[1], fe_prefix="fe_cndl_shift"
):
    org_columns = set(df.columns)
    org_columns.remove("_time")
    for sh in shift_sizes:
        df = df.with_columns(
            [
                (pl.col(f"M{tf}_CLOSE_-{sh}") - pl.col(f"M{tf}_OPEN_-{sh}")).alias(
                    f"{fe_prefix}_M{tf}_CLOSE_to_OPEN_-{sh}"
                ),
                (pl.col(f"M{tf}_HIGH_-{sh}") - pl.col(f"M{tf}_LOW_-{sh}")).alias(
                    f"{fe_prefix}_M{tf}_HIGH_to_LOW_-{sh}"
                ),
                (pl.col(f"M{tf}_HIGH_-{sh}") - pl.col(f"M{tf}_CLOSE_-{sh}")).alias(
                    f"{fe_prefix}_M{tf}_HIGH_to_CLOSE_-{sh}"
                ),
                (pl.col(f"M{tf}_HIGH_-{sh}") - pl.col(f"M{tf}_OPEN_-{sh}")).alias(
                    f"{fe_prefix}_M{tf}_HIGH_to_OPEN_-{sh}"
                ),
                (pl.col(f"M{tf}_OPEN_-{sh}") - pl.col(f"M{tf}_LOW_-{sh}")).alias(
                    f"{fe_prefix}_M{tf}_OPEN_to_LOW_-{sh}"
                ),
                (pl.col(f"M{tf}_CLOSE_-{sh}") - pl.col(f"M{tf}_LOW_-{sh}")).alias(
                    f"{fe_prefix}_M{tf}_CLOSE_to_LOW_-{sh}"
                ),
            ]
        )

    df = df.drop(*org_columns)

    return df

def history_basic_features(feature_config, logger=default_logger):
    logger.info("- " * 25)
    logger.info("--> start history_basic_features fumc:")
    try:

        fe_prefix = "fe_cndl"
        features_folder_path = f"{root_path}/data/features/{fe_prefix}/"
        Path(features_folder_path).mkdir(parents=True, exist_ok=True)


        for symbol in list(feature_config.keys()):
            if fe_prefix not in list(feature_config[symbol].keys()):
                continue
            logger.info(f"- " * 30)
            logger.info(f"--> {fe_prefix}, {symbol}:")
          
            tf_list = feature_config[symbol]["fe_cndl"]
            file_name = features_folder_path + f"/{fe_prefix}_{symbol}.parquet"
            df = pl.read_parquet(
                f"{root_path}/data/realtime_candle/{symbol}_realtime_candle.parquet"
            )
               
            df = df.sort("_time").drop("symbol")
            df = add_candle_features(
                df, symbol, tf_list=tf_list, fe_prefix=fe_prefix
            )
            df = df.with_columns(pl.lit(symbol).alias("symbol"))
            df.write_parquet(file_name)
            logger.info(f"--> fe_cndl {symbol} saved.")

        # ----------------------------------
        fe_prefix = "fe_cndl_shift"
        features_folder_path = f"{root_path}/data/features/{fe_prefix}/"
        Path(features_folder_path).mkdir(parents=True, exist_ok=True)

        for symbol in list(feature_config.keys()):
            if fe_prefix not in list(feature_config[symbol].keys()):
                continue

            logger.info(f"- " * 30)
            logger.info(f"--> {fe_prefix}, {symbol} :")
            
            df_sh = pl.read_parquet(
                f"{root_path}/data/features/fe_cndl_shift_raw/{fe_prefix}_{symbol}.parquet"
            )

            file_name = features_folder_path + f"/{fe_prefix}_{symbol}.parquet"
            shift_configs = feature_config[symbol]["fe_cndl_shift"]["shift_configs"]
            
            df_sh = df_sh.sort("_time")
            df_all = df_sh[["_time"]]

            for shift_config in shift_configs:
                timeframe = shift_config["timeframe"]

                shift_sizes = shift_config["shift_sizes"]
                df = add_shifted_candle_features(
                    df_sh,
                    tf=timeframe,
                    shift_sizes=shift_sizes,
                    fe_prefix=fe_prefix,
                )
                df_all = df_all.join(df, on="_time", how="left", coalesce=True)

            
            df_all = df_all.with_columns(pl.lit(symbol).alias("symbol"))
            df_all.write_parquet(file_name)
            logger.info(f"--> fe_cndl_shift_stage_2 {symbol} saved.")

        logger.info("--> history_basic_features run successfully.")
    except Exception as e:
        logger.exception("--> history_basic_features error.")
        logger.exception(f"--> error: {e}")
        raise ValueError("!!!")

def history_fe_market_close(feature_config, logger=default_logger):
    logger.info("- " * 25)
    logger.info("--> start history_fe_market_close fumc:")
    try:
        fe_prefix = "fe_market_close"
        features_folder_path = f"{root_path}/data/features/{fe_prefix}/"
        Path(features_folder_path).mkdir(parents=True, exist_ok=True)

        ##? in EST time zone
        markets_trade_times = {
            "New_York": {
                "hour": 16,
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
            "London": {
                "hour": 10,
                "minute": 55,
            },
        }

        for symbol in list(feature_config.keys()):
            logger.info(f"-->{symbol}")
            logger.info("= " * 40)

            columns = ["_time", "close"]
            df=pd.read_parquet(f"{root_path}/data/stage_one_data/{symbol}_stage_one.parquet",columns=columns)
            df["_time"] = df["_time"].dt.tz_localize(None)
            df.sort_values("_time", inplace=True)
            df.reset_index(drop=True, inplace=True)
            df["_time"] = df["_time"] - timedelta(hours=7)

            for market in markets_trade_times:
                fiter = (
                    df["_time"].dt.hour == markets_trade_times[market]["hour"]
                ) & (df["_time"].dt.minute == markets_trade_times[market]["minute"])


                df["last_close_price"] = None
                df.loc[fiter, "last_close_price"] = df.loc[fiter, "close"]

                df["last_close_time"] = None
                df.loc[fiter, "last_close_time"] = df.loc[fiter, "_time"]
                with pd.option_context("future.no_silent_downcasting", True):
                    df = df.ffill(inplace=False).infer_objects(copy=False)

                
                df[f"{fe_prefix}_{market}"] = (
                    df["close"] - df["last_close_price"]
                ) / symbols_dict[symbol]["pip_size"]
                df[f"{fe_prefix}_{market}_time"] = (
                    df["_time"] - df["last_close_time"]
                ).dt.seconds // 60

            ##? parquet save:
            df.drop(
                columns=["last_close_price", "close", "last_close_time"],
                inplace=True,
            )
            df["_time"] = df["_time"] + timedelta(hours=7)
            df.dropna(inplace=True)
            df.reset_index(drop=True, inplace=True)
            df["symbol"] = symbol
            file_name = f"{features_folder_path}/{fe_prefix}_{symbol}.parquet"
            df.to_parquet(file_name,index=False)

        logger.info("--> history_fe_market_close run successfully.")
    except Exception as e:
        logger.exception("--> history_fe_market_close error.")
        logger.exception(f"--> error: {e}")
        raise ValueError("!!!")

def history_fe_time(feature_config, logger=default_logger):
    logger.info("- " * 25)
    logger.info("--> start history_fe_time fumc:")
    try:
        prefix = "fe_time"
        symbol = list(feature_config.keys())[0]
        df= pd.read_parquet(f"{root_path}/data/stage_one_data/EURUSD_stage_one.parquet", columns=["_time"]).sort_values("_time").reset_index(drop=True)
        df["_time"] = df["_time"].dt.tz_localize(None)
        df.sort_values("_time", inplace=True)
        df.reset_index(drop=True, inplace=True)
        df["_time"] = df["_time"] - timedelta(hours=7)

        dtf = DatetimeFeatures(
            features_to_extract=[
                "month",
                "quarter",
                "semester",
                "week",
                "day_of_week",
                "day_of_month",
                "day_of_year",
                "month_start",
                "month_end",
                "quarter_start",
                "quarter_end",
                "year_start",
                "year_end",
                "hour",
                "minute",
            ],
            drop_original=False,
        )

        dtf.fit(df.rename(columns={"_time": prefix}))
        fe_time = dtf.transform(df.rename(columns={"_time": prefix})).rename(
            columns={prefix: "_time"}
        )
    
        markets_trade_times = {
            "New_York": (8, 17),
            "Tokyo": (19, 4),
            "Sydney": (15, 0),
            "London": (3, 11),
        }
        
        for market_name in markets_trade_times:
            start_time = markets_trade_times[market_name][0]
            stop_time = markets_trade_times[market_name][1]

            col_name = f"{prefix}_isin_{market_name}"
            fe_time[col_name] = 0

            if stop_time > start_time:
                fe_time.loc[
                    (fe_time["_time"].dt.hour >= start_time)
                    & (fe_time["_time"].dt.hour < stop_time),
                    col_name,
                ] = 1

            else:
                fe_time.loc[
                    (fe_time["_time"].dt.hour >= start_time)
                    | (fe_time["_time"].dt.hour < stop_time),
                    col_name,
                ] = 1

        fe_time["_time"] = fe_time["_time"] + timedelta(hours=7)
        fe_time["symbol"] = symbol
        features_folder_path = f"{root_path}/data/features/{prefix}/"
        Path(features_folder_path).mkdir(parents=True, exist_ok=True)
        file_name = features_folder_path + f"/{prefix}.parquet"
        fe_time.to_parquet(file_name,index=False)

        logger.info("--> history_fe_time run successfully.")
    except Exception as e:
        logger.exception("--> history_fe_time error.")
        logger.exception(f"--> error: {e}")
        raise ValueError("!!!")


if __name__ == "__main__":
    from configs.feature_configs_general import generate_general_config
    config_general = generate_general_config()

    history_basic_features(config_general)
    print(f"--> history_basic_features DONE.")

    history_fe_market_close(config_general)
    print(f"--> read_feature_config DONE.")

    history_fe_time(config_general)
    print(f"--> read_feature_config DONE.")
