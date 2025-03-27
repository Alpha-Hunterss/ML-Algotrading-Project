from pathlib import Path
from dataset.configs.history_data_crawlers_config import root_path, symbols_dict
import polars as pl
import numpy as np
import numba
from dataset.logging_tools import default_logger

def hurst_exponent(ts, max_lag=100):
    """Calculate the Hurst exponent of a time series using a log-log regression method."""
    lags = np.arange(2, max_lag)
    tau = np.array([np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags])
    return np.polyfit(np.log(lags), np.log(tau), 1)[0]

@numba.jit(nopython=True, parallel=True)
def cal_window_stats(array, window_size):
    res = np.empty((array.shape[0], 12))  # Increase to store the Hurst exponent
    res[:window_size, :] = np.nan
    for i in numba.prange(window_size, array.shape[0]):
        selected_slice = array[i - window_size + 1 : i + 1]
        res[i, :] = [
            np.min(selected_slice),
            1 - (np.argmin(selected_slice) / (window_size - 1)),
            np.max(selected_slice),
            1 - (np.argmax(selected_slice) / (window_size - 1)),
            np.mean(selected_slice),
            np.std(selected_slice),
            pl.Series(selected_slice).skew(),
            pl.Series(selected_slice).kurtosis(),
            np.median(selected_slice),
            np.percentile(selected_slice, 25),
            np.percentile(selected_slice, 75),
            hurst_exponent(selected_slice)  # Compute Hurst exponent
        ]
    return res

def add_win_fe_base_func(
    df, symbol, raw_features, timeframes, window_sizes, round_to=3, fe_prefix="fe_WIN"
):
    for tf in timeframes:
        for w_size in window_sizes:
            assert tf == 5, "!!! for now, this code only works with 5M timeframe, tf must be 5."
            
            col_names = [
                f"{fe_prefix}_min_W{w_size}_M{tf}",
                f"{fe_prefix}_argmin_W{w_size}_M{tf}",
                f"{fe_prefix}_max_W{w_size}_M{tf}",
                f"{fe_prefix}_argmax_W{w_size}_M{tf}",
                f"{fe_prefix}_mean_W{w_size}_M{tf}",
                f"{fe_prefix}_std_W{w_size}_M{tf}",
                f"{fe_prefix}_skew_W{w_size}_M{tf}",
                f"{fe_prefix}_kurt_W{w_size}_M{tf}",
                f"{fe_prefix}_median_W{w_size}_M{tf}",
                f"{fe_prefix}_q25_W{w_size}_M{tf}",
                f"{fe_prefix}_q75_W{w_size}_M{tf}",
                f"{fe_prefix}_hurst_W{w_size}_M{tf}"
            ]
            
            array = df[raw_features].to_numpy()
            res = cal_window_stats(array, w_size)
            df = df.with_columns(
                [pl.Series(col, res[:, i]).round(round_to) for i, col in enumerate(col_names)]
            )
    return df

def history_fe_WIN_features(feature_config, logger=default_logger):
    logger.info("- " * 25)
    logger.info("--> start history_fe_WIN_features func:")
    try:
        fe_prefix = "fe_WIN"
        features_folder_path = f"{root_path}/data/features/{fe_prefix}/"
        Path(features_folder_path).mkdir(parents=True, exist_ok=True)
        base_candle_folder_path = f"{root_path}/data/realtime_candle/"
        round_to = 4

        for symbol in list(feature_config.keys()):
            logger.info(f"---> symbol: {symbol}")
            logger.info("= " * 40)

            base_cols = feature_config[symbol][fe_prefix]["base_columns"]
            raw_features = [f"M5_{base_col}" for base_col in base_cols]
            needed_columns = ["_time", "minutesPassed", "symbol"] + raw_features
            file_name = base_candle_folder_path + f"{symbol}_realtime_candle.parquet"
            df = pl.read_parquet(file_name).select(needed_columns).sort("_time")
            
            df = df.with_columns(
                df["_time"].cast(pl.Datetime).alias("_time"),
                df["symbol"].cast(pl.Utf8).alias("symbol")
            ).drop("symbol")
            
            df = add_win_fe_base_func(
                df,
                symbol,
                raw_features=raw_features,
                timeframes=feature_config[symbol][fe_prefix]["timeframe"],
                window_sizes=feature_config[symbol][fe_prefix]["window_size"],
                round_to=round_to,
                fe_prefix="fe_WIN",
            )
            
            df = df.drop(raw_features + ["minutesPassed"]).with_columns(pl.lit(symbol).alias("symbol"))
            df.write_parquet(f"{features_folder_path}/{fe_prefix}_{symbol}.parquet")
        logger.info("--> history_fe_WIN_features run successfully.")
    except Exception as e:
        logger.exception("--> history_fe_WIN_features error.")
        logger.exception(f"--> error: {e}")
        raise ValueError("!!!")

if __name__ == "__main__":
    from configs.feature_configs_general import generate_general_config
    config_general = generate_general_config()
    history_fe_WIN_features(config_general)
    default_logger.info(f"--> history_fe_WIN_features DONE.")
