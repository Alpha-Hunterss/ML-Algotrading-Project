import numpy as np
import pandas as pd
from pathlib import Path
from dataset.configs.history_data_crawlers_config import root_path, symbols_dict
from dataset.logging_tools import default_logger

def hurst_exponent(ts, max_lag=100):
    """Calculate the Hurst exponent of a time series using a log-log regression method."""
    lags = np.arange(2, min(len(ts) // 2, max_lag))
    tau = np.array([np.std(ts[lag:] - ts[:-lag]) for lag in lags])
    return np.polyfit(np.log(lags), np.log(tau), 1)[0]

def cal_window_stats(series, window_size):
    """Compute rolling statistics using NumPy and Pandas."""
    roll = series.rolling(window=window_size, min_periods=1)

    return pd.DataFrame({
        "min": roll.min(),
        "argmin": (window_size - 1 - roll.apply(np.argmin, raw=True)) / (window_size - 1),
        "max": roll.max(),
        "argmax": (window_size - 1 - roll.apply(np.argmax, raw=True)) / (window_size - 1),
        "mean": roll.mean(),
        "std": roll.std(),
        "skew": roll.skew(),
        "kurt": roll.kurt(),
        "median": roll.median(),
        "q25": roll.quantile(0.25),
        "q75": roll.quantile(0.75),
        "hurst": roll.apply(lambda x: hurst_exponent(x.to_numpy()) if len(x.dropna()) > 10 else np.nan, raw=False),
    })

def add_win_fe_base_func(df, symbol, raw_features, timeframes, window_sizes, round_to=3, fe_prefix="fe_WIN"):
    for tf in timeframes:
        for w_size in window_sizes:
            assert tf == 5, "!!! Only works with 5M timeframe."

            feature_names = [
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
                f"{fe_prefix}_hurst_W{w_size}_M{tf}",
            ]

            # Compute rolling statistics
            for col in raw_features:
                stats = cal_window_stats(df[col], w_size).round(round_to)
                stats.columns = feature_names
                df = pd.concat([df, stats], axis=1)
    
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
            df = pd.read_parquet(file_name, columns=needed_columns).sort_values("_time")
            
            df["_time"] = pd.to_datetime(df["_time"])
            df["symbol"] = symbol  # Assign symbol column after processing

            df = add_win_fe_base_func(
                df,
                symbol,
                raw_features=raw_features,
                timeframes=feature_config[symbol][fe_prefix]["timeframe"],
                window_sizes=feature_config[symbol][fe_prefix]["window_size"],
                round_to=round_to,
                fe_prefix="fe_WIN",
            )
            
            df.drop(columns=raw_features + ["minutesPassed"], inplace=True)
            df.to_parquet(f"{features_folder_path}/{fe_prefix}_{symbol}.parquet")
        
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
