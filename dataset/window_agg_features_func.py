from pathlib import Path
from dataset.configs.history_data_crawlers_config import root_path, symbols_dict
import pandas as pd
import numpy as np
from dataset.logging_tools import default_logger

def cal_window_stats(array, window_size):
    res = np.zeros([array.shape[0], 11])
    res[:window_size, :] = np.nan
    for i in range(window_size, array.shape[0]):
        selected_slice = array[i - window_size + 1 : i + 1]
        res[i, :] = [
            np.min(selected_slice),
            1 - (np.argmin(selected_slice) / (window_size - 1)),
            np.max(selected_slice),
            1 - (np.argmax(selected_slice) / (window_size - 1)),
            np.mean(selected_slice),
            np.std(selected_slice),
            pd.Series(selected_slice).skew(),
            pd.Series(selected_slice).kurtosis(),
            np.median(selected_slice),
            np.percentile(selected_slice, 25),
            np.percentile(selected_slice, 75)
        ]
    return res

def add_win_fe_base_func(
    df, symbol, raw_features, timeframes, window_sizes, round_to=3, fe_prefix="fe_WIN"
):
    for tf in timeframes:
        for w_size in window_sizes:
            assert tf == 5, "!!! for now, this code only works with 5M timeframe, tf must be 5."

            col_min = f"{fe_prefix}_min_W{w_size}_M{tf}"
            col_argmin = f"{fe_prefix}_argmin_W{w_size}_M{tf}"
            col_max = f"{fe_prefix}_max_W{w_size}_M{tf}"
            col_argmax = f"{fe_prefix}_argmax_W{w_size}_M{tf}"
            col_mean = f"{fe_prefix}_mean_W{w_size}_M{tf}"
            col_std = f"{fe_prefix}_std_W{w_size}_M{tf}"
            col_skew = f"{fe_prefix}_skew_W{w_size}_M{tf}"
            col_kurt = f"{fe_prefix}_kurt_W{w_size}_M{tf}"
            col_median = f"{fe_prefix}_median_W{w_size}_M{tf}"
            col_q25 = f"{fe_prefix}_q25_W{w_size}_M{tf}"
            col_q75 = f"{fe_prefix}_q75_W{w_size}_M{tf}"

            array = df[raw_features].to_numpy()
            res = cal_window_stats(array, w_size)
            
            df[col_min] = (df["M5_CLOSE"] - res[:, 0]) / symbols_dict[symbol]["pip_size"]
            df[col_argmin] = np.round(res[:, 1], round_to)
            df[col_max] = (res[:, 2] - df["M5_CLOSE"]) / symbols_dict[symbol]["pip_size"]
            df[col_argmax] = np.round(res[:, 3], round_to)
            df[col_mean] = np.round(res[:, 4], round_to)
            df[col_std] = np.round(res[:, 5], round_to)
            df[col_skew] = np.round(res[:, 6], round_to)
            df[col_kurt] = np.round(res[:, 7], round_to)
            df[col_median] = np.round(res[:, 8], round_to)
            df[col_q25] = np.round(res[:, 9], round_to)
            df[col_q75] = np.round(res[:, 10], round_to)
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
            df = pd.read_parquet(file_name, columns=needed_columns)
            df.sort_values("_time", inplace=True)

            df["_time"] = df["_time"].dt.tz_localize(None)
            df.drop(columns=["symbol"], inplace=True)
            df.sort_values("_time", inplace=True)

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
            df["symbol"] = symbol
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
