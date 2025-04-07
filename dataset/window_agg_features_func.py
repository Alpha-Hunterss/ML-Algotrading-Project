from pathlib import Path
from dataset.configs.history_data_crawlers_config import root_path, symbols_dict
import pandas as pd
import numpy as np
from dataset.logging_tools import default_logger
from pathlib import Path
from dataset.configs.history_data_crawlers_config import root_path, symbols_dict
import pandas as pd
import numpy as np


def cal_window_max(array, window_size):
    res = np.zeros([array.shape[0], 4])
    res[:window_size, :] = np.nan
    for i in range(window_size, array.shape[0]):
        selected_slice = array[i - window_size + 1 : i + 1]

        res[i, :] = [
            np.min(selected_slice),
            1 - (np.argmin(selected_slice) / (window_size - 1)),
            np.max(selected_slice),
            1 - (np.argmax(selected_slice) / (window_size - 1)),
        ]

    return res


def add_win_fe_base_func(
    df, symbol, raw_features, timeframes, window_sizes, round_to=3, fe_prefix="fe_WIN"
):
    for tf in timeframes:
        for w_size in window_sizes:
            assert (
                tf == 5
            ), "!!! for now, this code only work with 5M timeframe, tf must be 5."
            # if int(tf) != 5:
            #     df = df[df.minutesPassed == tf -5]

            col_min = f"{fe_prefix}_min_W{w_size}_M{tf}"
            col_argmin = f"{fe_prefix}_argmin_W{w_size}_M{tf}"
            col_max = f"{fe_prefix}_max_W{w_size}_M{tf}"
            col_argmax = f"{fe_prefix}_argmax_W{w_size}_M{tf}"

            array = df[raw_features].to_numpy()
            # logger.info(col_max, col_max)
            res = cal_window_max(array, w_size)
            df[col_min] = (df["M5_CLOSE"] - res[:, 0]) / symbols_dict[symbol][
                "pip_size"
            ]
            df[col_argmin] = np.round(res[:, 1], round_to)
            df[col_max] = (res[:, 2] - df["M5_CLOSE"]) / symbols_dict[symbol][
                "pip_size"
            ]
            df[col_argmax] = np.round(res[:, 3], round_to)

    return df

def history_fe_WIN_features(feature_config, logger=default_logger):

    logger.info("- " * 25)
    logger.info("--> start history_fe_WIN_features fumc:")
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
            df.drop(columns=["symbol"])
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
