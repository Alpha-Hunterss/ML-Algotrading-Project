import polars as pl
from pathlib import Path
from dataset.configs.history_data_crawlers_config import symbols_dict
import os, glob
from typing import Callable, Dict, List, Tuple, Union
from dataset.configs.history_data_crawlers_config import root_path
import re
from dataset.logging_tools import default_logger


# ?? indicator ---------------------------------------------------

def cal_cndl_shape_n_cntxt_func(
    df: pl.DataFrame,
    w: int,
    time_frame: int,
    features: List[str],
    pip_size: float,
    prefix: str = "fe_cndl_shape_n_cntxt",
    normalize: bool = True,
) -> pl.DataFrame:
    """
    Calculates candle shape features.

    Args:
        df: The input DataFrame.
        w: Window size.
        time_frame: Timeframe of the candle.
        features: A list of features, including 'OPEN', 'HIGH', 'LOW', and 'CLOSE'.
        pip_size: Pip size for normalization.
        prefix: Prefix for the new feature columns.
        normalize: Whether to normalize the features.

    Returns:
        The input DataFrame with added candle shape features.
    """
    assert (
        len(features) == 4
    ), f"Only 4 feature should have been passed but {len(features)} received!"
    features = sorted(features)
    input_features = [
        f'M{time_frame}_CLOSE',
        f'M{time_frame}_HIGH',
        f'M{time_frame}_LOW',
        f'M{time_frame}_OPEN'
    ]
    if features != input_features:
        print('Input features are wrong')
        return
    # features[0] == f'M{time_frame}_CLOSE'
    # features[1] == f'M{time_frame}_HIGH'
    # features[2] == f'M{time_frame}_LOW'
    # features[3] == f'M{time_frame}_OPEN'

    df = df.sort("_time")

    # Determine higher and lower price (among OPEN & CLOSE)
    df = df.with_columns([
        pl.when(
            pl.col(features[3]) > pl.col(features[0])
        )
        .then(pl.col(features[3]))
        .otherwise(pl.col(features[0]))
        .alias(f"{prefix}_higher_price_M{time_frame}"),
        pl.when(
            pl.col(features[3]) < pl.col(features[0])
        )
        .then(pl.col(features[3]))
        .otherwise(pl.col(features[0]))
        .alias(f"{prefix}_lower_price_M{time_frame}")
    ]).lazy()

    # Calculate candle return, body, upper and lower shadows
    if normalize:

        context_features = [
            f"{prefix}_return_M{time_frame}_norm",
            f"{prefix}_up_shadow_M{time_frame}_norm",
            f"{prefix}_down_shadow_M{time_frame}_norm",
            f"{prefix}_body_length_M{time_frame}_norm"
        ]

        df = df.with_columns([
            (
                (
                    pl.col(features[0]) - pl.col(features[0]).shift(1)
                )
                * 1000 / (
                    pip_size * pl.col(features[0]).shift(1)
                )
            )
            .alias(context_features[0]),
            (
                (
                    pl.col(features[1])
                    - pl.col(
                        f"{prefix}_higher_price_M{time_frame}"
                    )
                ) * 1000 / (pip_size * pl.col(features[0]))
            )
            .alias(context_features[1]),
            (
                (
                    pl.col(
                        f"{prefix}_lower_price_M{time_frame}"
                    )
                    - pl.col(features[2])
                ) * 1000 / (pip_size * pl.col(features[0]))
            )
            .alias(context_features[2]),
            (
                (
                    pl.col(
                        f"{prefix}_higher_price_M{time_frame}"
                    )
                    - pl.col(
                        f"{prefix}_lower_price_M{time_frame}"
                    )
                ) * 1000 / (pip_size * pl.col(features[0]))
            )
            .alias(context_features[3]),
            (
                pl.col(features[1]) - pl.col(features[2])
            )
            .alias(f"{prefix}_candle_length_M{time_frame}"),
        ]).lazy()

    else:
        context_features = [
            f"{prefix}_return_M{time_frame}",
            f"{prefix}_up_shadow_M{time_frame}",
            f"{prefix}_down_shadow_M{time_frame}",
            f"{prefix}_body_length_M{time_frame}"
        ]

        df = df.with_columns([
            (
                pl.col(features[0]) - pl.col(features[0]).shift(1)
            )
            .alias(context_features[0]),
            (
                pl.col(features[1])
                - pl.col(
                    f"{prefix}_higher_price_M{time_frame}"
                )
            )
            .alias(context_features[1]),
            (
                pl.col(f"{prefix}_lower_price_M{time_frame}")
                - pl.col(features[2])
            )
            .alias(context_features[2]),
            (
                pl.col(f"{prefix}_higher_price_M{time_frame}")
                - pl.col(
                    f"{prefix}_lower_price_M{time_frame}"
                )
            )
            .alias(context_features[3]),
            (
                pl.col(features[1]) - pl.col(features[2])
            )
            .alias(f"{prefix}_candle_length_M{time_frame}"),
        ]).lazy()

    # Calculate tercile levels
    df = df.with_columns([
        (
            pl.col(features[2]) + pl.col(f"{prefix}_candle_length_M{time_frame}") / 3
        )
        .alias(f"{prefix}_lower_tercile_M{time_frame}"),
        (
            pl.col(features[1]) - pl.col(f"{prefix}_candle_length_M{time_frame}") / 3
        )
        .alias(f"{prefix}_upper_tercile_M{time_frame}")
    ]).lazy()

    # Identify pin bars
    df = df.with_columns([
        pl.when(
            pl.col(f"{prefix}_lower_price_M{time_frame}") > pl.col(f"{prefix}_upper_tercile_M{time_frame}")
        ).then(1)
        .otherwise(0)
        .alias(f"{prefix}_is_bullish_pin_bar_M{time_frame}"),
        pl.when(
            pl.col(f"{prefix}_higher_price_M{time_frame}") < pl.col(f"{prefix}_lower_tercile_M{time_frame}")
        ).then(1)
        .otherwise(0)
        .alias(f"{prefix}_is_bearish_pin_bar_M{time_frame}")
    ]).lazy()

    # Create lagged features for historical context
    for feature in context_features:
        for lag in range(1, w):  # w-1 previous candles
            df = df.with_columns([
                pl.col(feature).shift(lag).alias(f"{feature}_lag_{lag}")
            ]).lazy()

    # Drop unnecessary columns
    cols_to_drop = features
    cols_to_drop.extend([
        f"{prefix}_higher_price_M{time_frame}",
        f"{prefix}_lower_price_M{time_frame}",
        f"{prefix}_lower_tercile_M{time_frame}",
        f"{prefix}_upper_tercile_M{time_frame}",
        f"{prefix}_candle_length_M{time_frame}"
    ])
    df = df.collect()
    df = df.drop(cols_to_drop)

    return df


def cal_RSI_base_func(
    df: pl.DataFrame,
    w: int,
    time_frame: int,
    features: List[str],
    pip_size: float,  # only for compatibility
    prefix: str = "fe_RSI",
    percentage_feature: bool = False,
    add_30_70: bool = True,
) -> pl.DataFrame:
    """
    This function creates RSI feature
    inputs:
    df: dataframe containing the raw feature
    w: window size
    time_frame: time_frame for calculations
    feature: raw feature on which the RSI is calculated
    prefix: prefix of feature name
    percentage_feature: true for percentage features like price-percentage are diff features by nature
    add_30_70: add whether the RSI is above 70 or below 30 !

    To understand the code see the RSI formula
    https://www.wallstreetmojo.com/relative-strength-index/
    pandas version: https://github.com/twopirllc/pandas-ta/blob/main/pandas_ta/momentum/rsi.py

    """
    assert (
        len(features) == 1
    ), f"Only 1 feature should have been passed but {len(features)} received!"
    feature = features[0]

    df = df.sort("_time")
    if percentage_feature:
        # percentage features like price-percentage are diff features by nature
        df = df.with_columns((pl.col(feature)).alias(f"{feature}_diff")).lazy()
    else:
        df = df.with_columns((pl.col(feature).diff()).alias(f"{feature}_diff")).lazy()

    df = df.with_columns(
        ((pl.col(f"{feature}_diff") >= 0) * (pl.col(f"{feature}_diff"))).alias(
            f"{feature}_GAIN"
        )
    ).lazy()
    df = df.with_columns(
        ((pl.col(f"{feature}_diff") < 0) * -1 * (pl.col(f"{feature}_diff"))).alias(
            f"{feature}_LOSS"
        )
    ).lazy()

    df = df.with_columns(
        (
            pl.col(f"{feature}_GAIN").ewm_mean(
                alpha=1.0 / w, min_periods=w, ignore_nulls=True
            )
        ).alias(f"{feature}_Avg_GAIN_{w}")
    ).lazy()
    df = df.with_columns(
        (
            pl.col(f"{feature}_LOSS").ewm_mean(
                alpha=1.0 / w, min_periods=w, ignore_nulls=True
            )
        ).alias(f"{feature}_Avg_LOSS_{w}")
    ).lazy()

    # METHOD I
    df = df.with_columns(
        (
            (pl.col(f"{feature}_Avg_GAIN_{w}")) / ((pl.col(f"{feature}_Avg_LOSS_{w}")))
        ).alias(f"{feature}_RS_{w}")
    ).lazy()
    df = df.with_columns(
        (100 - (100 / (1 + pl.col(f"{feature}_RS_{w}")))).alias(
            f"{prefix}_{feature}_W{w}_cndl_M{time_frame}"
        )
    ).lazy()

    if add_30_70:
        df = df.with_columns(
            ((pl.col(f"{prefix}_{feature}_W{w}_cndl_M{time_frame}")) >= 70).alias(
                f"{prefix}_{feature}_W{w}_gte_70_cndl_M{time_frame}"
            )
        ).lazy()
        df = df.with_columns(
            ((pl.col(f"{prefix}_{feature}_W{w}_cndl_M{time_frame}")) <= 30).alias(
                f"{prefix}_{feature}_W{w}_lte_30_cndl_M{time_frame}"
            )
        ).lazy()

    df = df.drop(
        [
            f"{feature}",
            f"{feature}_diff",
            f"{feature}_GAIN",
            f"{feature}_LOSS",
            f"{feature}_Avg_GAIN_{w}",
            f"{feature}_Avg_LOSS_{w}",
            f"{feature}_RS_{w}",
        ],
    )
    return df.collect()


def cal_EMA_base_func(
    df: pl.DataFrame,
    w: int,
    time_frame: int,
    features: List[str],
    pip_size: float,
    prefix: str = "fe_EMA",
    normalize: bool = True,
) -> pl.DataFrame:
    """
    this function calculates exponantial moving average.
    inputs:
    df: dataframe containing the raw feature
    w: window size
    time_frame: time_frame for calculations
    feature: raw feature on which the RSI is calculated
    pip size: pip size of the pair
    prefix: prefix of feature name
    normalize: if True the function returns pipsize difference between EMA and last close price.

    """
    assert (
        len(features) == 1
    ), f"Only 1 feature should have been passed but {len(features)} received!"
    feature = features[0]

    df = df.sort("_time")

    if normalize:
        df = df.with_columns(
            (
                (
                    (pl.col(feature).ewm_mean(span=w, ignore_nulls=True))
                    - pl.col(feature)
                )
                / pip_size
            ).alias(f"{prefix}_{feature}_W{w}_cndl_M{time_frame}_norm")
        ).lazy()
    else:
        df = df.with_columns(
            (pl.col(feature).ewm_mean(span=w, ignore_nulls=True)).alias(
                f"{prefix}_{feature}_W{w}_cndl_M{time_frame}"
            )
        ).lazy()

    df = df.collect()

    df = df.drop([f"{feature}"])

    return df


def cal_SMA_base_func(
    df: pl.DataFrame,
    w: int,
    time_frame: int,
    features: List[str],
    pip_size: float,
    prefix: str = "fe_SMA",
    normalize: bool = True,
) -> pl.DataFrame:
    """
    this function calculates simple moving average.
    inputs:
    df: dataframe containing the raw feature
    w: window size
    time_frame: time_frame for calculations
    feature: raw feature on which the RSI is calculated
    pip size: pip size of the pair
    prefix: prefix of feature name
    normalize: if True the function returns pipsize difference between EMA and last close price.
    """
    assert (
        len(features) == 1
    ), f"Only 1 feature should have been passed but {len(features)} received!"
    feature = features[0]

    df = df.sort("_time")
    if normalize:
        df = df.with_columns(
            (
                ((pl.col(feature).rolling_mean(window_size=w)) - pl.col(feature))
                / pip_size
            ).alias(f"{prefix}_{feature}_W{w}_cndl_M{time_frame}_norm")
        ).lazy()

    else:
        df = df.with_columns(
            (pl.col(feature).rolling_mean(window_size=w)).alias(
                f"{prefix}_{feature}_W{w}_cndl_M{time_frame}"
            )
        ).lazy()
    df = df.collect()
    df = df.drop([f"{feature}"])

    return df


def add_candle_base_indicators_polars(
    df_base: pl.DataFrame,
    prefix: str,
    base_func: Callable[..., pl.DataFrame],
    opts: Dict[str, Union[str, List[int]]],
) -> None:
    """
    this function takes an indicator function, apply it and save the resulting parquet
    inputs:
    df_base: base dataframe containing the raw features
    prefix: prefix of feature name
    base_func: the indicator function
    opts: a dictionary of "symbol","base_feature","candle_timeframe","window_size" and "features_folder_path"
    """

    df_base = df_base.sort("_time")
    symbol = opts["symbol"]
    pip_size = symbols_dict[symbol]["pip_size"]
    features_folder_path = opts["features_folder_path"] + "/unmerged/"
    Path(features_folder_path).mkdir(parents=True, exist_ok=True)

    filelist = glob.glob(f"{features_folder_path}/*.parquet", recursive=True)
    for f in filelist:
        os.remove(f)

    features = opts["base_feature"]
    time_frames = opts["candle_timeframe"]
    window_sizes = opts["window_size"]

    for w in window_sizes:
        for time_frame in time_frames:
            df = df_base.filter(
                pl.col("minutesPassed") % time_frame == (time_frame - 5)
            )

            # Create a regex pattern to match 'M' followed by the time_frame number
            pattern = re.compile(rf"M{time_frame}_")

            # Find items where the number after 'M' is not equal to time_frame
            other_tf_features = [f for f in features if not pattern.match(f)]
            df = df.drop(other_tf_features + ["minutesPassed"])
            df = base_func(
                df=df,
                w=w,
                time_frame=time_frame,
                features=list(set(features) - set(other_tf_features)),
                pip_size=pip_size,
                prefix=prefix,
            )

            file_name = (
                features_folder_path + f"/{prefix}_{w}_{symbol}_M{time_frame}.parquet"
            )
            df.write_parquet(file_name)

    return


# ?? ratio  -----------------------------------------------------
def add_ratio_by_columns(
    df: pl.DataFrame, col_name_a: str, col_name_b: str, ratio_col_name
) -> pl.DataFrame:
    """
    this function calculates the ratio of two features
    inputs:
    df: dataframe containing the raw feature
    col_name_a: name of the first feature
    col_name_b: name of the second feature
    ratio_col_name: name of the ratio feature
    """
    df = df.with_columns(
        pl.when(pl.col(col_name_b) == 0)
        .then(0)  # or then(custom_value)
        .otherwise((pl.col(col_name_a) / pl.col(col_name_b)))
        .round(5)
        .alias(ratio_col_name)
    )

    return df


def add_ratio(
    df: pl.DataFrame,
    symbol: str,
    fe_name: str,
    timeframe: int,
    w1: int,
    w2: int,
    fe_prefix: str = "fe_ratio",
) -> pl.DataFrame:
    """
    this function takes whatever needed for defining ratio and then applies add_ratio_by_columns
    """

    if "RSI" in fe_name or "RSTD" in fe_name:
        col_a = f"fe_{fe_name}_M{timeframe}_CLOSE_W{w1}_cndl_M{timeframe}"
        col_b = f"fe_{fe_name}_M{timeframe}_CLOSE_W{w2}_cndl_M{timeframe}"
    elif "ATR" in fe_name:
        col_a = f"fe_{fe_name}_W{w1}_M{timeframe}"
        col_b = f"fe_{fe_name}_W{w2}_M{timeframe}"
    else:
        col_a = f"fe_{fe_name}_M{timeframe}_CLOSE_W{w1}_cndl_M{timeframe}_norm"
        col_b = f"fe_{fe_name}_M{timeframe}_CLOSE_W{w2}_cndl_M{timeframe}_norm"

    if col_a not in df.columns or col_b not in df.columns:
        print(f"!!! {col_a} not in df.columns or {col_b} not in df.columns.")
        return df

    ratio_col_name = (
        f"{fe_prefix}_{fe_name}_M{timeframe}_CLOSE_W{w1}_W{w2}_cndl_M{timeframe}"
    )

    df = add_ratio_by_columns(df, col_a, col_b, ratio_col_name)

    return df


def add_all_ratio_by_config(
    df: pl.DataFrame,
    symbol: str,
    fe_name: str,
    ratio_config: Dict[str, Dict[str, Union[List[int], List[Tuple[int, int]]]]],
    fe_prefix: str = "fe_ratio",
) -> pl.DataFrame:
    """
    this function takes the ratio config and applies add_ratio
    ratio_config: a dictionary of dictionaries containing list of time frames and list of pairs of window sizes needed for ratio
    """

    base_cols = set(df.columns) - set(["_time"])
    for timeframe in ratio_config["timeframe"]:
        for w_set in ratio_config["window_size"]:
            print(f"The timeframe {timeframe} of window size {w_set} is being processed ...")
            df = add_ratio(
                df, symbol, fe_name, timeframe, w_set[0], w_set[1], fe_prefix
            )

    return df.drop(base_cols)


# ?? volatility
def cal_ATR_func(
    df: pl.DataFrame,
    w: int,
    time_frame: int,
    features: List[str],
    pip_size: float,
    prefix: str = "fe_ATR",
    normalize: bool = False,
) -> pl.DataFrame:
    """
   Calculates the Average True Range (ATR), a technical indicator that
   measures market volatility by decomposing the entire range of an asset's
   price for a period. ATR is particularly useful for volatility-based
   position sizing and stop-loss placement.

   The ATR captures volatility through the greatest of:
   1. Current high - current low
   2. |Current high - previous close|
   3. |Current low - previous close|

   Key aspects for machine learning:
   1. Direct measure of market volatility
   2. Independent of price direction
   3. Adapts to changing market conditions
   4. Self-normalizing through rolling average
   5. Valuable for position sizing and risk management

   Implementation details:
   - Calculates true range considering overnight gaps
   - Applies simple moving average for smoothing
   - Offers normalization by close price option
   - Returns values in pips for easier interpretation

   Args:
       df (pl.DataFrame): DataFrame with OHLC price data
       w (int): Window size for ATR calculation (typical: 14)
       time_frame (int): Time frame in minutes for the calculation
       features (List[str]): List containing ['CLOSE', 'HIGH', 'LOW']
       pip_size (float): Size of one pip for scaling
       prefix (str, optional): Prefix for output column names.
           Defaults to "fe_ATR"
       normalize (bool, optional): If True, normalizes ATR by close price.
           Defaults to False

   Returns:
       pl.DataFrame: DataFrame with added ATR column:
           If normalize=True:
               - {prefix}_W{w}_M{time_frame}_norm: ATR/close_price
           If normalize=False:
               - {prefix}_W{w}_M{time_frame}: ATR in pips

   Notes:
       - Requires previous period's data for true range calculation
       - First w periods will contain incomplete ATR values
       - High ATR indicates high volatility, low ATR indicates low volatility
       - More reliable in trending markets than in ranging markets
       - Not predictive of price direction, only volatility
       - Commonly used window sizes: 14 (standard), 10 (more responsive)
       - Functions best with complete OHLC data
       - ATR tends to be larger for higher-priced assets (when not normalized)
   """
    assert (
        len(features) == 3
    ), f"Only 3 feature should have been passed but {len(features)} received!"
    features = sorted(features)
    input_features = [
        f'M{time_frame}_CLOSE',
        f'M{time_frame}_HIGH',
        f'M{time_frame}_LOW'
    ]
    if features != input_features:
        print('Input features are wrong')
        return
    # features[0] == f'M{time_frame}_CLOSE'
    # features[1] == f'M{time_frame}_HIGH'
    # features[2] == f'M{time_frame}_LOW'

    df = df.sort("_time")

    df = df.with_columns([
        pl.max_horizontal(
            pl.col(features[1]) - pl.col(features[2]),
            (pl.col(features[1]) - pl.col(features[0]).shift(1)).abs(),
            (pl.col(features[2]) - pl.col(features[0]).shift(1)).abs()
        ).alias("true_range")
    ]).lazy()

    df = df.with_columns([
        pl.col("true_range")
        .rolling_mean(window_size=w)
        .alias("atr_raw")
    ]).lazy()

    if normalize:
        column_name = f"{prefix}_W{w}_M{time_frame}_norm"
        df = df.with_columns([
            (pl.col("atr_raw") / (pl.col(features[0]) * pip_size)).alias(column_name)
        ]).lazy()
    else:
        column_name = f"{prefix}_W{w}_M{time_frame}"
        df = df.with_columns([
            (pl.col("atr_raw") / pip_size).alias(column_name)
        ]).lazy()

    df = df.drop(["true_range", "atr_raw"] + input_features)

    return df.collect()



def cal_RSTD_func(
    df: pl.DataFrame,
    w: int,
    time_frame: int,
    features: List[str],
    pip_size: float,
    prefix: str = "fe_RSTD",
    normalize: bool = False,
) -> pl.DataFrame:
    """
    this function calculates Standard Deviation of Return.
    inputs:
    df: dataframe containing the raw feature
    w: window size
    time_frame: time_frame for calculations
    feature: raw feature on which the RSI is calculated
    pip size: pip size of the pair
    prefix: prefix of feature name
    normalize: if True the function returns pipsize difference between EMA and last close price.

    """
    assert (
        len(features) == 1
    ), f"Only 1 feature should have been passed but {len(features)} received!"
    feature = features[0]

    df = df.sort("_time")
    if normalize:
        df = df.with_columns(
            (
                (
                    (
                        (
                            pl.col(feature).log() - pl.col(feature).shift(1).log()
                        ).rolling_std(window_size=w)
                    )
                    / pl.col(feature)
                )
                / pip_size
            ).alias(f"{prefix}_{feature}_W{w}_cndl_M{time_frame}")
        ).lazy()

    else:
        df = df.with_columns(
            (
                (
                    (
                        pl.col(feature).log() - pl.col(feature).shift(1).log()
                    ).rolling_std(window_size=w)
                )
                / pip_size
            ).alias(f"{prefix}_{feature}_W{w}_cndl_M{time_frame}")
        ).lazy()
    df = df.collect()
    df = df.drop([f"{feature}"])

    return df


def history_indicator_calculator(feature_config, logger=default_logger):
    """

    """

    logger.info("- " * 25)
    logger.info("--> start history_indicator_calculator fumc:")

    try:

        base_candle_folder_path = f"{root_path}/data/realtime_candle/"

        modes = {
            "fe_RSI": {"func": cal_RSI_base_func},
            "fe_EMA": {"func": cal_EMA_base_func},
            "fe_SMA": {"func": cal_SMA_base_func},
            "fe_ATR": {"func": cal_ATR_func},
            "fe_RSTD": {"func": cal_RSTD_func},
            "fe_cndl_shape_n_cntxt": {"func": cal_cndl_shape_n_cntxt_func},
        }

        for symbol in list(feature_config.keys()):
            logger.info("* " * 25)
            symbol_ratio_dfs = []


            for fe_prefix, func in modes.items():
                if fe_prefix not in list(feature_config[symbol].keys()):
                    continue
                logger.info("-" * 50)
                logger.info(f"--> symbol:{symbol} | fe_prefix:{fe_prefix}")

                features_folder_path = f"{root_path}/data/features/{fe_prefix}/"
                Path(features_folder_path).mkdir(parents=True, exist_ok=True)

                base_cols = feature_config[symbol][fe_prefix]["base_columns"]
                opts = {
                    "symbol": symbol,
                    "candle_timeframe": feature_config[symbol][fe_prefix]["timeframe"],
                    "window_size": feature_config[symbol][fe_prefix]["window_size"],
                    "features_folder_path": features_folder_path,
                }

                base_features = [
                    f"M{tf}_{col}"
                    for col in base_cols
                    for tf in opts["candle_timeframe"]
                ]
                opts["base_feature"] = base_features
                needed_columns = ["_time", "symbol", "minutesPassed"] + base_features
                file_name = base_candle_folder_path + f"{symbol}_realtime_candle.parquet"

                df = pl.read_parquet(file_name, columns=needed_columns)

                df = df.sort("_time").drop("symbol")

                add_candle_base_indicators_polars(
                    df_base=df,
                    prefix=fe_prefix,
                    base_func=func["func"],
                    opts=opts,
                )

                # ? merge
                df = df[["_time"]]
                pathes = glob.glob(
                    f"{features_folder_path}/unmerged/{fe_prefix}_**_{symbol}_*.parquet"
                )

                for df_path in pathes:
                    df_loaded = pl.read_parquet(df_path)
                    df = df.join(df_loaded, on="_time", how="left", coalesce=True)

                max_candle_timeframe = max(opts["candle_timeframe"])
                max_window_size = max(opts["window_size"])
                drop_rows = (max_window_size + 1) * (max_candle_timeframe / 5) - 1

                logger.info(
                    f"--> max_candle_timeframe:{max_candle_timeframe} | max_window_size:{max_window_size}| drop_rows:{drop_rows}"
                )

                df = df.with_row_index()
                df = (
                    df.filter(pl.col("index") >= drop_rows)
                    .fill_null(strategy="forward")
                    .drop(*["index"])
                )

                df = df.drop_nulls()
                df = df.with_columns(pl.lit(symbol).alias("symbol"))

                file_name = features_folder_path + f"/{fe_prefix}_{symbol}.parquet"
                df.write_parquet(file_name)

                logger.info(f"--> {fe_prefix}_{symbol} done.")

                ## add ratio: ------------------------------------------------------------------
                ratio_prefix = "fe_ratio"

                if ratio_prefix not in list(feature_config[symbol].keys()):
                    continue

                fe_prefix_replaced = fe_prefix.replace("fe_", "")

                if fe_prefix_replaced in list(
                    feature_config[symbol][ratio_prefix].keys()
                ):
                    print(f"The feature {fe_prefix_replaced} ratio is being processed ...")

                    ratio_config = feature_config[symbol][ratio_prefix][
                        fe_prefix_replaced
                    ]
                    features_folder_path = f"{root_path}/data/features/{ratio_prefix}/"
                    Path(features_folder_path).mkdir(parents=True, exist_ok=True)

                    symbol_ratio_dfs.append(
                        add_all_ratio_by_config(
                            df,
                            symbol,
                            fe_name=fe_prefix_replaced,
                            ratio_config=ratio_config,
                            fe_prefix="fe_ratio",
                        )
                    )

            # ? merge ratio for one symbol:
            if len(symbol_ratio_dfs) == 0:
                print(f"!!! no ratio feature for {symbol}.")
                continue
            elif len(symbol_ratio_dfs) == 1:
                df = symbol_ratio_dfs[0]
            else:
                df = symbol_ratio_dfs[0]
                for i in range(1, len(symbol_ratio_dfs)):
                    df = df.join(symbol_ratio_dfs[i], on="_time")

            df = df.with_columns(pl.lit(symbol).alias("symbol"))
            file_name = features_folder_path + f"/{ratio_prefix}_{symbol}.parquet"
            df.write_parquet(file_name)
            logger.info(f"--> {ratio_prefix}_{symbol} saved.")

        logger.info("--> history_indicator_calculator run successfully.")
    except Exception as e:
        logger.exception("--> history_indicator_calculator error.")
        logger.exception(f"--> error: {e}")
        raise ValueError("!!!")


if __name__ == "__main__":
    from configs.feature_configs_general import generate_general_config
    config_general = generate_general_config()
    history_indicator_calculator(config_general)
    print(f"--> history_indicator_calculator DONE.")
