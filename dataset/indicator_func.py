import polars as pl
import numpy as np
from pathlib import Path
from dataset.configs.history_data_crawlers_config import symbols_dict
import os, glob
from typing import Callable, Dict, List, Tuple, Union
from dataset.configs.history_data_crawlers_config import root_path
from dataset.configs.feature_configs_general import fe_leg_config
import re
from dataset.logging_tools import default_logger
from arch.unitroot import ADF
from numba import njit
from functools import reduce


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
    # and calculate number of digits in close price (excluding decimal places)
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
        .alias(f"{prefix}_lower_price_M{time_frame}"),
        (
            (
                pl.col(features[0]).log10() + 0.5 + 1e-9
            )
            .round()
        )
        .cast(pl.Int64)
        .alias(
            f"{prefix}_close_digits_M{time_frame}"
        )
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

    alpha = 2.0 / (w + 1)
    calcs = []

    # Create rolling statistics for each context feature
    for idx, feature in enumerate(context_features):
        # Calculate rolling stats
        if idx == 0:
            calcs.extend([
                pl.col(feature)
                .rolling_mean(window_size=w)
                .alias(f"{feature}_rolling_mean"),
                pl.col(feature)
                .rolling_median(window_size=w)
                .alias(f"{feature}_rolling_median"),
                (pl.col(feature).rolling_max(window_size=w) -
                pl.col(feature).rolling_min(window_size=w))
                .alias(f"{feature}_rolling_range"),
                (pl.col(feature).rolling_quantile(quantile=0.75, window_size=w) -
                pl.col(feature).rolling_quantile(quantile=0.25, window_size=w))
                .alias(f"{feature}_rolling_iqr"),
                pl.col(feature)
                .ewm_mean(alpha=alpha)
                .alias(f"{feature}_ema")
            ])
        else:
            calcs.extend([
                pl.col(feature)
                .rolling_mean(window_size=w)
                .alias(f"{feature}_rolling_mean"),
                pl.col(feature)
                .rolling_median(window_size=w)
                .alias(f"{feature}_rolling_median"),
                pl.col(feature)
                .rolling_std(window_size=w)
                .alias(f"{feature}_rolling_std"),
                (pl.col(feature).rolling_max(window_size=w) -
                pl.col(feature).rolling_min(window_size=w))
                .alias(f"{feature}_rolling_range"),
                (pl.col(feature).rolling_quantile(quantile=0.75, window_size=w) -
                pl.col(feature).rolling_quantile(quantile=0.25, window_size=w))
                .alias(f"{feature}_rolling_iqr"),
                pl.col(feature)
                .ewm_mean(alpha=alpha)
                .alias(f"{feature}_ema")
            ])

    # Calculate rounded price distances for different decimal places
    for i in range(2, 4):  # For n-2, n-3
        calcs.extend([
            # Calculate decimal places dynamically based on number of digits
            (
                (
                    pl.col(features[0]) /
                    (
                        10.0 ** (
                            pl.col(f"{prefix}_close_digits_M{time_frame}") - i
                        )
                    ) + (0.5 + 1e-9)
                )
                .round()
                * (
                    10.0 ** (
                        pl.col(f"{prefix}_close_digits_M{time_frame}") - i
                    )
                ) - pl.col(features[0])
            )
            .alias(f"{prefix}_dist_up_round_{i}_M{time_frame}"),
            (
                pl.col(features[0]) -
                (
                    pl.col(features[0]) /
                    (
                        10.0 ** (
                            pl.col(f"{prefix}_close_digits_M{time_frame}") - i
                        )
                    ) - (0.5 + 1e-9)
                )
                .round()
                * (
                    10.0 ** (
                        pl.col(f"{prefix}_close_digits_M{time_frame}") - i
                    )
                )
            )
            .alias(f"{prefix}_dist_down_round_{i}_M{time_frame}")
        ])

    df = df.with_columns(calcs).lazy()

    # Drop unnecessary columns
    cols_to_drop = features + context_features
    cols_to_drop.extend([
        f"{prefix}_higher_price_M{time_frame}",
        f"{prefix}_lower_price_M{time_frame}",
        f"{prefix}_lower_tercile_M{time_frame}",
        f"{prefix}_upper_tercile_M{time_frame}",
        f"{prefix}_candle_length_M{time_frame}",
        f"{prefix}_close_digits_M{time_frame}"
    ])
    df = df.collect()
    df = df.drop(cols_to_drop)

    return df


def cal_leg_base_func(
    df: pl.DataFrame,
    w: int,
    time_frame: int,
    features: List[str],
    pip_size: float,
    exponents: dict[str, Tuple[int, int]],
    prefix: str = "fe_leg",
    percentage_feature: bool = True,
    percentage: float = 0.001,
) -> pl.DataFrame:
    """
    This function calculates the distance of the current candle's close price from 
    the pivots (highs and lows) of the legs of price movements based upon 
    the zigzag indicator's calculations
    """
    assert (
        len(features) == 1
    ), f"Only 1 feature should have been passed but {len(features)} received!"
    # features[0] == f'M{time_frame}_CLOSE

    df = df.sort("_time")

    # Convert to numpy for easier calculations
    close_prices = df[features[0]].to_numpy()

    # Not a window size, but a return's threshold
    th = w

    # Set percentages and thresholds for timeframes other than 5M
    if time_frame == 15:
        percentage *= exponents.get('15')[0]
        th *= exponents.get('15')[1]
    elif time_frame == 60:
        percentage *= exponents.get('60')[0]
        th *= exponents.get('60')[1]

    # Initialize arrays for pivot points
    # pivot_indicators = np.zeros(len(df))
    pivot_points = np.zeros(len(df))
    pivot_prices = np.zeros(len(df))

    # First pass: identify potential pivot points
    trend = None
    leg_ended = True
    last_pivot_idx = 0
    movement_pivot_idx = 0
    last_pivot_idx_high = 0
    last_pivot_idx_low = 0
    # indicator = 0
    ups = 0
    downs = 0

    current_high = float('-inf')
    reserved_high = float('-inf')
    current_low = float('inf')
    reserved_low = float('inf')
    current_high_idx = 0
    reserved_high_idx = 0
    current_low_idx = 0
    reserved_low_idx = 0
    reserved_status = False
    leg_counter = 0
    pivot_status = 0

    if percentage_feature:
        bullish_high_pivot_distances = np.full(len(df), 150)
        bullish_low_pivot_distances = np.full(len(df), 150)
        bearish_high_pivot_distances = np.full(len(df), 150)
        bearish_low_pivot_distances = np.full(len(df), 150)
        suffix = "_pct"
    else:
        bullish_high_pivot_distances = np.full(len(df), 50/pip_size)
        bullish_low_pivot_distances = np.full(len(df), 50/pip_size)
        bearish_high_pivot_distances = np.full(len(df), 50/pip_size)
        bearish_low_pivot_distances = np.full(len(df), 50/pip_size)
        suffix = "_pips_norm"

    for i in range(1, len(df)):
        price_move = abs(close_prices[i] - close_prices[last_pivot_idx]) / close_prices[last_pivot_idx]

        if price_move >= percentage:
            # indicator += 10 # 10
            returns = np.zeros(i - last_pivot_idx)

            for j in range(last_pivot_idx, i):
                returns[j - last_pivot_idx] = close_prices[j+1] - close_prices[j]

            mean_return_norm = np.mean(returns)*1000 / (pip_size*close_prices[last_pivot_idx])

            if close_prices[i] > close_prices[last_pivot_idx]: # Upward movements

                if mean_return_norm > th:
                    # indicator += 5 # 15
                    if trend != 'up':
                        # indicator += 1 # 16
                        if not leg_ended:
                            pivot_points[last_pivot_idx_low] = -1 # Mark as low pivot (the end of the previous leg)
                            pivot_prices[last_pivot_idx_low] = close_prices[last_pivot_idx_low]
                        pivot_points[last_pivot_idx] = -1 # Mark as low pivot (the beginning of the current leg)
                        pivot_prices[last_pivot_idx] = close_prices[last_pivot_idx]
                        leg_ended = False
                        trend = 'up'
                        downs = 0
                    else:
                        # indicator += 2 # 17
                        if ups >= 2:
                            # indicator += 1 # 18
                            pivot_points[last_pivot_idx] = -1 # Mark as low pivot (the beginning of the current leg)
                            pivot_prices[last_pivot_idx] = close_prices[last_pivot_idx]
                            leg_ended = False
                            ups = 0

                    pivot_points[i] = 2 # Mark as middle high pivot (the current end of the current leg)
                    pivot_prices[i] = close_prices[i]

                    movement_pivot_idx = i
                    last_pivot_idx = i
                    last_pivot_idx_high = i
                else:
                    # indicator += 1 # 11
                    if trend == 'down':
                        # indicator += 1 # 12
                        if not leg_ended:
                            pivot_points[last_pivot_idx_low] = -1 # Mark as low pivot (the end of the previous leg)
                            pivot_prices[last_pivot_idx_low] = close_prices[last_pivot_idx_low]
                            leg_ended = True
                        trend = 'up'
                        downs = 0
                    elif trend == 'up':
                        # indicator += 2 # 13
                        ups += 1

                    movement_pivot_idx = i
                    last_pivot_idx = i
                    if ups == 2:
                        # indicator += 1 # 14
                        if not leg_ended:
                            pivot_points[last_pivot_idx_high] = 1 # Mark as high pivot (the end of the previous leg)
                            pivot_prices[last_pivot_idx_high] = close_prices[last_pivot_idx_high]
                            leg_ended = True

            elif close_prices[i] < close_prices[last_pivot_idx]: # Downward movements
                # indicator += 10 # 20

                if mean_return_norm < -th:
                    # indicator += 5 # 25
                    if trend != 'down':
                        # indicator += 1 # 26
                        if not leg_ended:
                            pivot_points[last_pivot_idx_high] = 1 # Mark as high pivot (the end of the previous leg)
                            pivot_prices[last_pivot_idx_high] = close_prices[last_pivot_idx_high]
                        pivot_points[last_pivot_idx] = 1 # Mark as high pivot (the beginning of the current leg)
                        pivot_prices[last_pivot_idx] = close_prices[last_pivot_idx]
                        leg_ended = False
                        trend = 'down'
                        ups = 0
                    else:
                        # indicator += 2 # 27
                        if downs >= 2:
                            # indicator += 1 # 28
                            pivot_points[last_pivot_idx] = 1 # Mark as high pivot (the beginning of the current leg)
                            pivot_prices[last_pivot_idx] = close_prices[last_pivot_idx]
                            leg_ended = False
                            downs = 0

                    pivot_points[i] = -2 # Mark as middle low pivot (the current end of the current leg)
                    pivot_prices[i] = close_prices[i]

                    movement_pivot_idx = i
                    last_pivot_idx = i
                    last_pivot_idx_low = i
                else:
                    # indicator += 1 # 21
                    if trend == 'up':
                        # indicator += 1 # 22
                        if not leg_ended:
                            pivot_points[last_pivot_idx_high] = 1 # Mark as high pivot (the end of the previous leg)
                            pivot_prices[last_pivot_idx_high] = close_prices[last_pivot_idx_high]
                            leg_ended = True
                        trend = 'down'
                        ups = 0
                    elif trend == 'down':
                        # indicator += 2 # 23
                        downs += 1

                    movement_pivot_idx = i
                    last_pivot_idx = i
                    if downs == 2:
                        # indicator += 1 # 24
                        if not leg_ended:
                            pivot_points[last_pivot_idx_low] = -1 # Mark as low pivot (the end of the previous leg)
                            pivot_prices[last_pivot_idx_low] = close_prices[last_pivot_idx_low]
                            leg_ended = True

            # pivot_indicators[i] = indicator
            # indicator = 0
            continue
        else:
            if trend == 'up':
                if close_prices[i] > close_prices[last_pivot_idx]:
                    # indicator -= 1 # -1
                    last_pivot_idx = i
                if close_prices[i] < close_prices[movement_pivot_idx]:
                    # indicator -= 2 # -2
                    movement_pivot_idx = i
            elif trend == 'down':
                if close_prices[i] < close_prices[last_pivot_idx]:
                    # indicator -= 3 # -3
                    last_pivot_idx = i
                if close_prices[i] > close_prices[movement_pivot_idx]:
                    # indicator -= 4 # -4
                    movement_pivot_idx = i

        # Calculating slow opposite-side movements effects
        mvmnt_price_move = abs(close_prices[i] - close_prices[movement_pivot_idx]) / close_prices[movement_pivot_idx]

        if mvmnt_price_move > percentage:
            # indicator -= 10 # -10, -11, -12, -13, -14
            returns = np.zeros(i - movement_pivot_idx)

            for j in range(movement_pivot_idx, i):
                returns[j - movement_pivot_idx] = close_prices[j+1] - close_prices[j]

            mean_return_norm = np.mean(returns)*1000 / (pip_size*close_prices[movement_pivot_idx])

            if trend == 'up': # Upward movements
                if mean_return_norm > th:
                    # indicator -= 10 # -20, -21, -22
                    if ups >= 2:
                        # indicator -= 5 # -25, -26, -27
                        pivot_points[movement_pivot_idx] = -1 # Mark as low pivot (the beginning of the current leg)
                        pivot_prices[movement_pivot_idx] = close_prices[movement_pivot_idx]
                        leg_ended = False
                        ups = 0

                    pivot_points[i] = 2 # Mark as middle high pivot (the current end of the current leg)
                    pivot_prices[i] = close_prices[i]

                    movement_pivot_idx = i
                    last_pivot_idx = i
                    last_pivot_idx_high = i
                else:
                    # indicator -= 20 # -30, -31, -32
                    ups += 1
                    movement_pivot_idx = i
                    last_pivot_idx = i
                    if ups == 2:
                        # indicator -= 5 # -35, -36, -37
                        if not leg_ended:
                            pivot_points[last_pivot_idx_high] = 1 # Mark as high pivot (the end of the previous leg)
                            pivot_prices[last_pivot_idx_high] = close_prices[last_pivot_idx_high]
                            leg_ended = True

            elif trend == 'down': # Downward movements
                if mean_return_norm < -th:
                    # indicator -= 10 # -20, -23, -24
                    if downs >= 2:
                        # indicator -= 5 # -25, -28, -29
                        pivot_points[movement_pivot_idx] = 1 # Mark as high pivot (the beginning of the current leg)
                        pivot_prices[movement_pivot_idx] = close_prices[movement_pivot_idx]
                        leg_ended = False
                        downs = 0

                    pivot_points[i] = -2 # Mark as middle low pivot (the current end of the current leg)
                    pivot_prices[i] = close_prices[i]

                    movement_pivot_idx = i
                    last_pivot_idx = i
                    last_pivot_idx_low = i
                else:
                    # indicator -= 20 # -30, -33, -34
                    downs += 1
                    movement_pivot_idx = i
                    last_pivot_idx = i
                    if downs == 2:
                        # indicator -= 5 # -35, -38, -39
                        if not leg_ended:
                            pivot_points[last_pivot_idx_low] = -1 # Mark as low pivot (the end of the previous leg)
                            pivot_prices[last_pivot_idx_low] = close_prices[last_pivot_idx_low]
                            leg_ended = True

        close = close_prices[i]
        # Calculate distances
        if close > current_low and close < current_high:
            total_range = current_high - current_low

            if percentage_feature:
                if current_high_idx > current_low_idx:
                    bullish_high_pivot_distances[i] = ((current_high - close) / total_range) * 100
                    bullish_low_pivot_distances[i] = ((close - current_low) / total_range) * 100
                else:
                    bearish_high_pivot_distances[i] = ((current_high - close) / total_range) * 100
                    bearish_low_pivot_distances[i] = ((close - current_low) / total_range) * 100
            else:
                if current_high_idx > current_low_idx:
                    bullish_high_pivot_distances[i] = (current_high - close)*1000 / (pip_size*close)
                    bullish_low_pivot_distances[i] = (close - current_low)*1000 / (pip_size*close)
                else:
                    bearish_high_pivot_distances[i] = (current_high - close)*1000 / (pip_size*close)
                    bearish_low_pivot_distances[i] = (close - current_low)*1000 / (pip_size*close)

        if pivot_points[i] == 1:  # High pivot
            if leg_counter == 1:
                if pivot_status == -1:
                    current_high = close
                    current_high_idx = i
                    pivot_status = 1
                    leg_counter += 1
                    if reserved_status:
                        current_low = reserved_low
                        current_low_idx = reserved_low_idx
                        reserved_status = False
                else:
                    raise Exception(f"There has come a duplicate high in the {i}th index at {df['_time'][i]} with th = {th}")
            elif leg_counter > 1:
                if pivot_status == -1:
                    current_high = close
                    current_high_idx = i
                    pivot_status = 1
                    leg_counter += 1
                else:
                    reserved_high = close
                    reserved_high_idx = i
                    reserved_status = True
                    leg_counter = 1
            else:
                current_high = close
                current_high_idx = i
                pivot_status = 1
                leg_counter += 1
        elif pivot_points[i] == -1:  # Low pivot
            if leg_counter == 1:
                if pivot_status == 1:
                    current_low = close
                    current_low_idx = i
                    pivot_status = -1
                    leg_counter += 1
                    if reserved_status:
                        current_high = reserved_high
                        current_high_idx = reserved_high_idx
                        reserved_status = False
                else:
                    raise Exception(f"There has come a duplicate low in the {i}th index at {df['_time'][i]} with th = {th}")
            elif leg_counter > 1:
                if pivot_status == 1:
                    current_low = close
                    current_low_idx = i
                    pivot_status = -1
                    leg_counter += 1
                else:
                    reserved_low = close
                    reserved_low_idx = i
                    reserved_status = True
                    leg_counter = 1
            else:
                current_low = close
                current_low_idx = i
                pivot_status = -1
                leg_counter += 1
        elif pivot_points[i] == 2:  # Middle high pivot
            current_high = close
        elif pivot_points[i] == -2:  # Middle low pivot
            current_low = close

        # pivot_indicators[i] = indicator
        # indicator = 0

    # current_high = float('-inf')
    # reserved_high = float('-inf')
    # current_low = float('inf')
    # reserved_low = float('inf')
    # current_high_idx = 0
    # reserved_high_idx = 0
    # current_low_idx = 0
    # reserved_low_idx = 0
    # reserved_status = False
    # leg_counter = 0
    # pivot_status = 0

    # if percentage_feature:
    #     bullish_high_pivot_distances = np.full(len(df), 150)
    #     bullish_low_pivot_distances = np.full(len(df), 150)
    #     bearish_high_pivot_distances = np.full(len(df), 150)
    #     bearish_low_pivot_distances = np.full(len(df), 150)
    #     suffix = "_pct"

    #     for i in range(len(df)):
    #         close = close_prices[i]
    #         # Calculate distances
    #         if close > current_low and close < current_high:
    #             total_range = current_high - current_low

    #             if current_high_idx > current_low_idx:
    #                 bullish_high_pivot_distances[i] = ((current_high - close) / total_range) * 100
    #                 bullish_low_pivot_distances[i] = ((close - current_low) / total_range) * 100
    #             else:
    #                 bearish_high_pivot_distances[i] = ((current_high - close) / total_range) * 100
    #                 bearish_low_pivot_distances[i] = ((close - current_low) / total_range) * 100

    #         if pivot_points[i] == 1:  # High pivot
    #             if leg_counter == 1:
    #                 if pivot_status == -1:
    #                     current_high = close
    #                     current_high_idx = i
    #                     pivot_status = 1
    #                     leg_counter += 1
    #                     if reserved_status:
    #                         current_low = reserved_low
    #                         current_low_idx = reserved_low_idx
    #                         reserved_status = False
    #                 else:
    #                     raise Exception(f"There has come a duplicate high in the {i}th index at {df['_time'][i]} with th = {th}")
    #             elif leg_counter > 1:
    #                 if pivot_status == -1:
    #                     current_high = close
    #                     current_high_idx = i
    #                     pivot_status = 1
    #                     leg_counter += 1
    #                 else:
    #                     reserved_high = close
    #                     reserved_high_idx = i
    #                     reserved_status = True
    #                     leg_counter = 1
    #             else:
    #                 current_high = close
    #                 current_high_idx = i
    #                 pivot_status = 1
    #                 leg_counter += 1
    #         elif pivot_points[i] == -1:  # Low pivot
    #             if leg_counter == 1:
    #                 if pivot_status == 1:
    #                     current_low = close
    #                     current_low_idx = i
    #                     pivot_status = -1
    #                     leg_counter += 1
    #                     if reserved_status:
    #                         current_high = reserved_high
    #                         current_high_idx = reserved_high_idx
    #                         reserved_status = False
    #                 else:
    #                     raise Exception(f"There has come a duplicate low in the {i}th index at {df['_time'][i]} with th = {th}")
    #             elif leg_counter > 1:
    #                 if pivot_status == 1:
    #                     current_low = close
    #                     current_low_idx = i
    #                     pivot_status = -1
    #                     leg_counter += 1
    #                 else:
    #                     reserved_low = close
    #                     reserved_low_idx = i
    #                     reserved_status = True
    #                     leg_counter = 1
    #             else:
    #                 current_low = close
    #                 current_low_idx = i
    #                 pivot_status = -1
    #                 leg_counter += 1
    #         elif pivot_points[i] == 2:  # Middle high pivot
    #             current_high = close
    #         elif pivot_points[i] == -2:  # Middle low pivot
    #             current_low = close

    # else:
    #     bullish_high_pivot_distances = np.full(len(df), 50/pip_size)
    #     bullish_low_pivot_distances = np.full(len(df), 50/pip_size)
    #     bearish_high_pivot_distances = np.full(len(df), 50/pip_size)
    #     bearish_low_pivot_distances = np.full(len(df), 50/pip_size)
    #     suffix = "_pips_norm"

    #     for i in range(len(df)):
    #         close = close_prices[i]
    #         # Calculate distances
    #         if close > current_low and close < current_high:
    #             total_range = current_high - current_low

    #             if current_high_idx > current_low_idx:
    #                 bullish_high_pivot_distances[i] = (current_high - close)*1000 / (pip_size*close)
    #                 bullish_low_pivot_distances[i] = (close - current_low)*1000 / (pip_size*close)
    #             else:
    #                 bearish_high_pivot_distances[i] = (current_high - close)*1000 / (pip_size*close)
    #                 bearish_low_pivot_distances[i] = (close - current_low)*1000 / (pip_size*close)

    #         if pivot_points[i] == 1:  # High pivot
    #             if leg_counter == 1:
    #                 if pivot_status == -1:
    #                     current_high = close
    #                     current_high_idx = i
    #                     pivot_status = 1
    #                     leg_counter += 1
    #                     if reserved_status:
    #                         current_low = reserved_low
    #                         current_low_idx = reserved_low_idx
    #                         reserved_status = False
    #                 else:
    #                     raise Exception(f"There has come a duplicate high in the {i}th index at {df['_time'][i]} with th = {th}")
    #             elif leg_counter > 1:
    #                 if pivot_status == -1:
    #                     current_high = close
    #                     current_high_idx = i
    #                     pivot_status = 1
    #                     leg_counter += 1
    #                 else:
    #                     reserved_high = close
    #                     reserved_high_idx = i
    #                     reserved_status = True
    #                     leg_counter = 1
    #             else:
    #                 current_high = close
    #                 current_high_idx = i
    #                 pivot_status = 1
    #                 leg_counter += 1
    #         elif pivot_points[i] == -1:  # Low pivot
    #             if leg_counter == 1:
    #                 if pivot_status == 1:
    #                     current_low = close
    #                     current_low_idx = i
    #                     pivot_status = -1
    #                     leg_counter += 1
    #                     if reserved_status:
    #                         current_high = reserved_high
    #                         current_high_idx = reserved_high_idx
    #                         reserved_status = False
    #                 else:
    #                     raise Exception(f"There has come a duplicate low in the {i}th index at {df['_time'][i]} with th = {th}")
    #             elif leg_counter > 1:
    #                 if pivot_status == 1:
    #                     current_low = close
    #                     current_low_idx = i
    #                     pivot_status = -1
    #                     leg_counter += 1
    #                 else:
    #                     reserved_low = close
    #                     reserved_low_idx = i
    #                     reserved_status = True
    #                     leg_counter = 1
    #             else:
    #                 current_low = close
    #                 current_low_idx = i
    #                 pivot_status = -1
    #                 leg_counter += 1
    #         elif pivot_points[i] == 2:  # Middle high pivot
    #             current_high = close
    #         elif pivot_points[i] == -2:  # Middle low pivot
    #             current_low = close

    # Create leg columns in DataFrame
    df = df.with_columns([
        # pl.Series(name=f"{prefix}_pvt_indicators_M{time_frame}_th_{th}{suffix}",values=pivot_indicators),
        # pl.Series(name=f"{prefix}_pvt_points_M{time_frame}_th_{th}{suffix}",values=pivot_points),
        pl.Series(name=f"{prefix}_blsh_high_dist_M{time_frame}_th_{th}{suffix}", values=bullish_high_pivot_distances),
        pl.Series(name=f"{prefix}_blsh_low_dist_M{time_frame}_th_{th}{suffix}", values=bullish_low_pivot_distances),
        pl.Series(name=f"{prefix}_brsh_high_dist_M{time_frame}_th_{th}{suffix}", values=bearish_high_pivot_distances),
        pl.Series(name=f"{prefix}_brsh_low_dist_M{time_frame}_th_{th}{suffix}", values=bearish_low_pivot_distances)
    ]).lazy()

    # Dropping price column (Comment it if you want to plot legs in Colab)
    df = df.drop(features[0])

    return df.collect()


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
    percentage_feature: true for percentage features like price-percentage 
        are diff features by nature
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
            f"{feature}_RS_{w}"
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
    opts: a dictionary of "symbol", "base_feature", "candle_timeframe",
        "window_size" and "features_folder_path"
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

    if prefix == 'fe_GMA':
        devs = opts['feature_config']['devs']
    elif prefix == 'fe_FFD':
        n_splits = opts['feature_config']['n_splits']
    elif prefix == 'fe_OL':
        w_sma = opts['feature_config']['window_size_SMA']
    elif prefix == 'fe_supertrend':
        multipliers = opts['feature_config']['multipliers']

    if prefix == "fe_leg":
        exponents = opts["exponents"]
        percentage = opts["percentage"]

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
                    exponents=exponents,
                    prefix=prefix,
                    percentage=percentage,
                )

                file_name = (
                    features_folder_path + f"/{prefix}_{w}_{symbol}_M{time_frame}.parquet"
                )

                df.write_parquet(file_name)
    else:
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

                if prefix == 'fe_GMA':
                    df = base_func(
                        df=df,
                        w=w,
                        time_frame=time_frame,
                        features=list(set(features) - set(other_tf_features)),
                        pip_size=pip_size,
                        prefix=prefix,
                        devs=devs,
                    )
                elif prefix == 'fe_FFD':
                    df = base_func(
                        df=df,
                        time_frame=time_frame,
                        features=list(set(features) - set(other_tf_features)),
                        prefix=prefix,
                        n_splits=n_splits,
                    )
                elif prefix == 'fe_OL':
                    df = base_func(
                        df=df,
                        w=w,
                        w_sma=w_sma,
                        time_frame=time_frame,
                        features=list(set(features) - set(other_tf_features)),
                        pip_size=pip_size,
                        prefix=prefix,
                    )
                elif prefix == 'fe_supertrend':
                    df = base_func(
                        df=df,
                        w=w,
                        time_frame=time_frame,
                        features=list(set(features) - set(other_tf_features)),
                        multipliers=multipliers,
                        prefix=prefix,
                    )
                else:
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


# ??  ratio  -----------------------------------------------------
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
    ratio_config: a dictionary of dictionaries containing list of time frames
        and list of pairs of window sizes needed for ratio
    """

    base_cols = set(df.columns) - set(["_time"])
    for timeframe in ratio_config["timeframe"]:
        for w_set in ratio_config["window_size"]:
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


def cal_supertrend_func(
    df: pl.DataFrame,
    w: int,
    time_frame: int,
    features: List[str],
    multipliers: list[float],
    prefix: str = "fe_supertrend",
) -> pl.DataFrame:
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
            (pl.col(features[1]) - pl.col(features[2])).abs(),
            (pl.col(features[1]) - pl.col(features[0]).shift(1)).abs(),
            (pl.col(features[2]) - pl.col(features[0]).shift(1)).abs(),
        ).alias("true_range")
    ]).lazy()

    df = df.with_columns([
        pl.col("true_range").rolling_mean(window_size=w).alias("atr")
    ]).lazy()

    upper_band_column_names = []
    lower_band_column_names = []

    for idx, multiplier in enumerate(multipliers):
        upper_band_column_names.append(f"upper_band_mp{multiplier}")
        lower_band_column_names.append(f"lower_band_mp{multiplier}")
        # Calculate basic upper and lower bands
        df = df.with_columns([
            (
                (pl.col(features[1]) + pl.col(features[2])) / 2 + (multiplier * pl.col("atr"))
            ).alias(upper_band_column_names[idx]),
            (
                (pl.col(features[1]) + pl.col(features[2])) / 2 - (multiplier * pl.col("atr"))
            ).alias(lower_band_column_names[idx]),
        ]).lazy()

        column_name = f"{prefix}_trend_direction_tf{time_frame}_w{w}_mp{multiplier}"

        # Initialize Supertrend columns
        df = df.with_columns([
            pl.lit(0).alias(column_name)
        ]).lazy()

        # Iterate over rows to calculate Supertrend
        eager_df = df.collect()
        closes = eager_df[features[0]].to_numpy()
        lower_bands = eager_df[lower_band_column_names[idx]].to_numpy()
        upper_bands = eager_df[upper_band_column_names[idx]].to_numpy()
        supertrend = np.zeros(len(eager_df))
        trend_direction = np.zeros(len(eager_df))
        trend_changed = False

        for i in range(len(eager_df)):
            if i == 0:
                # First row initialization
                supertrend[i] = upper_bands[i]
                trend_direction[i] = 1
            else:
                if closes[i] > supertrend[i-1]:
                    if trend_direction[i-1] == 0:
                        trend_changed = True
                    trend_direction[i] = 1
                elif closes[i] < supertrend[i-1]:
                    if trend_direction[i-1] == 1:
                        trend_changed = True
                    trend_direction[i] = 0
                else:
                    trend_direction[i] = trend_direction[i-1]

                if trend_changed:
                    if trend_direction[i-1] == 1:
                        supertrend[i] = lower_bands[i]
                    else:
                        supertrend[i] = upper_bands[i]
                    trend_changed = False
                else:
                    if trend_direction[i-1] == 1:
                        supertrend[i] = max(lower_bands[i], supertrend[i-1])
                    else:
                        supertrend[i] = min(upper_bands[i], supertrend[i-1])

        df = df.with_columns([
            pl.Series(name=column_name, values=trend_direction)
        ]).lazy()

    df = df.drop(["true_range", "atr"] + input_features + upper_band_column_names + lower_band_column_names)

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


def cal_FFD_func(
    df: pl.DataFrame,
    features: List[str],
    time_frame: int,
    n_splits: List[int],
    Auto_optimaze_d : bool|List[int] = True,
    prefix: str = "fe_FFD",
) -> pl.DataFrame:

    df = df.sort("_time")
    col_drop = list(set(list(df.columns)) - set(['_time']))

    def base_FFD(series, d, thres=1e-5):
        def getWeights(d, size, thres=1e-5):
            w = [1.0]
            for k in range(1, size):
                w_ = -w[-1] / k * (d - k + 1)
                w.append(w_)
            return np.array(w)[np.abs(w) > thres]

        w = getWeights(d, size=10000, thres=thres)

        result = np.convolve(series.to_numpy(), w, mode="valid")

        final_result = np.full_like(series.to_numpy(), np.nan, dtype=np.float64)

        final_result[len(w) - 1:] = result

        return pl.Series(final_result)

    def adf_test(series):

        result = ADF(series.drop_nulls().drop_nans().to_numpy()).pvalue

        return result

    def split_dataframe(df, n_splits):
        indices = np.linspace(0, df.height, n_splits + 1, dtype=int)
        splits = [df[indices[i]:indices[i + 1]] for i in range(n_splits)]

        return splits + [df]

    def Optimaze_d(list_df, base_feature, min_d=0, max_d=1, step=0.01):
        list_d = [min_d]
        for df in list_df:
            for d in np.arange(list_d[-1], max_d + step, step):
                ser = base_FFD(df[base_feature], d, 1e-5)
                pval_adf = adf_test(ser)
                if pval_adf < 0.05:
                    list_d.append(d)
                    break

        return max(list_d)

    @njit
    def correlation(x, y):
        x_mean, y_mean = np.mean(x), np.mean(y)
        cov = np.mean((x - x_mean) * (y - y_mean))
        corr = cov / (np.std(x) * np.std(y))

        return corr

    if type(Auto_optimaze_d) == bool and Auto_optimaze_d:
        for ns in n_splits:
            list_df = split_dataframe(df, ns)
            fea = features[features.index(f"M{time_frame}_CLOSE")]

            best_d = Optimaze_d(list_df, fea)
            ser = base_FFD(df[fea], best_d)
            best_d = round(best_d, 3)

            df = df.with_columns(ser.alias(f"{prefix}-{fea}_{best_d}"))

            corr = correlation(
                df.filter(pl.col(f"{prefix}-{fea}_{best_d}").is_not_nan())[fea].to_numpy(),
                df.filter(pl.col(f"{prefix}-{fea}_{best_d}").is_not_nan())[f"{prefix}-{fea}_{best_d}"].to_numpy(),
            )
            print(f"{fea}_{ns} : best_d = {best_d} | corr : {corr}")

    else:
        fea = features[features.index(f"M{time_frame}_CLOSE")]
        for d in Auto_optimaze_d:
            ser = base_FFD(df[fea], d)
            df = df.with_columns(ser.alias(f"{prefix}-{fea}_{d}"))
            corr = correlation(
                df.filter(pl.col(f"{prefix}-{fea}_{d}").is_not_nan())[fea].to_numpy(),
                df.filter(pl.col(f"{prefix}-{fea}_{d}").is_not_nan())[f"{prefix}-{fea}_{d}"].to_numpy(),
            )
            print(f"{fea} : d-value = {d} | corr : {corr}")

            # for feature in list(set(features) - set([fea])):
            #     ser = base_FFD(df[feature], best_d)
            #     df = df.with_columns(ser.alias(f"{prefix}-{feature}_{best_d}"))

    df =df.drop(col_drop)
    df = df.filter(
        reduce(
            lambda acc, col: acc & pl.col(col).is_not_nan() if col != '_time' else acc,
            df.columns,
            pl.lit(True)
        )
    )

    return df


def cal_GMA_n_GBB_func(
    df: pl.DataFrame,
    w: int,
    time_frame: int,
    features: List[str],
    devs: List[int|float],
    pip_size: float,
    prefix: str = "fe_GMA",
) -> pl.DataFrame:
    """
    this function calculates Gaussian Moving Average And Weighted Bollinger Band.
    inputs:
    df: dataframe containing the raw feature
    w: Gaussian Parameter
    time_frame: time_frame for calculations
    devs: Parameter for calculations Bollinger Band,
    pip size: pip size of the pair
    prefix: prefix of feature name
    normalize: if True the function returns pipsize difference between GMA
        and last close-high-low price.
    """
    # print('=================')
    # print(features)

    # Assuming `df` is a polars DataFrame
    df = df.sort("_time")

    def gaussian_vectorized(source, bw):
        return np.exp(-1 * ((source / bw) ** 2)) / np.sqrt(2 * np.pi)

    i_values = np.arange(500)  
    array_w = gaussian_vectorized(i_values, w)
    array_w = array_w[array_w * 1e10 > 1]
    Sum_w = np.sum(array_w)
    base_features = features
    array_price = df.select(base_features).to_numpy()
    window_size = len(array_w)
    devs = np.array(devs)

    # Rolling calculations using convolution for weighted values
    rolled_close = np.convolve(
        array_price[:, base_features.index(f"M{time_frame}_CLOSE")]**1, array_w, 'valid'
    ) / Sum_w
    rolled_close_sq = np.convolve(
        array_price[:, base_features.index(f"M{time_frame}_CLOSE")]**2, array_w, 'valid'
    ) / Sum_w
    rolled_high = np.convolve(
        array_price[:, base_features.index(f"M{time_frame}_HIGH")]**1, array_w, 'valid'
    ) / Sum_w
    rolled_high_sq = np.convolve(
        array_price[:, base_features.index(f"M{time_frame}_HIGH")]**2, array_w, 'valid'
    ) / Sum_w
    rolled_low = np.convolve(
        array_price[:, base_features.index(f"M{time_frame}_LOW")]**1, array_w, 'valid'
    ) / Sum_w
    rolled_low_sq = np.convolve(
        array_price[:, base_features.index(f"M{time_frame}_LOW")]**2, array_w, 'valid'
    ) / Sum_w

    # Calculate standard deviations
    std_close = np.sqrt(np.maximum(0, rolled_close_sq - rolled_close**2))
    std_high = np.sqrt(np.maximum(0, rolled_high_sq - rolled_high**2))
    std_low = np.sqrt(np.maximum(0, rolled_low_sq - rolled_low**2))

    df = df.slice(window_size - 1)
    df = df.with_columns(
        [
            pl.lit(rolled_close).alias("close_GMA"),
            pl.lit(rolled_high).alias("high_GMA"),
            pl.lit(rolled_low).alias("low_GMA"),
        ]
    )

    # Add Bollinger Bands for each deviation level
    for dev in devs:

        df = df.with_columns([
            (pl.col('close_GMA') + std_close * dev).alias(f'Close-GMA_UBB{dev}'),
            (pl.col('close_GMA') - std_close * dev).alias(f'Close-GMA_LBB{dev}'),
            (pl.col('high_GMA') + std_high * dev).alias(f'High-GMA_UBB{dev}'),
            (pl.col('high_GMA') - std_high * dev).alias(f'High-GMA_LBB{dev}'),
            (pl.col('low_GMA') + std_low * dev).alias(f'Low-GMA_UBB{dev}'),
            (pl.col('low_GMA') - std_low * dev).alias(f'Low-GMA_LBB{dev}')
        ]).lazy()

    col_drop = list(set(df.collect_schema().names()) - set(['_time']))
    df = df.with_columns(
        (pl.col('close_GMA').diff()).alias(f"{prefix}_GMAClose_W{w}_diff_cndl_M{time_frame}")
    ).lazy()
    df = df.with_columns(
        (pl.col('high_GMA').diff()).alias(f"{prefix}_GMAHigh_W{w}_diff_cndl_M{time_frame}")
    ).lazy()
    df = df.with_columns(
        (pl.col('low_GMA').diff()).alias(f"{prefix}_GMALow_W{w}_diff_cndl_M{time_frame}")
    ).lazy()

    for base_feature in base_features:

        df = df.with_columns(
            (
                (
                    pl.col(base_feature) - pl.col('close_GMA')
                )
                / pip_size
            ).alias(f"{prefix}_{base_feature}-GMAClose_W{w}_cndl_M{time_frame}_norm")
        ).lazy()

        df = df.with_columns(
            (
                (
                    pl.col(base_feature) - pl.col('high_GMA')
                )
                / pip_size
            ).alias(f"{prefix}_{base_feature}-GMAHigh_W{w}_cndl_M{time_frame}_norm")
        ).lazy()

        df = df.with_columns(
            (
                (
                    pl.col(base_feature) - pl.col('low_GMA')
                )
                / pip_size
            ).alias(f"{prefix}_{base_feature}-GMALow_W{w}_cndl_M{time_frame}_norm")
        ).lazy()

        for dev in devs:
            if base_feature == base_features[0]:
                df = df.with_columns(
                    (
                        (
                            pl.col(f'Close-GMA_UBB{dev}')- pl.col(f'Close-GMA_LBB{dev}')
                        )/ pip_size
                    ).alias(f"{prefix}_UGBBClose{dev}-LGBBClose{dev}_W{w}_cndl_M{time_frame}")
                ).lazy()

                # df = df.with_columns(
                #     (
                #         (
                #             pl.col(f'Close-GMA_UBB{dev}')/ pl.col(f'Close-GMA_LBB{dev}')
                #         )
                #     ).alias(f"{prefix}_UGBBClose{dev}/LGBBClose{dev}_W{w}_cndl_M{time_frame}")
                # ).lazy()

                df = df.with_columns(
                    (
                        (
                            pl.col(f'High-GMA_UBB{dev}')- pl.col(f'High-GMA_LBB{dev}')
                        )/ pip_size
                    ).alias(f"{prefix}_UGBBHigh{dev}-LGBBHigh{dev}_W{w}_cndl_M{time_frame}")
                ).lazy()

                # df = df.with_columns(
                #     (
                #         (
                #             pl.col(f'High-GMA_UBB{dev}')/ pl.col(f'High-GMA_LBB{dev}')
                #         )
                #     ).alias(f"{prefix}_UGBBHigh{dev}/LGBBHigh{dev}_W{w}_cndl_M{time_frame}")
                # ).lazy()

                df = df.with_columns(
                    (
                        (
                            pl.col(f'Low-GMA_UBB{dev}')- pl.col(f'Low-GMA_LBB{dev}')
                        )/ pip_size
                    ).alias(f"{prefix}_UGBBLow{dev}-LGBBLow{dev}_W{w}_cndl_M{time_frame}")
                ).lazy()

                # df = df.with_columns(
                #     (
                #         (
                #             pl.col(f'Low-GMA_UBB{dev}')/ pl.col(f'Low-GMA_LBB{dev}')
                #         )
                #     ).alias(f"{prefix}_UGBBLow{dev}/LGBBLow{dev}_W{w}_cndl_M{time_frame}")
                # ).lazy()

            df = df.with_columns(
                (
                    (
                        pl.col(base_feature) - pl.col(f'Close-GMA_UBB{dev}')
                    )
                    / pip_size
                ).alias(f"{prefix}_{base_feature}-UGBBClose{dev}_W{w}_cndl_M{time_frame}_norm")
            ).lazy()

            df = df.with_columns(
                (
                    (
                        pl.col(base_feature) - pl.col(f'Close-GMA_LBB{dev}')
                    )
                    / pip_size
                ).alias(f"{prefix}_{base_feature}-LGBBClose{dev}_W{w}_cndl_M{time_frame}_norm")
            ).lazy()

            df = df.with_columns(
                (
                    (
                        pl.col(base_feature) - pl.col(f'High-GMA_UBB{dev}')
                    )
                    / pip_size
                ).alias(f"{prefix}_{base_feature}-UGBBHigh{dev}_W{w}_cndl_M{time_frame}_norm")
            ).lazy()

            df = df.with_columns(
                (
                    (
                        pl.col(base_feature) - pl.col(f'High-GMA_LBB{dev}')
                    )
                    / pip_size
                ).alias(f"{prefix}_{base_feature}-LGBBHigh{dev}_W{w}_cndl_M{time_frame}_norm")
            ).lazy()

            df = df.with_columns(
                (
                    (
                        pl.col(base_feature) - pl.col(f'Low-GMA_UBB{dev}')
                    )
                    / pip_size
                ).alias(f"{prefix}_{base_feature}-UGBBLow{dev}_W{w}_cndl_M{time_frame}_norm")
            ).lazy()

            df = df.with_columns(
                (
                    (
                        pl.col(base_feature) - pl.col(f'Low-GMA_LBB{dev}')
                    )
                    / pip_size
                ).alias(f"{prefix}_{base_feature}-LGBBLow{dev}_W{w}_cndl_M{time_frame}_norm")
            ).lazy()

    df = df.drop(col_drop)

    return df.collect()


def cal_OverLap_func(
    df: pl.DataFrame,
    features: List[str],
    w: int,
    w_sma : List[int],
    time_frame: int,
    pip_size: float,  # only for compatibility
    prefix: str = "fe_OL",
) -> pl.DataFrame:
    """
    This function calculates the overlap feature."""
    df = df.sort("_time")
    low = features[features.index(f'M{time_frame}_LOW')]
    high = features[features.index(f'M{time_frame}_HIGH')]

    df = df.with_columns(
        [
            pl.col(low).rolling_min(window_size=w).alias("Low_window"),
            pl.col(high).rolling_max(window_size=w).alias("High_window"),
        ]
    ).lazy()

    df = df.with_columns(
        [
            pl.col("Low_window").shift(w).alias("Low_shifted_window"),
            pl.col("High_window").shift(w).alias("High_shifted_window"),
        ]
    ).lazy()

    df = df.with_columns(
        [
            (pl.when(
                pl.min_horizontal([pl.col("High_window"), pl.col("High_shifted_window")])
                - pl.max_horizontal([pl.col("Low_window"), pl.col("Low_shifted_window")])
                >= 0
            ).then(
                pl.min_horizontal([pl.col("High_window"), pl.col("High_shifted_window")])
                - pl.max_horizontal([pl.col("Low_window"), pl.col("Low_shifted_window")])
            ).otherwise(0)).alias("Overlap"),
            (pl.col("High_window") - pl.col("Low_window")).alias("BarAmount"),
        ]
    ).lazy()

    df = df.with_columns(
        pl.when(
            pl.col("BarAmount") == 0
        ).then(0.001).otherwise(pl.col("BarAmount")).alias("BarAmount")
    ).lazy()

    df = df.with_columns(
        ((pl.col("Overlap") / pl.col("BarAmount")) * 100)
        .alias(f"{prefix}_W{w}_cndl_M{time_frame}")
    ).lazy()

    col_drop = list(
        set(df.collect_schema().names()) - set(['_time', f"{prefix}_W{w}_cndl_M{time_frame}"])
    )

    for window in w_sma:
        df = df.with_columns(
            (pl.col(f"{prefix}_W{w}_cndl_M{time_frame}").rolling_mean(window_size=window)).alias(
                f"{prefix}_W{w}_SMA{window}_cndl_M{time_frame}"
            )
        ).lazy()

    df = df.collect()
    df = df.drop(col_drop)

    return df


def history_indicator_calculator(feature_config, logger=default_logger):
    """
    Creating all indicators as features
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
            "fe_leg": {"func": cal_leg_base_func},
            "fe_cndl_shape_n_cntxt": {"func": cal_cndl_shape_n_cntxt_func},
            "fe_supertrend": {"func": cal_supertrend_func},
            "fe_FFD": {"func": cal_FFD_func},
            "fe_GMA": {"func": cal_GMA_n_GBB_func},
            "fe_OL": {"func": cal_OverLap_func},
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

                if fe_prefix == "fe_leg":
                    opts = {
                        "symbol": symbol,
                        "candle_timeframe": fe_leg_config[symbol]["timeframe"],
                        "window_size": fe_leg_config[symbol]["window_size"],
                        "exponents": fe_leg_config[symbol]["exponents"],
                        "percentage": fe_leg_config[symbol]["percentage"],
                        "features_folder_path": features_folder_path,
                    }
                else:
                    opts = {
                        "symbol": symbol,
                        "candle_timeframe": feature_config[symbol][fe_prefix]["timeframe"],
                        "window_size": feature_config[symbol][fe_prefix]["window_size"],
                        "features_folder_path": features_folder_path,
                        "feature_config": feature_config[symbol][fe_prefix],
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

                # Uncomment the for loop in order to plot legs (fe_leg feature) in Colab
                for df_path in pathes:
                    df_loaded = pl.read_parquet(df_path)
                    df = df.join(df_loaded, on="_time", how="left", coalesce=True)

                max_candle_timeframe = max(opts["candle_timeframe"])
                max_window_size = max(opts["window_size"])

                if fe_prefix == 'fe_GMA':
                    def gaussian_vectorized(source, bw):
                        return np.exp(-1 * ((source / bw) ** 2)) / np.sqrt(2 * np.pi)

                    i_values = np.arange(500)  
                    array_w = gaussian_vectorized(i_values, max_window_size)
                    max_window_size = len(array_w[array_w * 1e10 > 1])

                drop_rows = (max_window_size + 1) * (max_candle_timeframe / 5) - 1

                logger.info(
                    f"--> max_candle_timeframe:{max_candle_timeframe} | max_window_size:{max_window_size}| drop_rows:{drop_rows}"
                )

                df = df.with_row_index()
                if fe_prefix != 'fe_leg':
                    df = (
                        df.filter(pl.col("index") >= drop_rows)
                        .fill_null(strategy="forward")
                        .drop(*["index"])
                    )
                else:
                    df = (
                        df.fill_null(strategy="forward")
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
                features_folder_path = f"{root_path}/data/features/{ratio_prefix}/"
                Path(features_folder_path).mkdir(parents=True, exist_ok=True)

                if fe_prefix_replaced in list(
                    feature_config[symbol][ratio_prefix].keys()
                ):
                    ratio_config = feature_config[symbol][ratio_prefix][fe_prefix_replaced]

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

            if len(symbol_ratio_dfs) == 1:
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
    print("--> history_indicator_calculator DONE.")
