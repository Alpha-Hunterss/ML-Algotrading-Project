import json
import pandas as pd
import numpy as np
from typing import Tuple, List, Callable
import re
# from configs.selected_symbols_config import selected_symbol
from dataset.logging_tools import default_logger
from dataset.configs.history_data_crawlers_config import root_path, symbols_dict

def get_all_selected_features(feature_info: str | set[str]) -> Tuple[set, pd.DataFrame]:
    match feature_info:
        case str(f_set):
            with open(feature_info) as f:
                ff = json.load(f)
            f_set = {f for f in ff}
            df_fe = pd.DataFrame(f_set, columns=["feature_names"])
        case set():
            df_fe = pd.DataFrame(feature_info, columns=["feature_names"])
        case _:
            raise ValueError(f"Value Not Match: {feature_info}")
    return df_fe


def get_fe_timeframe(name: str):
    pattern = r"M(\d+)"
    matches = re.findall(pattern, name)
    # Filter out duplicates by converting the list to a set sicnce in many features we have two M followed by timeframe
    unique_matches = set(matches)
    # Convert the set back to a list to preserve order
    unique_matches = list(unique_matches)
    # return the first one found
    if unique_matches:
        return int(unique_matches[0])
    else:
        return None


def get_fe_windowsize(name: str):
    pattern = r"W(\d+)"
    matches = re.findall(pattern, name)
    # return the first one found
    if matches:
        return int(matches[0])
    else:
        return None


def get_fe_ratio_windowsize(name):
    if "ratio" in name:
        # Define the regex pattern for 'W' followed by a number
        pattern = r"W(\d+)"
        matches = re.findall(pattern, name)
        # return the second windowsize in the name
        if len(matches) >= 2:
            return int(matches[1])
        return None


def get_fe_shift_size(name):
    if "shift" in name:
        result = re.search(r"(?<=-)\d+", name)
        if result:
            return int(result.group())
    return None


def get_fe_symbol(f: str, selected_symbol=symbols_dict):
    all_symbols = list(selected_symbol.keys())
    for feature_type in all_symbols:
        if feature_type in f:
            return feature_type
    return None


def get_fe_type(f: str):
    if "_cndl_" in f and all(
        i not in f
        for i in ["shift", "EMA", "SMA", "RSI", "ptrn", "ratio", "WIN", "RSTD", "ATR"]
    ):
        return "fe_cndl"
    elif "ratio" in f:
        return "fe_ratio"
    elif f == "_time":
        return "_time"
    else:
        for feature_type in [
            "fe_time",
            "fe_RSTD",
            "fe_ATR",
            "fe_RSI",
            "fe_ratio",
            "fe_EMA",
            "fe_SMA",
            "fe_cndl_shift",
            "fe_WIN",
            "fe_cndl_ptrn",
            "fe_market_close",
            "RANDOM",
        ]:
            if feature_type in f:
                return feature_type
        return "other!"


def get_fe_ratio_subtype(f: str):

    if "ratio" in f:
        for ratio_fe in [
            "time",
            "RSTD",
            "ATR",
            "RSI",
            "EMA",
            "SMA",
            "cndl_shift",
            "WIN",
            "cndl_ptrn",
            "market_close",
            "RANDOM",
        ]:
            if ratio_fe in f:
                return ratio_fe
    return None


def get_feature_config(
    df_fe: pd.DataFrame, get_fe_funcs: List[Callable]
) -> pd.DataFrame:
    for func in get_fe_funcs:
        df_fe[f'{func.__name__.split("get_fe_")[1]}'] = df_fe["feature_names"].apply(
            func
        )
    assert (
        df_fe[df_fe.type == "other!"].shape[0] == 0
    ), f"!!! Invalid feature {df_fe[df_fe.type == 'other!'].feature_names}"
    return df_fe


def cal_max_lookback(df_fe: pd.DataFrame) -> int:
    max_lb = [
        df_fe["timeframe"],
        df_fe["timeframe"] * df_fe["windowsize"],
        df_fe.loc[df_fe.type == "fe_ratio", "timeframe"]
        * df_fe.loc[df_fe.type == "fe_ratio", "ratio_windowsize"],
    ]
    # print(f'Max tf: {max(max_lb[0]):.0f}, Max tf*window {max(max_lb[1]):.0f}, Max tf*ratio_window {max(max_lb[2]):.0f} \n')
    max_lookback = int(max([max(i) for i in max_lb]))
    argmax_lookback = np.argmax([max(i) for i in max_lb])
    df_fe["max_lookback"] = max_lb[argmax_lookback]
    feature_max_lookback = df_fe.loc[df_fe["max_lookback"].idxmax(), "feature_names"]
    df_fe.drop(columns=["max_lookback"], inplace=True)
    # print(f'Max lookback period is {max_lookback} minutes or {max_lookback//288} days \n')
    # print(f'The feature causing the max_lookback is {feature_max_lookback}')
    return max_lookback


def extract_config_from_selected_feature_init(df_fe: pd.DataFrame, logger=default_logger):

    logger.info(f"All feature types in the dataset: {df_fe.type.unique()}")
    logger.info(f"All symbols in the dataset: {df_fe.symbol.unique()}")
    logger.info(f"All timeframes in the features: {df_fe.timeframe.unique()}")
    logger.info(f"All windowsizes in the features: {df_fe.windowsize.unique()}")

    selected_config = {}
    selected_config["NOT_SYMBOL"] = {}
    selected_config["NOT_SYMBOL"]["fe_time"] = list(
        set(df_fe[df_fe.type == "fe_time"].feature_names)
    )
    selected_config
    for symbol, df_sym in df_fe.groupby("symbol"):
        # ? base_candle_timeframe
        selected_config[f"{symbol}"] = {}
        selected_config[f"{symbol}"]["base_candle_timeframe"] = sorted(
            [int(f) for f in df_sym.timeframe.unique() if pd.notna(f) and int(f) != 5]
        )

        for fe_type, df in df_sym.groupby("type"):
            # ? candle
            if fe_type == "fe_cndl":
                selected_config[f"{symbol}"]["fe_cndl"] = sorted(
                    [int(f) for f in df.timeframe.unique() if pd.notna(f)]
                )

            # ? indicators
            elif fe_type in [
                "fe_WIN",
                "fe_RSI",
                "fe_ATR",
                "fe_RSTD",
                "fe_EMA",
                "fe_SMA",
            ]:
                selected_config[f"{symbol}"][f"{fe_type}"] = {}
                selected_config[f"{symbol}"][f"{fe_type}"]["timeframe"] = sorted(
                    [int(f) for f in df.timeframe.unique() if pd.notna(f)]
                )
                selected_config[f"{symbol}"][f"{fe_type}"]["window_size"] = sorted(
                    [int(f) for f in df.windowsize.unique() if pd.notna(f)]
                )
                if fe_type == "fe_ATR":
                    selected_config[f"{symbol}"][f"{fe_type}"]["base_columns"] = [
                        "CLOSE",
                        "HIGH",
                        "LOW",
                    ]
                else:
                    selected_config[f"{symbol}"][f"{fe_type}"]["base_columns"] = [
                        "CLOSE"
                    ]

            # ? ratio
            elif fe_type == "fe_ratio":
                selected_config[f"{symbol}"][f"{fe_type}"] = {}
                for ratio_type, df_r in df.groupby("ratio_subtype"):
                    selected_config[f"{symbol}"][f"{fe_type}"][f"{ratio_type}"] = {}
                    selected_config[f"{symbol}"][f"{fe_type}"][f"{ratio_type}"][
                        "timeframe"
                    ] = sorted([int(f) for f in df_r.timeframe.unique() if pd.notna(f)])
                    selected_config[f"{symbol}"][f"{fe_type}"][f"{ratio_type}"][
                        "window_size"
                    ] = sorted(
                        [
                            (int(r.windowsize), int(r.ratio_windowsize))
                            for _, r in df_r.iterrows()
                        ]
                    )

            # ? candle shift
            elif fe_type == "fe_cndl_shift":
                # fe_type_new = fe_type.replace('fe_','')
                selected_config[f"{symbol}"][f"{fe_type}"] = {}
                # columns
                fe_sh_names = list(df.feature_names)
                cndl_type_columns = []
                for cndl_type in ["OPEN", "HIGH", "LOW", "CLOSE"]:
                    if any([cndl_type in f for f in fe_sh_names]):
                        cndl_type_columns.append(cndl_type)
                selected_config[f"{symbol}"][f"{fe_type}"][
                    "columns"
                ] = cndl_type_columns

                # timeframe and shift_sizes
                tf_shift = list(
                    set(
                        [
                            (int(r.timeframe), int(r.shift_size))
                            for _, r in df.iterrows()
                        ]
                    )
                )
                tf_shift_final = []
                for t in tf_shift:
                    tf_shift_final.append({"timeframe": t[0], "shift_sizes": [t[1]]})
                selected_config[f"{symbol}"][f"{fe_type}"][
                    "shift_configs"
                ] = tf_shift_final
    return selected_config


def update_parent_features_using_ratio_fe(df_fe, selected_config, save_path):
    for symbol, df_sym in df_fe.groupby("symbol"):
        for ratio_type, df_r in df_sym.groupby("ratio_subtype"):

            tf_sym_type = [int(f) for f in df_r.timeframe.unique()]
            w_sym_type = list(
                set(
                    [
                        int(f)
                        for f in list(df_r.windowsize.unique())
                        + list(df_r.ratio_windowsize.unique())
                    ]
                )
            )
            fe_sh_names = list(df_r.feature_names)
            cndl_type_columns = []
            for cndl_type in ["OPEN", "HIGH", "LOW", "CLOSE"]:
                if any([cndl_type in f for f in fe_sh_names]):
                    cndl_type_columns.append(cndl_type)

            if tf_sym_type:
                if symbol in selected_config:
                    if f"fe_{ratio_type}" in selected_config[f"{symbol}"]:
                        # print('@@@@@@@@',symbol, ratio_type)
                        # print(f"before {selected_config[f'{symbol}'][f'fe_{ratio_type}']['timeframe']}")
                        # print(f"before {selected_config[f'{symbol}'][f'fe_{ratio_type}']['window_size']}")

                        selected_config[f"{symbol}"][f"fe_{ratio_type}"][
                            "timeframe"
                        ] = list(
                            set(
                                selected_config[f"{symbol}"][f"fe_{ratio_type}"][
                                    "timeframe"
                                ]
                                + tf_sym_type
                            )
                        )
                        selected_config[f"{symbol}"][f"fe_{ratio_type}"][
                            "window_size"
                        ] = list(
                            set(
                                selected_config[f"{symbol}"][f"fe_{ratio_type}"][
                                    "window_size"
                                ]
                                + w_sym_type
                            )
                        )

                        # print(f"after {selected_config[f'{symbol}'][f'fe_{ratio_type}']['timeframe']}")
                        # print(f"after {selected_config[f'{symbol}'][f'fe_{ratio_type}']['window_size']}")
                    else:
                        # print('$$$$$$$$$',symbol, ratio_type)
                        selected_config[f"{symbol}"][f"fe_{ratio_type}"] = {}
                        selected_config[f"{symbol}"][f"fe_{ratio_type}"][
                            "base_columns"
                        ] = cndl_type_columns
                        selected_config[f"{symbol}"][f"fe_{ratio_type}"][
                            "timeframe"
                        ] = tf_sym_type
                        selected_config[f"{symbol}"][f"fe_{ratio_type}"][
                            "window_size"
                        ] = w_sym_type
                else:
                    # print('########',symbol, ratio_type)
                    selected_config[f"{symbol}"] = {}
                    selected_config[f"{symbol}"][f"fe_{ratio_type}"] = {}
                    selected_config[f"{symbol}"][f"fe_{ratio_type}"][
                        "base_columns"
                    ] = cndl_type_columns
                    selected_config[f"{symbol}"][f"fe_{ratio_type}"][
                        "timeframe"
                    ] = tf_sym_type
                    selected_config[f"{symbol}"][f"fe_{ratio_type}"][
                        "window_size"
                    ] = w_sym_type
    with open(save_path, "w") as f:
        json.dump(selected_config, f, indent=5)
    return selected_config

def extract_config_from_selected_feature(
    feature_info: str | set[str], save_path: str = f"{root_path}/configs/updated_feature_config.json"
):
    df_fe = get_all_selected_features(feature_info=feature_info)
    df_fe = get_feature_config(
        df_fe=df_fe,
        get_fe_funcs=[
            get_fe_symbol,
            get_fe_type,
            get_fe_timeframe,
            get_fe_windowsize,
            get_fe_ratio_subtype,
            get_fe_ratio_windowsize,
            get_fe_shift_size,
        ],
    )
    max_lookback = cal_max_lookback(df_fe=df_fe)
    selected_config = extract_config_from_selected_feature_init(df_fe=df_fe)
    selected_config = update_parent_features_using_ratio_fe(
        df_fe=df_fe, selected_config=selected_config, save_path=save_path
    )
    return selected_config
