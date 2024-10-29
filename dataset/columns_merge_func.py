import polars as pl
include_target = False
import gc
import pandas as pd
from dataset.logging_tools import default_logger
import polars as pl
from collections import defaultdict
from dataset.extract_config_from_features import (
    get_all_selected_features,
)
from dataset.configs.history_data_crawlers_config import root_path
from pathlib import Path
from dataset.utils.reduce_memory import reduce_mem_usage


# Define the prefixes - !!! order maters.
prefixes = [
    "fe_cndl_shift",
    "fe_cndl",
    "fe_WIN_argmin",
    "fe_WIN_argmax",
    "fe_WIN_max",
    "fe_WIN_min",
    "fe_WIN",
    "fe_ratio_RSI",
    "fe_ratio_EMA",
    "fe_ratio_RSTD",
    "fe_ratio_SMA",
    "fe_ratio_ATR",
    "fe_ratio",
    "fe_RSTD",
    "fe_RSI",
    "fe_EMA",
    "fe_SMA",
    "fe_ATR",
    "fe_market_close",
]


# Function to categorize keys
def categorize_keys(data, prefixes=prefixes):
    categorized = defaultdict(list)
    uncategorized = []

    for key in data.keys():
        categorized_flag = False
        for prefix in prefixes:
            if key.startswith(prefix):
                categorized[prefix].append(key)
                categorized_flag = True
                break
        if not categorized_flag:
            uncategorized.append(key)

    return categorized, uncategorized

def add_symbol(col_name, symbol, prefixes):
    for prefix in prefixes:
        if col_name.startswith(prefix):
            new_col_name = col_name.replace(prefix, f"{prefix}_{symbol}")
            return new_col_name
    return col_name

def add_symbol_to_prefixes(df_cols, symbol, prefixes=prefixes):
    """
    Adds a given symbol after specific prefixes in the column names of a Polars DataFrame.

    Parameters:
    df (pl.DataFrame): The Polars DataFrame.
    symbol (str): The symbol to add after the prefixes.
    prefixes (list): List of known prefixes to search for in the column names.

    Returns:
    pl.DataFrame: The DataFrame with updated column names.
    """

    new_column_names = {col: add_symbol(col, symbol, prefixes) for col in df_cols}

    return new_column_names

def group_by_symbol_and_rename(df, symbol_col="symbol"):
    """
    Groups the DataFrame by the symbol column, removes the symbol column, and appends the symbol value to each feature column name.

    Parameters:
    df (pl.DataFrame): The Polars DataFrame.
    symbol_col (str): The name of the column containing the symbols.

    Returns:
    pl.DataFrame: The DataFrame with updated column names and without the symbol column.
    """
    partitions = df.partition_by(symbol_col)

    # Initialize an empty list to store the renamed partitions
    renamed_partitions = []

    # Process each partition
    for partition in partitions:
        # Get the symbol value from the first row (all rows have the same symbol in the partition)
        symbol = partition[symbol_col][0]
        renamed_partition = partition.rename(
            add_symbol_to_prefixes(partition.columns, symbol)
        )
        renamed_partition = renamed_partition.drop(symbol_col)

        # Add the renamed partition to the list
        renamed_partitions.append(renamed_partition)

    # Combine all the renamed partitions on the _time column
    result_df = renamed_partitions[0]
    for renamed_partition in renamed_partitions[1:]:
        result_df = result_df.join(
            renamed_partition, on="_time", how="full", coalesce=True
        )

    return result_df

def history_columns_merge(feature_config, logger=default_logger,general_mode=False):
    logger.info("- " * 25)
    logger.info("--> start history_columns_merge fumc:")

    if not general_mode:
        feature_map_path = "data/models/jamesv01/tradeset_usdjpy_feature_map.json"
        f_cols = set(get_all_selected_features(feature_map_path)["feature_names"])

    fe_refrece_list = [
        "fe_cndl",
        "fe_RSI",
        "fe_RSTD",
        "fe_ATR",
        "fe_EMA",
        "fe_SMA",
        "fe_ratio",
        "fe_cndl_shift",
        "fe_WIN",
        "fe_cndl_ptrn",
        "fe_market_close",
    ]
    fe_list = []
    for sym in feature_config:
        fe_list += list(feature_config[sym].keys())

    fe_list = list(set(fe_refrece_list) & set(fe_list))

    basic_sym = list(feature_config.keys())[0]
    
    # try:
    feature_folder = f"{root_path}/data/features/"
    # base dataframe to merge data on it.
    main_symbol_st_one_path = f"{root_path}/data/stage_one_data/{sym}_stage_one.parquet"
    df_dataset = pl.read_parquet(main_symbol_st_one_path,columns=["_time"])
    df_dataset = df_dataset.sort("_time")

    logger.info("--> add fe_time.")

    file_name = f"{feature_folder}/fe_time/fe_time.parquet"
    df_dataset = df_dataset.join(
    pl.read_parquet(file_name).sort("_time").drop("symbol"),
    left_on="_time", right_on="_time", how="left", coalesce=True
    )
    gc.collect()

    for symbol in feature_config:
        logger.info(f" ^ - ^ " * 10)
        sy_fe = list(
            set(list(feature_config[symbol].keys())) & set(fe_refrece_list)
        )
        sy_fe.append("fe_market_close")
        sy_fe = list(set(sy_fe))
        for feture in sy_fe:
            try:
            
                logger.info(f"--> {symbol} | {feture}")
                logger.info(f"--> {symbol} | -->{feture}<---- --------------------------")
                df = pl.read_parquet(
                    f"{feature_folder}/{feture}/{feture}_{symbol}.parquet"
                )
                
                df = df.sort("_time").drop("symbol")
                df = df.rename(add_symbol_to_prefixes(df.columns, symbol))
            except Exception as e:
                logger.error(f"!!! cant load {symbol} | {feture}")
                logger.error(e)
                raise ValueError("!!!!")

            df_dataset = df_dataset.join(
                df, left_on="_time", right_on="_time", how="left", coalesce=True
            )
            del df
            gc.collect()

    df_colls = list(df_dataset.columns)
    gg = [f for f in df_colls if "fe_ratio" in f]

    if not general_mode:
        diff_cols = set(f_cols) - set(list(df_colls))
        logger.info(f"--> len final diff cols: {len(diff_cols)}")
        # logger.info(f"--> diff cols: {diff_cols}")
        # logger.info(gg)
        df_dataset = df_dataset[list(f_cols)]


    logger.info(f"--> {df_dataset.shape}")
    log = df_dataset.select(pl.all().is_null().sum()).to_dicts()[0]
    log_df = pd.DataFrame(log, index=["nulls"]).T.sort_values(
        "nulls", ascending=False
    )
    log = log_df.loc[log_df.nulls > 0].to_dict()["nulls"]

    n_nulls_all = df_dataset.select(pl.all().is_null().sum()).sum_horizontal()
    logger.info(
        f"--> number of nulls all in df_dataset: {n_nulls_all} / {df_dataset.shape[0]}"
    )
    # logger.info(f"--> columns with nulls: {log}")

    df_dataset = df_dataset.drop_nulls()

    # leakage assert
    cols_assert = [
        col
        for col in df_dataset.columns
        if ("fe_" not in col) and ("trg_" not in col) and ("_time" != col)
    ]
    assert (
        len(cols_assert) == 0
    ), f"!!! columns must be either target or features. {cols_assert}"

    all_cols = df_dataset.columns
    trg_cols = [col for col in all_cols if "trg_" in col]
    fe_cols = list(set(all_cols) - set(trg_cols + ["_time"]))
    logger.info(f"--> trg_cols:{len(trg_cols)} | fe_cols:{len(fe_cols)}")

    # save dataset
    df_dataset = df_dataset.with_columns(pl.lit("dataset").alias("symbol"))
    fe_prefix = "dataset"
    dataset_folder_path = f"{root_path}/data/{fe_prefix}/"
    Path(dataset_folder_path).mkdir(parents=True, exist_ok=True)
    file_name = dataset_folder_path + f"/{fe_prefix}.parquet"
    reduce_mem_usage(df_dataset.to_pandas()).to_parquet(file_name)

    logger.info(f"--> df final shape: {df_dataset.shape} | dataset saved.")
    logger.info("--> history_fe_time run successfully.")


if __name__ == "__main__":
    from configs.feature_configs_general import generate_general_config
    config_general = generate_general_config()
    history_columns_merge(config_general,general_mode=True)
    default_logger.info(f"--> history_columns_merge DONE.")
