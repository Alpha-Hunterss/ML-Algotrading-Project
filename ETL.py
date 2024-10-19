import polars as pl
import numpy as np
import pandas as pd
import datetime
# from utils.general_utils import reduce_mem_usage
from configss.symbols_info import symbols_dict
from pyarrow.parquet import ParquetFile
from target_generation import calculate_classification_target_numpy_ver
import time
from typing import List



def read_data_manual(
    path: str ="/kaggle/input/tradeset-002/database.parquet",
    selected_fes : List[str]|None = None):
    """
    This function reads input dataset including final_dataset, basic df and stage_two df
    Inputs:
        path: path to dfs' folder
    Output:
        dataset
    """
    if selected_fes:
        print(f'Selecting ')
        df_all = pd.read_parquet(path, columns=selected_fes)
    else:
        df_all = pd.read_parquet(path)

    print("df shape: ", df_all.shape)
    return df_all

def ETL(
    path,# path of dataset
    C5M_data_path,
    trade_mode,
    target_symbol,
    trg_look_ahead,
    trg_take_profit,
    trg_stop_loss,
    n_rand_features,
    target_col,# name of target column
    base_time_frame,# for calculating targerts
):
    raw_columns = [f.name for f in ParquetFile(path).schema]
    print(f'Len all columns in dataframe is {len(raw_columns)}')
    df = pd.read_parquet(path)
    print(f'Len read columns is {df.shape[1]}')
    print("Calculating target --->")
    window_size = int(trg_look_ahead // base_time_frame)
    
    df_raw = pd.read_parquet(f"{C5M_data_path}/{target_symbol}_stage_one.parquet", 
      columns = [
        '_time',
        "close",
        "high",
        "low",
      ]
    ).rename(columns={
        "close":f"{target_symbol}_M5_CLOSE",
        "high":f"{target_symbol}_M5_HIGH",
        "low":f"{target_symbol}_M5_LOW",
    })

    
    
    
    array = df.merge(df_raw, on = '_time', how = 'left')[
        [f"{target_symbol}_M5_CLOSE", f"{target_symbol}_M5_HIGH", f"{target_symbol}_M5_LOW"]
    ].to_numpy()
    tic = time.time()
    df["target"] = calculate_classification_target_numpy_ver(
            array,
            window_size,
            symbol_decimal_multiply = symbols_dict[target_symbol]["pip_size"],
            take_profit = trg_take_profit,
            stop_loss = trg_stop_loss,
            mode = trade_mode,
        )
    toc = time.time()
    df.dropna(inplace = True)
    print(f"---> Target {target_col} has been generated in {toc-tic:.2f} seconds")
    print("df shape: ", df.shape)
    df.set_index(["_time"], inplace=True, drop=True)
    df["target"] = df["target"].astype(int)

    ##? set targets to 0 in bad hours 
    df.loc[(df.index.get_level_values('_time').time>=datetime.time(0, 0))&(df.index.get_level_values('_time').time<=datetime.time(1, 0)),'target'] = 0
    df = remove_future_redundendat_columns(df)

    # _____________________________ADD RANDOM FEATURES_______________________________________________
    random_features = []
    if n_rand_features is not None:
        for i in range(n_rand_features):
            df[f'RANDOM_{i}'] = np.random.random(df.shape[0])
            random_features.append(f'RANDOM_{i}')
    print("=" * 30)
    print("--> df final shape:", df.shape)
    print(
        f"--> df min_time: {df.index.get_level_values('_time').min()} | df max_time: {df.index.get_level_values('_time').max()}"
    )
    print(
        f"--> number of unique days: {df.index.get_level_values('_time').unique().shape[0]}"
    )
    print("=" * 30)
    return df


def remove_some_rows(df_all, nan_count_remover=20):
    """
    some rows need to be removed (exp_NaN)
    :nan_count_remover: maximum number of NaN it will drop
    """
    if df_all[df_all.isnull().any(axis=1)].shape[0] < nan_count_remover:
        print("df_all_shape_before_null_Removal", df_all.shape)
        nan_columns = [i for i in df_all.columns if df_all[i].isna().any()]
        df_all = df_all.dropna(subset=nan_columns)
        print("df_all_shape_after_null_Removal", df_all.shape)
    return df_all


def remove_future_redundendat_columns(df_all):
    """
    get dataframe and remove listed futures(cols) and return the dataframe
    """

    other_target_cols = [col for col in df_all.columns if "trg_" in col]

    if len(other_target_cols) > 0:
        print("columns_removed: ", other_target_cols)
    
    
    df_all = df_all.drop(columns=other_target_cols, errors="ignore")
    
    from sklearn.feature_selection import VarianceThreshold
    
    # #?? DROP constant columns:
    # print("--> DROP constant columns.")
    # sel = VarianceThreshold(threshold=0.01) # 0.1 indicates 99% of observations approximately
    # sel.fit(df_all)  # fit finds the features with zero variance
    # constant_cols = [x for x in df_all.columns if x not in df_all.columns[sel.get_support()]]
    # df_all.drop(columns=constant_cols,inplace=True)
    

    return df_all


def add_columns(df_all):
    """
    gets dataframe adds some columns to it and returns it
    """
    time_df = pd.DataFrame(df_all.index.get_level_values("_time").unique()).sort_values(
        ["_time"]
    )
    time_df["day_num"] = time_df["_time"].rank()
    df_all = pd.merge(
        df_all.reset_index(), time_df, how="inner", left_on="_time", right_on="_time"
    ).set_index(["_time"])
    return df_all
