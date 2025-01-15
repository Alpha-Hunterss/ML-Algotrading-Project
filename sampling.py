import pandas as pd
import polars as pl
import os, glob
from pyarrow.parquet import ParquetFile
import re
def sampling_func(
    sampling_dict : dict,
    C5M_data_path : str,
    target_symbol : str,
    trade_mode : str,
):
    features_folder_path = C5M_data_path.replace('stage_one_data' , 'features')
    df = pl.read_parquet(f"{C5M_data_path}/{target_symbol}_stage_one.parquet", columns=['_time' , 'close'])
    df = df.sort("_time")
    df = df.with_row_index()
    if sampling_dict['method_st'] == 'EMA':
        max_candle_timeframe = max(sampling_dict['sterategy']['EMA']['time_frame'])
        window_size = sampling_dict['sterategy']['EMA']['window_size']
        needed_col_pattern = re.compile(rf'EMA')
        name_feature = []
        for tf in sampling_dict['sterategy']['EMA']['time_frame'] :
            file_path = f"{features_folder_path}/fe_EMA/unmerged/fe_EMA_{window_size}_{target_symbol}_M{tf}.parquet"
            needed_columns = [f.name for f in ParquetFile(file_path).schema if needed_col_pattern.match(f.name) or f.name=='_time']
            df_loaded = pl.read_parquet(file_path, columns=needed_columns)
            df = df.join(df_loaded, on="_time", how="left", coalesce=True)
            name_feature.append(f"EMA_M{tf}_CLOSE_W{window_size}_cndl_M{tf}")
            
        drop_rows = (window_size + 1) * (max_candle_timeframe / 5) - 1
        df = (
            df.filter(pl.col("index") >= drop_rows)
            .fill_null(strategy="forward")
            .drop(*["index"])
        )
        df = df.drop_nulls()
        for name in name_feature:
            if trade_mode == 'long':
                df = df.filter((pl.col("close") > pl.col(name)))
            else :
                df = df.filter((pl.col("close") < pl.col(name)))
    return df["_time"].to_list()
        





