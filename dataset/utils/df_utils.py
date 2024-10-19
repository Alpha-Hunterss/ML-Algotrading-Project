import numpy as np
import pandas as pd
import polars as pl
from utils.clean_data import remove_weekends


def generate_true_time_df_pandas(df):
    start_date = df.iloc[0, 0]
    end_date = df.iloc[-1, 0]
    true_time_df = pd.DataFrame(
        pd.date_range(start_date, end_date, freq="5min"), columns=["_time"]
    )
    true_time_df = remove_weekends(
        true_time_df, weekends_day=["Saturday", "Sunday"], convert_tz=False
    )
    return true_time_df


def ffill_df_to_true_time_steps(df):
    true_time_df = generate_true_time_df_pandas(df)
    df = true_time_df.merge(df, on=["_time"], how="left")
    # print("--> number of nulls in percent:",(df.isnull()["open"].sum()/df.shape[0])*100)
    df.sort_values("_time", inplace=True)
    df.reset_index(drop=True)
    df_filled = df.ffill()

    # Create a column to indicate if any cell in the row was forward-filled
    df["was_ffilled"] = (df != df_filled).any(axis=1)
    # Update the DataFrame with forward-filled values
    df.update(df_filled)
    return df
