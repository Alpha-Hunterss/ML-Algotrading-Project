import pandas as pd
import numpy as np


def remove_sequntaial_unchanged_rows(
    df,
    size_limit=500,
):
    # Get boolean mask where value changes
    mask_1 = df["Open"].ne(df["Open"].shift())
    mask_2 = df["Close"].ne(df["Close"].shift())
    mask = mask_1 & mask_2

    # Get indices where value changes
    change_indices = mask[mask].index

    # Get count of sequential values
    value_counts = np.diff(np.append(change_indices, len(df)))

    # Filter for counts greater than 100
    long_blocks = np.where(value_counts > size_limit)[0]

    drop_index = []

    for i, (start, end) in enumerate(
        zip(change_indices[long_blocks], change_indices[long_blocks + 1])
    ):
        # print(f"--> start: {start} , end: {end}")
        # df = df.drop(index=range(start,end))
        drop_index += list(range(start, end))

    df.drop(index=drop_index, inplace=True)
    df.sort_values("_time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    # print(f"--> len shapes: {len(drop_index)}")
    return df


def remove_weekends(df, weekends_day=["Saturday", "Sunday"], convert_tz=True):
    """
    the input df _time must be in UTC

    """

    if convert_tz:
        raise ValueError("!!! neet to change this part")
        df["_time"] = df["_time"].dt.tz_localize("UTC").dt.tz_convert("Europe/Istanbul")

    df.sort_values("_time", inplace=True)
    df["date"] = df["_time"].dt.normalize()
    df["week_day_name"] = df["date"].dt.day_name()
    df = df.loc[~df["week_day_name"].isin(weekends_day)]
    df = df.drop(columns=["date", "week_day_name"])
    df.sort_values("_time", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df
