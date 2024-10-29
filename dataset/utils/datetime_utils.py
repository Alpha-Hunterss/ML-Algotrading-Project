from dateutil.relativedelta import relativedelta


def create_n_month_intervals(start_time, stop_time, month_offset=3, overlap_days=5):
    intervals = []
    current_start = start_time

    while current_start < stop_time:
        current_end = current_start + relativedelta(months=month_offset)
        if current_end > stop_time:
            current_end = stop_time
        intervals.append(
            (current_start - relativedelta(days=overlap_days), current_end)
        )
        current_start = current_end

    return intervals



def drop_first_day_pandas(df):
    # ? drop first day
    firt_date = df["_time"].dt.date[0]
    df["_date"] = df["_time"].dt.date
    df = df.loc[df["_date"] != firt_date]
    return df
