import polars as pl
from pathlib import Path
from dataset.logging_tools import default_logger
from dataset.configs.history_data_crawlers_config import root_path

#!!: add truncate first rows
def shift_by_params(periods, time_frame, col_name=None):
    """
    this function returns shifted feature
    when col_name == none it returns shift indices
    """
    min_valid_index = periods * (time_frame // 5)
    # df_truncated = pl.col("index") >= min_valid_index
    df_truncated = pl.when(pl.col("index")>=min_valid_index).then(pl.col("index")).otherwise(0)

    idx = df_truncated - (
        ((pl.col("minutesPassed") % time_frame) / 5)
        + (periods - 1) * (time_frame / 5)
        + 1
    )

    validated_idx = pl.when(idx > 0).then(idx).otherwise(0).cast(pl.Int32, strict=False)

    if not (col_name):
        return validated_idx
    target_col = (pl.col(col_name).gather(validated_idx)).alias(
        f"{col_name}_-{periods}"
    )
    return target_col


def create_shifted_col(
    df, pair_name, periods, time_frame, columns=["OPEN", "HIGH", "LOW", "CLOSE"]
):  
    
    # print(f"-----> df.col: {df.columns}")
    return df.with_columns(
        [
            shift_by_params(
                periods=periods,
                time_frame=time_frame,
                col_name=f"M{time_frame}_{col}",
            )
            for col in columns
        ]
    )


def drop_first_day_polars(df):
    # Extract the first date from the "_time" column
    first_date = df.select(pl.col("_time").dt.date()).row(0)[0]

    # Filter the DataFrame to drop rows with the first date
    df_filtered = df.filter(pl.col("_time").dt.date() != first_date)

    return df_filtered


def history_cndl_shift(feature_config, logger=default_logger):
    logger.info("= " * 35)
    logger.info("--> start history_cndl_shift fumc:")
    fe_prefix = "fe_cndl_shift"

    base_candle_folder_path = f"{root_path}/data/realtime_candle/"
    features_folder_path = f"{root_path}/data/features/{fe_prefix}_raw/"
    Path(features_folder_path).mkdir(parents=True, exist_ok=True)
    for symbol in list(feature_config.keys()):
        if fe_prefix not in list(feature_config[symbol].keys()):
            continue
        logger.info("-  " * 20)

        sh_dfs = []
        shift_columns = feature_config[symbol][fe_prefix]["columns"]
        file_name = base_candle_folder_path + f"{symbol}_realtime_candle.parquet"
        df = pl.read_parquet(file_name)


        df = df.sort("_time").drop("symbol")
        shift_columns = feature_config[symbol][fe_prefix]["columns"]
        shift_configs = feature_config[symbol][fe_prefix]["shift_configs"]
        for shift_config in shift_configs:
            timeframe = shift_config["timeframe"]
            shift_sizes = shift_config["shift_sizes"]

            for shift_size in shift_sizes:
                # logger.info(
                #     f"symbol:{symbol} , timeframe: {timeframe} , shift_size: {shift_size} :"
                # )
                sh_df = create_shifted_col(
                    df, pair_name=symbol, periods=shift_size, time_frame=timeframe
                )
                new_cols = [
                    f"M{timeframe}_{col}_-{shift_size}" for col in shift_columns
                ]
                sh_dfs.append(sh_df[["_time"] + new_cols])

        shift_df = sh_dfs[0]

        if len(sh_dfs) > 1:
            for sh_df in sh_dfs[1:]:
                shift_df = shift_df.join(sh_df, on="_time", how="inner")

        elif len(sh_dfs) == 0:
            raise ValueError("!!! nothing to save.")

        save_file_name_ = features_folder_path + f"/{fe_prefix}_{symbol}.parquet"
        shift_df.write_parquet(save_file_name_)
        logger.info(f"--> {fe_prefix} | {symbol} saved.")

    logger.info("--> history_cndl_shift run successfully.")


if __name__ == "__main__":
    from configs.feature_configs_general import generate_general_config
    config_general = generate_general_config()
    history_cndl_shift(config_general)
    default_logger.info(f"--> history_cndl_shift DONE.")
