import pandas as pd
import numpy as np
from pathlib import Path
from dataset.logging_tools import default_logger
from dataset.configs.history_data_crawlers_config import root_path, symbols_dict
# import pywt  # Add wavelet transform support
# from scipy.signal import hilbert


def apply_hanning_window(array, window_size):
    """Apply a Hanning window to the data."""
    hanning_window = np.hanning(window_size)
    return array.flatten() * hanning_window

def compute_fft_features(selected_slice, window_size, sampling_rate):
    """Compute FFT features with DC component removed."""
    fft_values = np.fft.fft(selected_slice)
    fft_amplitude = np.abs(fft_values[:window_size // 2])
    fft_amplitude[1:] = 2 * fft_amplitude[1:]  # Double the amplitudes for positive frequencies
    
    # fft_amplitude[0] = 0  # Remove the DC component by setting the first vaue to 0
    
    # frequencies = np.fft.fftfreq(len(fft_amplitude), d=1 / sampling_rate)
    # positive_frequencies = frequencies[:window_size // 2]
    
    return fft_amplitude

def extract_fft_features(array, window_size, sampling_rate):
    num_features_fft = 20
    total_features = num_features_fft #+ num_features_wavelet + num_features_envelope + num_features_cepstrum + num_features_stats
    
    res = np.zeros([array.shape[0], total_features])  # result array
    res[:window_size, :] = np.nan  # fill initial rows with NaN
    
    for i in range(window_size, array.shape[0]):
        selected_slice = apply_hanning_window(array[i - window_size + 1: i + 1], window_size)

        # Compute FFT features
        fft_amplitude = compute_fft_features(selected_slice, window_size, sampling_rate)

        # Split the FFT amplitude into 10 quantiles and get the maximum value from each quantile
        quantiles = np.percentile(fft_amplitude[1:], np.linspace(0, 100, 21)[1:])  # 10 quantiles excluding the DC component
        for j in range(19):
            # Find the maximum value in each quantile range
            mask = (fft_amplitude[1:] >= quantiles[j]) & (fft_amplitude[1:] < quantiles[j+1])
            if np.any(mask):
                max_value_in_quantile = np.max(fft_amplitude[1:][mask])
                quantile_median = np.median(fft_amplitude[1:][mask])
                max_value_in_quantile = max_value_in_quantile ** quantile_median
            else:
                max_value_in_quantile = 0
            res[i, j] = max_value_in_quantile

    return res

def add_win_fe_base_func(
    df, symbol, raw_features, timeframes, window_sizes, sampling_rate, round_to=4, fe_prefix="fe_WIN_FREQ"
):
    new_columns = []

    for tf in timeframes:
        for w_size in window_sizes:
            assert tf == 5, "!!! For now, this code only works with 5M timeframe; tf must be 5."

            # Define feature column names based on updated `cal_window_max` function
            col_fft_magnitudes = [f"{fe_prefix}_fft_mag_W{w_size}_M{tf}_Top{i+1}" for i in range(20)]
            # col_fft_frequencies = [f"{fe_prefix}_fft_freq_W{w_size}_M{tf}_Top{i+1}" for i in range(10)]
            # col_fft_phases = [f"{fe_prefix}_fft_phase_W{w_size}_M{tf}_Top{i+1}" for i in range(10)]

            array = df[raw_features].to_numpy()

            res = extract_fft_features(array, w_size, sampling_rate)

            # Append the calculated results to the new_columns list as DataFrames
            new_columns.append(pd.DataFrame(res[:, 0:20].round(round_to), columns=col_fft_magnitudes, index=df.index))
            # new_columns.append(pd.DataFrame(res[:, 10:20].round(round_to), columns=col_fft_frequencies, index=df.index))
            # new_columns.append(pd.DataFrame(res[:, 20:30].round(round_to), columns=col_fft_phases, index=df.index))


    df = pd.concat([df] + new_columns, axis=1)

    return df
def add_win_fe_base_func(
    df, symbol, raw_features, timeframes, window_sizes, sampling_rate, round_to=4, fe_prefix="fe_WIN_FREQ"
):
    new_columns = []

    for tf in timeframes:
        for w_size in window_sizes:
            assert tf == 5, "!!! For now, this code only works with 5M timeframe; tf must be 5."

            # Define feature column names based on updated `cal_window_max` function
            col_fft_magnitudes = [f"{fe_prefix}_fft_MaxAmp_W{w_size}_M{tf}_Quantile_{i+1}" for i in range(20)]

            array = df[raw_features].to_numpy()

            # Extract FFT features
            res = extract_fft_features(array, w_size, sampling_rate)

            # Append the calculated results to the new_columns list as DataFrames
            new_columns.append(pd.DataFrame(res[:, 0:20].round(round_to), columns=col_fft_magnitudes, index=df.index))


    # Concatenate the original DataFrame with the newly calculated columns
    df = pd.concat([df] + new_columns, axis=1)

    return df


def history_fe_WIN_features_FREQ(feature_config, logger=default_logger):
    logger.info("- " * 25)
    logger.info("--> Start history_fe_WIN_FREQ_features function:")
    try:
        fe_prefix = "fe_WIN_FREQ"
        features_folder_path = f"{root_path}/data/features/{fe_prefix}/"
        Path(features_folder_path).mkdir(parents=True, exist_ok=True)


        base_candle_folder_path = f"{root_path}/data/features/fe_FFD/" # address fe_FFD parquet

        round_to = 7
        sampling_rate = 2  # Assumed sampling rate in Hz; adjust if necessary
        
        for symbol in feature_config.keys():
            logger.info(f"---> Symbol: {symbol}")
            logger.info("= " * 40)

            # base_cols = feature_config[symbol][fe_prefix]["base_columns"]
            # raw_features = [rf"fe_FFD-M5_{base_col}.*" for base_col in base_cols]

            file_name = base_candle_folder_path + f"fe_FFD_{symbol}.parquet"

            # Read the data using Pandas
            df = pd.read_parquet(file_name)
            raw_features = df.columns[1]  # Get the name of the second column
            needed_columns = ["_time", raw_features]
            df = pd.read_parquet(file_name, columns=needed_columns).sort_values("_time")
        

            # Ensure `_time` column is a datetime type
            if not pd.api.types.is_datetime64_any_dtype(df["_time"]):
                df["_time"] = pd.to_datetime(df["_time"], format="%Y-%m-%d %H:%M:%S")

            # Add the window-based features
            df = add_win_fe_base_func(
                df,
                symbol,
                raw_features=raw_features,
                timeframes=feature_config[symbol][fe_prefix]["timeframe"],
                window_sizes=feature_config[symbol][fe_prefix]["window_size"],
                sampling_rate=sampling_rate,
                round_to=round_to,
                fe_prefix=fe_prefix,
            )

            # Clean up the DataFrame, dropping the raw features and adding symbol info
            df = df.drop(columns=[raw_features])

            df["symbol"] = symbol
            df.to_parquet(f"{features_folder_path}/{fe_prefix}_{symbol}.parquet", index=False)
        logger.info("--> history_fe_WIN_FREQ_features run successfully.")
    except Exception as e:
        logger.exception("--> history_fe_WIN_FREQ_features error.")
        logger.exception(f"--> Error: {e}")
        raise ValueError("!!!")


if __name__ == "__main__":
    from configs.feature_configs_general import generate_general_config

    config_general = generate_general_config()
    history_fe_WIN_features_FREQ(config_general)
    default_logger.info("--> history_fe_WIN_FREQ_features DONE.")