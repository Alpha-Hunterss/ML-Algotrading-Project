import polars as pl
import numpy as np
from pathlib import Path
from dataset.logging_tools import default_logger
from dataset.configs.history_data_crawlers_config import root_path
import pywt
import time
from statsmodels.tsa.stattools import adfuller
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from numba import njit
from joblib import Parallel, delayed

# Optimized fractional differentiation with Numba
@njit
def frac_diff_fast(series, d, window_length):
    weights = np.ones(1)
    for k in range(1, window_length):
        weight = -weights[-1] * (d - k + 1) / k
        weights = np.append(weights, weight)
    output = np.convolve(series, weights[::-1], mode='valid')
    return output

# Optimized function to find optimal d
def find_optimal_d(series, window_length, d_start=0.4, d_end=0.9, d_step=0.025, target_p=0.05, max_iter=50):
    series_np = series.to_numpy()  # Convert Polars Series to NumPy
    best_d = d_start
    best_p = float('inf')
    best_diff = float('inf')
    for i, d in enumerate(np.arange(d_start, d_end + d_step, d_step)):
        if i >= max_iter:
            print(f"Max iterations reached. Best d={best_d}, p={best_p}, diff={best_diff}")
            break
        ffd_series = frac_diff_fast(series_np, d, window_length)
        if len(ffd_series) < 2:
            continue
        adf_result = adfuller(ffd_series)
        p_value = adf_result[1]
        diff = abs(p_value - target_p)
        if diff < best_diff:
            best_d = d
            best_p = p_value
            best_diff = diff
        if best_diff < 0.15:
            print(f"Found optimal d={best_d} with p={best_p} (diff={best_diff})")
            break
    return best_d, best_p, frac_diff_fast(series_np, best_d, window_length)

# Wavelet denoising (kept as-is, could be optimized further with batch processing)
def wavelet_denoise(signal, wavelet, level):
    coeffs = pywt.wavedec(signal.flatten(), wavelet, level=level)
    coeff_array, coeff_slices = pywt.coeffs_to_array(coeffs)
    threshold = np.std(coeff_array) * np.sqrt(2 * np.log(len(coeff_array)))
    coeff_array[np.abs(coeff_array) < threshold] = 0
    filtered_coeffs = pywt.array_to_coeffs(coeff_array, coeff_slices, output_format="wavedec")
    reconstructed_signal = pywt.waverec(filtered_coeffs, wavelet)

    original_length = len(signal)
    if len(reconstructed_signal) > original_length:
        reconstructed_signal = reconstructed_signal[:original_length]
    elif len(reconstructed_signal) < original_length:
        reconstructed_signal = np.pad(reconstructed_signal, (0, original_length - len(reconstructed_signal)), 'edge')
    return reconstructed_signal

# Parallelized window processing function
def process_window(i, array, window_size, sampling_rate):
    selected_slice = array[i - window_size + 1: i + 1]
    if len(selected_slice) < 2:
        return np.full(10, np.nan)

    # 1. Wavelet Denoising
    reconstructed_level4 = wavelet_denoise(selected_slice, "bior4.4", level=4)
    
    # 2. FFD Calculation
    reconstructed_series = pl.Series(reconstructed_level4)
    optimal_d, optimal_p, FFD_slice = find_optimal_d(reconstructed_series, window_length=5)
    
    # 3. FFD Centered
    FFD_centered = FFD_slice - np.mean(FFD_slice)

    # 4. FFT
    if len(FFD_centered) < 2:
        return np.full(10, np.nan)
    
    fft_result = np.fft.fft(FFD_centered)
    n = len(FFD_centered)
    frequencies = np.fft.fftfreq(n, d=5/60)
    magnitude_spectrum = np.abs(fft_result)
    phase_spectrum = np.angle(fft_result)

    half_n = n // 2
    if half_n == 0:
        return np.full(10, np.nan)

    positive_frequencies = frequencies[:half_n]
    positive_magnitudes = magnitude_spectrum[:half_n]
    positive_phases = phase_spectrum[:half_n]
    positive_frequencies[0] = 0

    top_10_indices = np.argsort(positive_magnitudes)[-10:][::-1]
    top_10_frequencies = positive_frequencies[top_10_indices]
    top_10_magnitudes = positive_magnitudes[top_10_indices]
    top_10_phases = positive_phases[top_10_indices]

    data_3d = np.column_stack((top_10_frequencies, top_10_magnitudes, top_10_phases))
    scaler = StandardScaler()
    data_3d_scaled = scaler.fit_transform(data_3d)
    pca = PCA(n_components=1)
    data_1d = pca.fit_transform(data_3d_scaled)

    return data_1d.flatten()

def cal_window_max(array, window_size, sampling_rate, logger=default_logger):
    num_features = 10
    res = np.zeros((array.shape[0], num_features))
    res[:window_size, :] = np.nan

    # Parallel processing of windows
    results = Parallel(n_jobs=-1)(delayed(process_window)(i, array, window_size, sampling_rate)
                                  for i in range(window_size, array.shape[0]))
    
    # Fill results into res array
    for i, result in enumerate(results, start=window_size):
        res[i, :] = result

    # Progress logging (simplified)
    array_shape = array.shape[0] // 10
    for perc in range(10, 100, 10):
        if window_size + perc * array_shape // 10 < array.shape[0]:
            logger.info(f"---> Did {perc} perc of the job ...")

    return res

def add_win_fe_base_func(
    df, raw_features, timeframes, window_sizes,
    sampling_rate, round_to=4, fe_prefix="fe_WIN_FREQ", logger=default_logger,
):
    new_columns = []

    for tf in timeframes:
        for w_size in window_sizes:
            logger.info(f"---> Doing window size {w_size} ...")
            assert tf == 5, "!!! For now, this code only works with 5M timeframe; tf must be 5."

            col_FREQ_ftr = [f"{fe_prefix}_FREQ_W{w_size}_M{tf}_Top{i+1}" for i in range(10)]
            array = df.select(raw_features).to_numpy()

            res = cal_window_max(array, w_size, sampling_rate)

            # Convert results to Polars DataFrame
            new_columns.append(pl.DataFrame(
                res[:, 0:10].round(round_to),
                schema=col_FREQ_ftr
            ))

    # Concatenate horizontally with Polars
    df = pl.concat([df] + new_columns, how="horizontal")
    return df

def history_fe_WIN_features_FREQ(feature_config, logger=default_logger):
    logger.info("- " * 25)
    logger.info("--> Start history_fe_WIN_FREQ_features function:")
    try:
        tic = time.time()
        fe_prefix = "fe_WIN_FREQ"
        features_folder_path = f"{root_path}/data/features/{fe_prefix}/"
        Path(features_folder_path).mkdir(parents=True, exist_ok=True)

        base_candle_folder_path = f"{root_path}/data/realtime_candle/"
        round_to = 4
        sampling_rate = 1 / 12

        for symbol in feature_config.keys():
            logger.info(f"---> Symbol: {symbol}")
            logger.info("= " * 40)

            base_cols = feature_config[symbol][fe_prefix]["base_columns"]
            raw_features = [f"M5_{base_col}" for base_col in base_cols]
            needed_columns = ["_time", "minutesPassed", "symbol"] + raw_features

            file_name = f"{base_candle_folder_path}{symbol}_realtime_candle.parquet"
            
            # Read with Polars
            df = pl.read_parquet(file_name, columns=needed_columns).sort("_time")

            # Ensure _time is datetime
            if not df["_time"].dtype.is_temporal():
                df = df.with_columns(pl.col("_time").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S"))

            logger.info("---> Entering the main func ...")
            df = add_win_fe_base_func(
                df,
                raw_features=raw_features,
                timeframes=feature_config[symbol][fe_prefix]["timeframe"],
                window_sizes=feature_config[symbol][fe_prefix]["window_size"],
                sampling_rate=sampling_rate,
                round_to=round_to,
                fe_prefix=fe_prefix,
            )
            logger.info("---> Exiting the main func ...")

            # Clean up and save
            df = df.drop(raw_features).with_columns(pl.lit(symbol).alias("symbol"))
            df.write_parquet(f"{features_folder_path}/{fe_prefix}_{symbol}.parquet")

        toc = time.time()
        logger.info(f"--> took {round(toc - tic, 2)} seconds to complete fe_WIN_FREQ.")
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