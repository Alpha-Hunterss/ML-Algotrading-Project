import cudf
import cupy as cp
import numpy as np
from pathlib import Path
from dataset.logging_tools import default_logger
from dataset.configs.history_data_crawlers_config import root_path
import time
from statsmodels.tsa.stattools import adfuller  # CPU-bound
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# GPU-optimized fractional differentiation
def frac_diff_gpu(series, d, window_length):
    weights = cp.ones(1)
    for k in range(1, window_length):
        weight = -weights[-1] * (d - k + 1) / k
        weights = cp.append(weights, weight)
    output = cp.convolve(series, weights[::-1], mode='valid')
    return output

# Simplified find_optimal_d (fewer iterations)
def find_optimal_d(series, window_length, d_start=0.0, d_end=1.0, d_step=0.05, target_p=0.05):
    series_np = cp.asnumpy(series)  # Convert to NumPy for adfuller
    best_d = d_start
    best_p = float('inf')
    best_diff = float('inf')
    for d in cp.arange(d_start, d_end + d_step, d_step):
        ffd_series = cp.asnumpy(frac_diff_gpu(series, d, window_length))
        if len(ffd_series) < 2:
            continue
        adf_result = adfuller(ffd_series)  # CPU-bound
        p_value = adf_result[1]
        diff = abs(p_value - target_p)
        if diff < best_diff:
            best_d = float(d)
            best_p = p_value
            best_diff = diff
        if best_diff < 0.025:
            break
    return best_d, best_p, frac_diff_gpu(series, best_d, window_length)

# GPU-optimized wavelet denoising (FFT approximation)
def wavelet_denoise_gpu(signal, level=4):
    fft_signal = cp.fft.fft(signal)
    threshold = cp.std(fft_signal) * cp.sqrt(2 * cp.log(len(fft_signal)))
    fft_signal[cp.abs(fft_signal) < threshold] = 0
    return cp.fft.ifft(fft_signal).real

# GPU-optimized window processing
def process_window_gpu(windows, window_size, sampling_rate):
    num_features = 10
    res = cp.zeros((windows.shape[0], num_features))

    # 1. Wavelet Denoising (batch process)
    denoised = cp.zeros_like(windows)
    for i in range(windows.shape[0]):
        denoised[i] = wavelet_denoise_gpu(windows[i])

    # 2. FFD Calculation (batch process)
    ffd_slices = cp.zeros((windows.shape[0], window_size - 4))  # Adjust for valid convolution
    for i in range(len(denoised)):
        d, _, ffd = find_optimal_d(denoised[i], window_length=5)
        ffd_slices[i] = ffd

    # 3. FFD Centered
    ffd_centered = ffd_slices - cp.mean(ffd_slices, axis=1, keepdims=True)

    # 4. FFT (batch process)
    fft_results = cp.fft.fft(ffd_centered, axis=1)
    n = ffd_centered.shape[1]
    frequencies = cp.fft.fftfreq(n, d=5/60)
    magnitude_spectrum = cp.abs(fft_results)
    phase_spectrum = cp.angle(fft_results)

    half_n = n // 2
    positive_frequencies = frequencies[:half_n]
    positive_magnitudes = magnitude_spectrum[:, :half_n]
    positive_phases = phase_spectrum[:, :half_n]
    positive_frequencies[0] = 0

    # Top 10 components (batch process)
    top_10_indices = cp.argsort(positive_magnitudes, axis=1)[:, -10:][:, ::-1]
    batch_size = windows.shape[0]
    top_10_frequencies = cp.zeros((batch_size, 10))
    top_10_magnitudes = cp.zeros((batch_size, 10))
    top_10_phases = cp.zeros((batch_size, 10))

    for i in range(batch_size):
        top_10_frequencies[i] = positive_frequencies[top_10_indices[i]]
        top_10_magnitudes[i] = positive_magnitudes[i, top_10_indices[i]]
        top_10_phases[i] = positive_phases[i, top_10_indices[i]]

    # 5. PCA (CPU fallback)
    data_3d = cp.stack((top_10_frequencies, top_10_magnitudes, top_10_phases), axis=2)
    data_3d_np = cp.asnumpy(data_3d.reshape(-1, 3))
    scaler = StandardScaler()
    data_3d_scaled = scaler.fit_transform(data_3d_np)
    pca = PCA(n_components=1)
    data_1d = pca.fit_transform(data_3d_scaled).reshape(batch_size, 10)
    
    return cp.asarray(data_1d)

def cal_window_max(array, window_size, sampling_rate, logger=default_logger):
    num_features = 10
    res = cp.zeros((array.shape[0], num_features))
    res[:window_size, :] = cp.nan

    # Aggregate multiple features into a single series (e.g., mean across columns)
    if array.ndim > 1:
        array = cp.mean(array, axis=1)  # Reduce to 1D if multiple features

    # Manually create sliding windows on GPU
    n_windows = array.shape[0] - window_size + 1
    windows = cp.zeros((n_windows, window_size))
    for i in range(n_windows):
        windows[i] = array[i:i + window_size]

    batch_size = 1000  # Adjust based on GPU memory
    for start in range(0, n_windows, batch_size):
        end = min(start + batch_size, n_windows)
        batch_windows = windows[start:end]
        batch_res = process_window_gpu(batch_windows, window_size, sampling_rate)
        res[start + window_size - 1:start + window_size - 1 + batch_res.shape[0]] = batch_res

    # Simplified logging
    for perc in range(10, 100, 10):
        if perc * array.shape[0] // 100 < array.shape[0]:
            logger.info(f"---> Did {perc} perc of the job ...")

    return cp.asnumpy(res)

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
            array = df[raw_features].to_cupy()

            res = cal_window_max(array, w_size, sampling_rate)

            # Convert to cuDF DataFrame
            new_columns.append(cudf.DataFrame(res[:, 0:10].round(round_to), columns=col_FREQ_ftr))

    df = cudf.concat([df] + new_columns, axis=1)
    return df

def history_fe_WIN_features_FREQ(feature_config, logger=default_logger):
    logger.info("- " * 25)
    logger.info("--> Start history_fe_WIN_freq_features function:")
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
            
            # Read with cuDF
            df = cudf.read_parquet(file_name, columns=needed_columns).sort_values("_time")

            # Ensure _time is datetime
            df['_time'] = df['_time'].astype('datetime64[ns]')

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
            df = df.drop(raw_features)
            df["symbol"] = symbol
            df.to_parquet(f"{features_folder_path}/{fe_prefix}_{symbol}.parquet")

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