import pandas as pd
import numpy as np
from pathlib import Path
from dataset.logging_tools import default_logger
from dataset.configs.history_data_crawlers_config import root_path
import pywt  # Add wavelet transform support
import time
from statsmodels.tsa.stattools import adfuller
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Function for fractional differentiation
def frac_diff(series, d, window=10):
    weights = [1.0]
    for k in range(1, window):
        weight = -weights[-1] * (d - k + 1) / k
        weights.append(weight)
    weights = np.array(weights)
    output = np.convolve(series, weights[::-1], mode='valid')
    return pd.Series(output, index=series.index[len(weights)-1:len(weights)-1+len(output)])

# Function to find optimal d
def find_optimal_d(series, window=10, d_start=0.0, d_end=1.0, d_step=0.01, target_p=0.05, max_iter=100):
    best_d = d_start
    best_p = float('inf')
    best_diff = float('inf')
    for i, d in enumerate(np.arange(d_start, d_end + d_step, d_step)):
        if i >= max_iter:
            print(f"Max iterations reached. Best d={best_d}, p={best_p}, diff={best_diff}")
            break
        ffd_series = frac_diff(series, d, window=window)
        if len(ffd_series) < 2:
            print(f"Skipping d={d}: ffd_series length {len(ffd_series)} too short")
            continue
        adf_result = adfuller(ffd_series)
        p_value = adf_result[1]
        diff = abs(p_value - target_p)
        print(f"d={d:.2f}, p={p_value:.4f}, diff={diff:.4f}")  # Debug output
        if diff < best_diff:
            best_d = d
            best_p = p_value
            best_diff = diff
        if best_diff < 0.025:
            print(f"Found optimal d={best_d} with p={best_p} (diff={best_diff})")
            break
    return best_d, best_p, frac_diff(series, best_d, window=window)


# Function to process wavelet decomposition, thresholding, and reconstruction
def wavelet_denoise(signal, wavelet, level):
    # Remove .values since signal is already a NumPy array
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

    # Return as a pandas Series with the original index if needed, but here we return NumPy array
    return reconstructed_signal



def cal_window_max(array, window_size, sampling_rate, logger=default_logger):
    """
    Compute various features (FFT, Wavelet, Envelope, Cepstrum) for different windows.
    """
    num_features_fft = 10  # Number of features (one per top 10 FFT component)
    total_features = num_features_fft
    
    res = np.zeros([array.shape[0], total_features])  # Result array
    res[:window_size, :] = np.nan  # Fill initial rows with NaN

    flags = [True for _ in range(9)]
    array_shape = array.shape[0] // 10  # Use integer division for clarity

    for i in range(window_size, array.shape[0]):
        # Progress logging
        if (i >= array_shape) & flags[0]:
            logger.info("---> Did 10 perc of the job ...")
            flags[0] = False
        elif (i >= 2 * array_shape) & flags[1]:
            logger.info("---> Did 20 perc of the job ...")
            flags[1] = False
        elif (i >= 3 * array_shape) & flags[2]:
            logger.info("---> Did 30 perc of the job ...")
            flags[2] = False
        elif (i >= 4 * array_shape) & flags[3]:
            logger.info("---> Did 40 perc of the job ...")
            flags[3] = False
        elif (i >= 5 * array_shape) & flags[4]:
            logger.info("---> Did 50 perc of the job ...")
            flags[4] = False
        elif (i >= 6 * array_shape) & flags[5]:
            logger.info("---> Did 60 perc of the job ...")
            flags[5] = False
        elif (i >= 7 * array_shape) & flags[6]:
            logger.info("---> Did 70 perc of the job ...")
            flags[6] = False
        elif (i >= 8 * array_shape) & flags[7]:
            logger.info("---> Did 80 perc of the job ...")
            flags[7] = False
        elif (i >= 9 * array_shape) & flags[8]:
            logger.info("---> Did 90 perc of the job ...")
            flags[8] = False

        selected_slice = array[i - window_size + 1: i + 1]

        # 1 Wavelet Denoising
        if len(selected_slice) < 2:
            res[i, :] = np.nan  # Handle short slices
            continue
        reconstructed_level4 = wavelet_denoise(selected_slice, "bior4.4", level=4)
        
        # 2 FFD Calculation - Use a fixed window for frac_diff, not the full series length
        reconstructed_series = pd.Series(reconstructed_level4, index=range(len(reconstructed_level4)))
        optimal_d, optimal_p, FFD_slice = find_optimal_d(reconstructed_series, window=10)  # Fixed window=10
        
        # 3 FFD Centered
        FFD_centered = FFD_slice - FFD_slice.mean()

        # 4 Applying FFT - Add length check
        if len(FFD_centered) < 2:
            logger.debug(f"Skipping FFT at i={i}: FFD_centered length {len(FFD_centered)} too short")
            res[i, :] = np.nan
            continue

        fft_result = np.fft.fft(FFD_centered.values)  # FFD_slice is a Series, so .values is fine here
        n = len(FFD_centered)
        frequencies = np.fft.fftfreq(n, d=5/60)  # 5 minutes = 5/60 hours

        # Compute magnitude spectrum and phase
        magnitude_spectrum = np.abs(fft_result)
        phase_spectrum = np.angle(fft_result)

        # Only take the positive frequencies (first half of the FFT output)
        half_n = n // 2
        if half_n == 0:
            logger.debug(f"Skipping FFT at i={i}: half_n is 0 (n={n})")
            res[i, :] = np.nan
            continue

        positive_frequencies = frequencies[:half_n]
        positive_magnitudes = magnitude_spectrum[:half_n]
        positive_phases = phase_spectrum[:half_n]
        positive_frequencies[0] = 0  # DC component frequency

        # Find the 10 components with the highest amplitudes
        top_10_indices = np.argsort(positive_magnitudes)[-10:][::-1]  # Top 10 in descending order
        top_10_frequencies = positive_frequencies[top_10_indices]
        top_10_magnitudes = positive_magnitudes[top_10_indices]
        top_10_phases = positive_phases[top_10_indices]

        # Prepare the 3D data (top 10 components)
        data_3d = np.column_stack((top_10_frequencies, top_10_magnitudes, top_10_phases))  # Shape: (10, 3)

        # Standardize the data
        scaler = StandardScaler()
        data_3d_scaled = scaler.fit_transform(data_3d)  # Shape: (10, 3)

        # Apply PCA to reduce 3 columns to 1 column, keeping 10 rows
        pca = PCA(n_components=1)
        data_1d = pca.fit_transform(data_3d_scaled)  # Shape: (10, 1)

        # Assign the 10 PCA values to the result array
        res[i, :] = data_1d.flatten()  # Flatten (10, 1) to (10,) for assignment

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

            # Define feature column names based on updated `cal_window_max` function
            col_FREQ_ftr = [f"{fe_prefix}_FREQ_W{w_size}_M{tf}_Top{i+1}" for i in range(10)]
            array = df[raw_features].to_numpy()

            res = cal_window_max(array, w_size, sampling_rate)

            # Append the calculated results to the new_columns list as DataFrames
            new_columns.append(pd.DataFrame(res[:, 0:10].round(round_to), columns=col_FREQ_ftr, index=df.index))

    df = pd.concat([df] + new_columns, axis=1)

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
        sampling_rate = 1 / 12  # Assumed sampling rate in Hz; adjust if necessary

        for symbol in feature_config.keys():
            logger.info(f"---> Symbol: {symbol}")
            logger.info("= " * 40)

            base_cols = feature_config[symbol][fe_prefix]["base_columns"]
            raw_features = [f"M5_{base_col}" for base_col in base_cols]
            needed_columns = ["_time", "minutesPassed", "symbol"] + raw_features

            file_name = base_candle_folder_path + f"{symbol}_realtime_candle.parquet"
            
            

            # raw_features = df.columns[1]  # Get the name of the second column
            

            
            df = pd.read_parquet(file_name, columns=needed_columns).sort_values("_time")

            # Ensure `_time` column is a datetime type
            
            if not pd.api.types.is_datetime64_any_dtype(df["_time"]):
                df["_time"] = pd.to_datetime(df["_time"], format="%Y-%m-%d %H:%M:%S")

            # Add the window-based features
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

            # Clean up the DataFrame, dropping the raw features and adding symbol info
            df = df.drop(columns=[raw_features])

            df["symbol"] = symbol
            df.to_parquet(f"{features_folder_path}/{fe_prefix}_{symbol}.parquet", index=False)

        toc = time.time()
        logger.info(f"--> took {round(toc-tic, 2)} seconds to complete fe_WIN_FREQ.")
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
