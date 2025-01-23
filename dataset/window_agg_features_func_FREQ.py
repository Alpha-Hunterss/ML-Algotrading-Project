import pandas as pd
import numpy as np
from pathlib import Path
from dataset.logging_tools import default_logger
from dataset.configs.history_data_crawlers_config import root_path, symbols_dict
import pywt  # Add wavelet transform support
# from scipy.signal import hilbert
from scipy.stats import skew, kurtosis

def compute_fft(coeffs):
    "Computes the FFT of given cA and cD"
    fft_result = np.fft.fft(coeffs)
    frequencies = np.fft.fftfreq(len(coeffs))
    # Only take the positive half of the spectrum
    positive_freqs = frequencies[:len(frequencies)//2]
    positive_magnitude = np.abs(fft_result[:len(frequencies)//2])
    positive_magnitude[1:] = 2 * positive_magnitude[1:]
    positive_magnitude[0] = 0  # Remove the DC component by setting the first value to 0
    return positive_magnitude, positive_freqs


def cal_window_max(array, window_size, sampling_rate):
    """
    Compute various features (FFT, Wavelet, Envelope, Cepstrum) for different windows.
    """
    
    num_features_wavelet = 6
    
    total_features =  num_features_wavelet 
    
    res = np.zeros([array.shape[0], total_features])  # result array
    res[:window_size, :] = np.nan  # fill initial rows with NaN

    for i in range(window_size, array.shape[0]):
        selected_slice = array[i - window_size + 1: i + 1]
        
        
        # Compute Wavelet features

        cA, cD = pywt.dwt(selected_slice, 'bior4.4')
        cA_positive_magnitude, cA_positive_freqs = compute_fft(cA)
        cD_positive_magnitude, cD_positive_freqs = compute_fft(cD)

        sortinx_wavelet_cA = np.argsort(cA_positive_magnitude)[::-1][0] 
        sortinx_wavelet_cD = np.argsort(cD_positive_magnitude)[::-1][0] 
        #
        sortval_wavelet_cA = cA_positive_freqs[sortinx_wavelet_cA]
        sortval_wavelet_cD = cD_positive_freqs[sortinx_wavelet_cD]
        skew_wavelet_cA = skew(cA_positive_magnitude)
        skew_wavelet_cD = skew(cD_positive_magnitude)
        # kurt_wavelet_cA = kurtosis(cA_positive_magnitude, fisher=True)
        # kurt_wavelet_cD = kurtosis(cD_positive_magnitude, fisher=True)
        #
        coeffs = pywt.wavedec(selected_slice, "bior4.4", level=4)
        coeffs = [coeff if coeff.ndim == 1 else coeff.flatten() for coeff in coeffs]  # Ensure 1D arrays
        coeff_array, coeff_slices = pywt.coeffs_to_array(coeffs)
        threshold = np.std(coeff_array) * np.sqrt(2 * np.log(len(coeff_array)))
        coeff_array[np.abs(coeff_array) < threshold] = 0

        filtered_coeffs = pywt.array_to_coeffs(coeff_array, coeff_slices, output_format="wavedec")
        filtered_coeffs = [coeff.reshape(slc.shape) for coeff, slc in zip(filtered_coeffs, coeff_slices)]  # Reshape coefficients
        reconstructed_signal = pywt.waverec(filtered_coeffs, "bior4.4")

        Re_positive_magnitude, Re_positive_freqs = compute_fft(reconstructed_signal)

        sortinx_wavelet_Re = np.argsort(Re_positive_magnitude)[::-1][0] 
        #
        sortval_wavelet_Re = Re_positive_freqs[sortinx_wavelet_Re]
        skew_wavelet_Re = skew(Re_positive_magnitude)
        # kurt_wavelet_Re = kurtosis(Re_positive_magnitude, fisher=True)
        #
        res[i, 0] = sortval_wavelet_cA
        res[i, 1] = sortval_wavelet_cD
        res[i, 2] = sortval_wavelet_Re

        res[i, 3] = skew_wavelet_cA
        res[i, 4] = skew_wavelet_cD
        # res[i, 5] = kurt_wavelet_cA
        # res[i, 6] = kurt_wavelet_cD
        res[i, 5] = skew_wavelet_Re
        # res[i, 8] = kurt_wavelet_Re
        # Print progress every 1000 iterations
        if i % 5000 == 0:
            print(f"Wavelet done on slices: {i}")
        
    return res

def add_win_fe_base_func(
    df, symbol, raw_features, timeframes, window_sizes, sampling_rate, round_to=4, fe_prefix="fe_WIN_FREQ"
):
    new_columns = []

    for tf in timeframes:
        for w_size in window_sizes:
            assert tf == 5, "!!! For now, this code only works with 5M timeframe; tf must be 5."

            # Define feature column names based on updated `cal_window_max` function
            col_sortval_wavelet_cA = [f"{fe_prefix}_Val_cA_W{w_size}_M{tf}"]
            col_sortval_wavelet_cD = [f"{fe_prefix}_Val_cD_W{w_size}_M{tf}"]
            col_sortval_wavelet_Re = [f"{fe_prefix}_Val_Re_W{w_size}_M{tf}"]

            col_skew_wavelet_cA = [f"{fe_prefix}_skew_cA_W{w_size}_M{tf}"]
            col_skew_wavelet_cD = [f"{fe_prefix}_skew_cD_W{w_size}_M{tf}" ]
            
            # col_kurt_wavelet_cA = [f"{fe_prefix}_kurt_cA_W{w_size}_M{tf}"]
            # col_kurt_wavelet_cD = [f"{fe_prefix}_kurt_cD_W{w_size}_M{tf}"]

            
            col_skew_wavelet_Re = [f"{fe_prefix}_skew_Re_W{w_size}_M{tf}"]
            # col_kurt_wavelet_Re = [f"{fe_prefix}_kurt_Re_W{w_size}_M{tf}"]


            array = df[raw_features].to_numpy()

            res = cal_window_max(array, w_size, sampling_rate)

            # Append the calculated results to the new_columns list as DataFrames
            new_columns.append(pd.DataFrame(res[:, 0].round(round_to), columns=col_sortval_wavelet_cA, index=df.index))
            new_columns.append(pd.DataFrame(res[:, 1].round(round_to), columns=col_sortval_wavelet_cD, index=df.index))
            new_columns.append(pd.DataFrame(res[:, 2].round(round_to), columns=col_skew_wavelet_cA, index=df.index))
            new_columns.append(pd.DataFrame(res[:, 3].round(round_to), columns=col_skew_wavelet_cD, index=df.index))
            # new_columns.append(pd.DataFrame(res[:, 4].round(round_to), columns=col_kurt_wavelet_cA, index=df.index))
            # new_columns.append(pd.DataFrame(res[:, 5].round(round_to), columns=col_kurt_wavelet_cD, index=df.index))
            new_columns.append(pd.DataFrame(res[:, 4].round(round_to), columns=col_sortval_wavelet_Re, index=df.index))
            new_columns.append(pd.DataFrame(res[:, 5].round(round_to), columns=col_skew_wavelet_Re, index=df.index))
            # new_columns.append(pd.DataFrame(res[:, 8].round(round_to), columns=col_kurt_wavelet_Re, index=df.index))
            

    df = pd.concat([df] + new_columns, axis=1)

    return df

def history_fe_WIN_features_FREQ(feature_config, logger=default_logger):
    logger.info("- " * 25)
    logger.info("--> Start history_fe_WIN_FREQ_features function:")
    try:
        fe_prefix = "fe_WIN_FREQ"
        # features_folder_path = f"{root_path}/data/features/{fe_prefix}/"
        # Path(features_folder_path).mkdir(parents=True, exist_ok=True)
        # base_candle_folder_path = f"{root_path}/data/features/fe_FFD/" # address fe_FFD parquet

        features_folder_path = f"{root_path}/data/features/{fe_prefix}/"
        Path(features_folder_path).mkdir(parents=True, exist_ok=True)
        base_candle_folder_path = f"{root_path}/data/realtime_candle/"
        
        
        round_to = 5
        sampling_rate = 2  # Assumed sampling rate in Hz; adjust if necessary
        

        for symbol in feature_config.keys():
            logger.info(f"---> Symbol: {symbol}")
            logger.info("= " * 40)

            # base_cols = feature_config[symbol][fe_prefix]["base_columns"]
            # raw_features = [rf"fe_FFD-M5_{base_col}.*" for base_col in base_cols]
            # file_name = base_candle_folder_path + f"fe_FFD_{symbol}.parquet"

            base_cols = feature_config[symbol][fe_prefix]["base_columns"]
            raw_features = [f"M5_{base_col}" for base_col in base_cols]
            needed_columns = ["_time", "minutesPassed", "symbol"] + raw_features
            file_name = base_candle_folder_path + f"{symbol}_realtime_candle.parquet"
            
            # Read the data using Pandas
            # df = pd.read_parquet(file_name)
            # raw_features = df.columns[1]  # Get the name of the second column
            # needed_columns = ["_time", raw_features]

            df = pd.read_parquet(file_name, columns=needed_columns)
            df.sort_values("_time", inplace=True)

            df["_time"] = df["_time"].dt.tz_localize(None)
            df.drop(columns=["symbol"])
            df.sort_values("_time", inplace=True)
        

            # # Ensure `_time` column is a datetime type
            # if not pd.api.types.is_datetime64_any_dtype(df["_time"]):
            #     df["_time"] = pd.to_datetime(df["_time"], format="%Y-%m-%d %H:%M:%S")

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
            # df = df.drop(columns=[raw_features])

            df.drop(columns=raw_features + ["minutesPassed"], inplace=True)
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