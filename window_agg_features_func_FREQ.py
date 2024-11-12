from pathlib import Path
from dataset.configs.history_data_crawlers_config import root_path, symbols_dict
import pandas as pd
import numpy as np
from dataset.logging_tools import default_logger
from scipy.stats import skew, kurtosis
from scipy.signal import cwt, ricker, stft
from scipy.fftpack import fft

def cal_window_max(array, window_size):
    res = np.zeros([array.shape[0], 4])
    res[:window_size, :] = np.nan
    # Add logic or return res if needed

def calculate_log_returns(array):
    return np.log(array[1:] / array[:-1])

def cal_frequency_features(array, window_size, method="stft"):
    res = np.full([array.shape[0], 8], np.nan)  # Updated shape
    
    log_returns = calculate_log_returns(array[:, 0]) 
    log_returns = np.insert(log_returns, 0, 0)

    for i in range(window_size, array.shape[0]):
        selected_slice = log_returns[i - window_size + 1 : i + 1]

        if method == "stft":
            f, t, Zxx = stft(selected_slice, nperseg=window_size)
            power_spectrum = np.abs(Zxx) ** 2
            
            res[i, :] = [
                np.max(selected_slice),
                np.min(selected_slice),
                np.mean(selected_slice),
                np.std(selected_slice),
                skew(selected_slice),  # Corrected skewness calculation
                kurtosis(selected_slice),  # Corrected kurtosis calculation
                np.sqrt(np.mean(selected_slice**2)),
                np.var(selected_slice)
            ]

        elif method == "fft":
            fft_result = fft(selected_slice)
            power_spectrum = np.abs(fft_result) ** 2
            
            res[i, :] = [
                np.max(selected_slice),
                np.min(selected_slice),
                np.mean(selected_slice),
                np.std(selected_slice),
                skew(selected_slice),
                kurtosis(selected_slice),
                np.sqrt(np.mean(selected_slice**2)),
                np.var(selected_slice)
            ]

        elif method == "wavelet":
            scales = range(1, 6)
            coeffs = cwt(selected_slice, ricker, scales)
            
            res_wavelet = []
            for coeff in coeffs:
                res_wavelet.extend([
                    np.max(coeff),
                    np.min(coeff),
                    np.mean(coeff),
                    np.std(coeff),
                    skew(coeff),
                    kurtosis(coeff),
                    np.sqrt(np.mean(coeff**2)),
                    np.var(coeff)
                ])
            res[i, :] = res_wavelet[:8]  # Match res shape for each scale
            
    return res

def add_win_fe_base_func_FREQ(df, symbol, raw_features, timeframes, window_sizes, round_to=5, fe_prefix="fe_WIN_FREQ"):
    for tf in timeframes:
        for w_size in window_sizes:
            assert tf == 5, "Only works with 5M timeframe."

            cols = {}
            metrics = ['max', 'min', 'mean', 'std', 'skew', 'kurtosis', 'rms', 'var']
            methods = ['stft', 'fft', 'wavelet']

            for method in methods:
                cols[method] = {metric: f"{fe_prefix}_{method}_{metric}_W{w_size}_M{tf}" for metric in metrics}

            array = df[raw_features].to_numpy()
            
            stft_res = cal_frequency_features(array, w_size, method="stft")
            fft_res = cal_frequency_features(array, w_size, method="fft")
            wavelet_res = cal_frequency_features(array, w_size, method="wavelet")

            for method, res in zip(['stft', 'fft', 'wavelet'], [stft_res, fft_res, wavelet_res]):
                for idx, metric in enumerate(metrics):
                    df[cols[method][metric]] = np.round(res[:, idx], round_to)

    return df

def history_fe_WIN_features_FREQ(feature_config, logger=default_logger):
    logger.info("- " * 25)
    logger.info("--> start history_fe_WIN_FREQ sfeatures func:")
    try:
        fe_prefix = "fe_WIN_FREQ"
        features_folder_path = f"{root_path}/data/features/{fe_prefix}/"
        Path(features_folder_path).mkdir(parents=True, exist_ok=True)
        base_candle_folder_path = f"{root_path}/data/realtime_candle/"
        round_to = 4

        for symbol in list(feature_config.keys()):
            logger.info(f"---> symbol: {symbol}")
            logger.info("= " * 40)

            base_cols = feature_config[symbol][fe_prefix]["base_columns"]
            raw_features = [f"M5_{base_col}" for base_col in base_cols]
            needed_columns = ["_time", "minutesPassed", "symbol"] + raw_features
            file_name = base_candle_folder_path + f"{symbol}_realtime_candle.parquet"
            df = pd.read_parquet(file_name, columns=needed_columns)
            df.sort_values("_time", inplace=True)
            df["_time"] = df["_time"].dt.tz_localize(None)
            df.drop(columns=["symbol"], inplace=True)
            
            df = add_win_fe_base_func_FREQ(df, symbol, raw_features, feature_config[symbol][fe_prefix]["timeframe"], feature_config[symbol][fe_prefix]["window_size"], round_to)

            df.drop(columns=raw_features + ["minutesPassed"], inplace=True, errors='ignore')
            df["symbol"] = symbol
            df.to_parquet(f"{features_folder_path}/{fe_prefix}_{symbol}.parquet")
        
        logger.info("--> history_fe_WIN_features_FREQ run successfully.")
    except Exception as e:
        logger.exception("--> history_fe_WIN_features_FREQ error.")
        raise ValueError("!!!")

if __name__ == "__main__":
    from configs.feature_configs_general import generate_general_config
    config_general = generate_general_config()
    history_fe_WIN_features_FREQ(config_general)
    default_logger.info(f"--> history_fe_WIN_features_FREQ DONE.")
