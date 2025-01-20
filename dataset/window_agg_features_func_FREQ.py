import pandas as pd
import numpy as np
from pathlib import Path
from dataset.logging_tools import default_logger
from dataset.configs.history_data_crawlers_config import root_path
import pywt  # Add wavelet transform support
import time
from scipy.signal import hilbert


def hilbert_cupy(x):
    """
    Compute the analytic signal using the Hilbert transform, with CuPy arrays.

    Parameters
    ----------
    x : cupy.ndarray
        Signal data. Must be real.

    Returns
    -------
    xa : cupy.ndarray
        Analytic signal of the input signal, x
    """
    import cupy as cp

    if x.dtype not in [cp.float32, cp.float64]:
        x = x.astype(cp.float64)

    N = x.shape[-1]
    Xf = cp.fft.fft(x)

    # Create the Hilbert transformer
    h = cp.zeros(N)
    if N % 2 == 0:
        h[0] = h[N//2] = 1
        h[1:N//2] = 2
    else:
        h[0] = 1
        h[1:(N+1)//2] = 2

    return cp.fft.ifft(Xf * h)


def apply_hanning_window(array, window_size, use_cudf=False):
    """Apply a Hanning window to the data."""
    if use_cudf:
        import cupy as cp

    hanning_window = np.hanning(window_size) if not use_cudf else cp.hanning(window_size)

    return array.flatten() * hanning_window


def compute_fft_features(selected_slice, window_size, sampling_rate, use_cudf=False):
    """Compute FFT features."""
    if use_cudf:
        import cupy as cp

        fft_values = cp.fft.fft(selected_slice)
        fft_amplitude = cp.abs(fft_values[: window_size // 2])
        fft_amplitude[1:] = 2 * fft_amplitude[1:]
        frequencies = cp.fft.fftfreq(len(fft_amplitude), d = 1/sampling_rate)
        positive_frequencies = frequencies[: window_size // 2]
        fft_phase = cp.angle(fft_values)
    else:
        fft_values = np.fft.fft(selected_slice)
        fft_amplitude = np.abs(fft_values[: window_size // 2])
        fft_amplitude[1:] = 2 * fft_amplitude[1:]
        frequencies = np.fft.fftfreq(len(fft_amplitude), d = 1/sampling_rate)
        positive_frequencies = frequencies[: window_size // 2]
        fft_phase = np.angle(fft_values)

    return fft_amplitude, positive_frequencies, fft_phase


def compute_wavelet_features(selected_slice, window_size, sampling_rate, use_cudf=False):
    """Compute Wavelet features."""
    if use_cudf:
        selected_slice = selected_slice.get()

    coeffs = pywt.wavedec(selected_slice, "bior4.4", level=3)
    coeff_array, coeff_slices = pywt.coeffs_to_array(coeffs)
    threshold = np.std(coeff_array) * np.sqrt(2 * np.log(len(coeff_array)))
    coeff_array[np.abs(coeff_array) < threshold] = 0
    filtered_coeffs = pywt.array_to_coeffs(coeff_array, coeff_slices, output_format="wavedec")
    reconstructed_signal = pywt.waverec(filtered_coeffs, "bior4.4")

    if use_cudf:
        import cupy as cp

        reconstructed_signal = cp.asarray(reconstructed_signal)

    return compute_fft_features(reconstructed_signal, window_size, sampling_rate, use_cudf=use_cudf)


def compute_envelope_features(selected_slice, window_size, sampling_rate, use_cudf=False):
    """Compute Envelope features."""
    if use_cudf:
        import cupy as cp

        analytic_signal = hilbert_cupy(selected_slice)

        envelope = cp.abs(analytic_signal)
        instantaneous_phase = cp.unwrap(cp.angle(analytic_signal))
        instantaneous_frequency = cp.diff(instantaneous_phase) * sampling_rate / (2.0 * cp.pi)
        envelope_fft_values = cp.fft.fft(envelope)
        envelope_fft_amplitude = cp.abs(envelope_fft_values[:window_size // 2])
        envelope_fft_amplitude[1:] = 2 * envelope_fft_amplitude[1:]
        envelope_frequencies = cp.fft.fftfreq(len(envelope_fft_amplitude), d=1 / sampling_rate)
        positive_envelope_frequencies = envelope_frequencies[:window_size // 2]
        envelope_fft_phase = cp.angle(envelope_fft_values)
    else:
        analytic_signal = hilbert(selected_slice)

        envelope = np.abs(analytic_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = np.diff(instantaneous_phase) * sampling_rate / (2.0 * np.pi)
        envelope_fft_values = np.fft.fft(envelope)
        envelope_fft_amplitude = np.abs(envelope_fft_values[:window_size // 2])
        envelope_fft_amplitude[1:] = 2 * envelope_fft_amplitude[1:]
        envelope_frequencies = np.fft.fftfreq(len(envelope_fft_amplitude), d=1 / sampling_rate)
        positive_envelope_frequencies = envelope_frequencies[:window_size // 2]
        envelope_fft_phase = np.angle(envelope_fft_values)

    return envelope_fft_amplitude, positive_envelope_frequencies, envelope_fft_phase, instantaneous_frequency


def compute_cepstrum_features(selected_slice, use_cudf=False):
    """Compute Cepstrum features."""
    if use_cudf:
        import cupy as cp

        fft_values = cp.fft.fft(selected_slice)
        log_spectrum = cp.log(cp.abs(fft_values) + 1e-10)
        cepstrum = cp.fft.ifft(log_spectrum).real
    else:
        fft_values = np.fft.fft(selected_slice)
        log_spectrum = np.log(np.abs(fft_values) + 1e-10)
        cepstrum = np.fft.ifft(log_spectrum).real

    return cepstrum


def cal_window_max(array, window_size, sampling_rate, use_cudf=False, use_wavelet=True, logger=default_logger):
    """
    Compute various features (FFT, Wavelet, Envelope, Cepstrum) for different windows.
    """
    num_features_fft = 30
    num_features_wavelet = 30
    num_features_envelope = 40
    num_features_cepstrum = 20
    num_features_stats = 6
    total_features = num_features_fft + num_features_wavelet + num_features_envelope + num_features_cepstrum + num_features_stats

    if use_cudf:
        import cupy as cp

        res = cp.zeros([array.shape[0], total_features], dtype=cp.float32)  # result array
        res[:window_size, :] = cp.nan
    else:
        res = np.zeros([array.shape[0], total_features])  # result array
        res[:window_size, :] = np.nan  # fill initial rows with NaN

    flags = [True for i in range(9)]
    array_shape = array.shape[0]/10

    for i in range(window_size, array.shape[0]):
        if (i >= array_shape) & flags[0]:
            logger.info("---> Did 10 perc of the job ...")
            flags[0] = False
        elif (i >= 2*array_shape) & flags[1]:
            logger.info("---> Did 20 perc of the job ...")
            flags[1] = False
        elif (i >= 3*array_shape) & flags[2]:
            logger.info("---> Did 30 perc of the job ...")
            flags[2] = False
        elif (i >= 4*array_shape) & flags[3]:
            logger.info("---> Did 40 perc of the job ...")
            flags[3] = False
        elif (i >= 5*array_shape) & flags[4]:
            logger.info("---> Did 50 perc of the job ...")
            flags[4] = False
        elif (i >= 6*array_shape) & flags[5]:
            logger.info("---> Did 60 perc of the job ...")
            flags[5] = False
        elif (i >= 7*array_shape) & flags[6]:
            logger.info("---> Did 70 perc of the job ...")
            flags[6] = False
        elif (i >= 8*array_shape) & flags[7]:
            logger.info("---> Did 80 perc of the job ...")
            flags[7] = False
        elif (i >= 9*array_shape) & flags[8]:
            logger.info("---> Did 90 perc of the job ...")
            flags[8] = False

        selected_slice = apply_hanning_window(
            array[i - window_size + 1: i + 1], window_size, use_cudf=use_cudf
        )

        # Compute FFT features
        fft_amplitude, positive_frequencies, fft_phase = compute_fft_features(
            selected_slice, window_size, sampling_rate, use_cudf=use_cudf
        )

        if use_cudf:
            sorted_indices_fft = cp.argsort(fft_amplitude[1:])[::-1][:10] + 1
        else:
            sorted_indices_fft = np.argsort(fft_amplitude[1:])[::-1][:10] + 1

        res[i, 0:10] = fft_amplitude[sorted_indices_fft]
        res[i, 10:20] = positive_frequencies[sorted_indices_fft]
        res[i, 20:30] = fft_phase[sorted_indices_fft]

        # Compute Wavelet features
        if use_wavelet:
            fft_amplitude_wavelet, positive_frequencies_wavelet, fft_phase_wavelet = compute_wavelet_features(
                selected_slice, window_size, sampling_rate, use_cudf=use_cudf
            )

            if use_cudf:
                sorted_indices_wavelet_fft = cp.argsort(fft_amplitude_wavelet[1:])[::-1][:10] + 1
            else:
                sorted_indices_wavelet_fft = np.argsort(fft_amplitude_wavelet[1:])[::-1][:10] + 1

            res[i, 30:40] = fft_amplitude_wavelet[sorted_indices_wavelet_fft]
            res[i, 40:50] = positive_frequencies_wavelet[sorted_indices_wavelet_fft]
            res[i, 50:60] = fft_phase_wavelet[sorted_indices_wavelet_fft]

        # Compute Envelope features
        envelope_fft_amplitude, positive_envelope_frequencies, envelope_fft_phase, instantaneous_frequency = compute_envelope_features(
            selected_slice, window_size, sampling_rate, use_cudf=use_cudf
        )

        if use_cudf:
            sorted_indices_envelope = cp.argsort(envelope_fft_amplitude[1:])[::-1][:10] + 1
            sorted_indices_if = cp.argsort(cp.abs(instantaneous_frequency))[::-1][:10]
        else:
            sorted_indices_envelope = np.argsort(envelope_fft_amplitude[1:])[::-1][:10] + 1
            sorted_indices_if = np.argsort(np.abs(instantaneous_frequency))[::-1][:10]

        res[i, 60:70] = envelope_fft_amplitude[sorted_indices_envelope]
        res[i, 70:80] = positive_envelope_frequencies[sorted_indices_envelope]
        res[i, 80:90] = envelope_fft_phase[sorted_indices_envelope]
        res[i, 90:100] = instantaneous_frequency[sorted_indices_if]

        # Compute Cepstrum features
        cepstrum = compute_cepstrum_features(selected_slice, use_cudf=use_cudf)

        if use_cudf:
            sorted_indices_cepstrum = cp.argsort(cp.abs(cepstrum[1:]))[::-1][:10] + 1
        else:
            sorted_indices_cepstrum = np.argsort(np.abs(cepstrum[1:]))[::-1][:10] + 1

        res[i, 100:110] = cepstrum[sorted_indices_cepstrum]
        res[i, 110:120] = sorted_indices_cepstrum / sampling_rate

        if use_cudf:
            # Compute statistical features
            res[i, 120] = cp.mean(positive_frequencies[sorted_indices_fft])
            res[i, 121] = cp.std(positive_frequencies[sorted_indices_fft])
            if use_wavelet:
                res[i, 122] = cp.mean(positive_frequencies_wavelet[sorted_indices_wavelet_fft])
                res[i, 123] = cp.std(positive_frequencies_wavelet[sorted_indices_wavelet_fft])
            res[i, 124] = cp.mean(positive_envelope_frequencies[sorted_indices_envelope])
            res[i, 125] = cp.std(positive_envelope_frequencies[sorted_indices_envelope])
        else:
            # Compute statistical features
            res[i, 120] = np.mean(positive_frequencies[sorted_indices_fft])
            res[i, 121] = np.std(positive_frequencies[sorted_indices_fft])
            if use_wavelet:
                res[i, 122] = np.mean(positive_frequencies_wavelet[sorted_indices_wavelet_fft])
                res[i, 123] = np.std(positive_frequencies_wavelet[sorted_indices_wavelet_fft])
            res[i, 124] = np.mean(positive_envelope_frequencies[sorted_indices_envelope])
            res[i, 125] = np.std(positive_envelope_frequencies[sorted_indices_envelope])

    return res


def add_win_fe_base_func(
    df, raw_features, timeframes, window_sizes,
    sampling_rate, round_to=4, fe_prefix="fe_WIN_FREQ",
    use_cudf=False, use_wavelet=True, logger=default_logger,
):
    new_columns = []

    for tf in timeframes:
        for w_size in window_sizes:
            logger.info(f"---> Doing window size {w_size} ...")
            assert tf == 5, "!!! For now, this code only works with 5M timeframe; tf must be 5."

            # Define feature column names based on updated `cal_window_max` function
            col_fft_magnitudes = [f"{fe_prefix}_fft_mag_W{w_size}_M{tf}_Top{i+1}" for i in range(10)]
            col_fft_frequencies = [f"{fe_prefix}_fft_freq_W{w_size}_M{tf}_Top{i+1}" for i in range(10)]
            col_fft_phases = [f"{fe_prefix}_fft_phase_W{w_size}_M{tf}_Top{i+1}" for i in range(10)]

            col_wavelet_magnitudes = [f"{fe_prefix}_wavelet_mag_W{w_size}_M{tf}_Top{i+1}" for i in range(10)]
            col_wavelet_frequencies = [f"{fe_prefix}_wavelet_freq_W{w_size}_M{tf}_Top{i+1}" for i in range(10)]
            col_wavelet_phases = [f"{fe_prefix}_wavelet_phase_W{w_size}_M{tf}_Top{i+1}" for i in range(10)]

            col_envelope_magnitudes = [f"{fe_prefix}_env_mag_W{w_size}_M{tf}_Top{i+1}" for i in range(10)]
            col_envelope_frequencies = [f"{fe_prefix}_env_freq_W{w_size}_M{tf}_Top{i+1}" for i in range(10)]
            col_envelope_phases = [f"{fe_prefix}_env_phase_W{w_size}_M{tf}_Top{i+1}" for i in range(10)]
            col_instant_freq = [f"{fe_prefix}_inst_freq_W{w_size}_M{tf}_Top{i+1}" for i in range(10)]

            col_cepstrum_amplitudes = [f"{fe_prefix}_cepstrum_amp_W{w_size}_M{tf}_Top{i+1}" for i in range(10)]
            col_cepstrum_quefrencies = [f"{fe_prefix}_cepstrum_quef_W{w_size}_M{tf}_Top{i+1}" for i in range(10)]

            col_stats = [
                f"{fe_prefix}_fft_freq_mean_W{w_size}_M{tf}",
                f"{fe_prefix}_fft_freq_std_W{w_size}_M{tf}",
                f"{fe_prefix}_wavelet_freq_mean_W{w_size}_M{tf}",
                f"{fe_prefix}_wavelet_freq_std_W{w_size}_M{tf}",
                f"{fe_prefix}_env_freq_mean_W{w_size}_M{tf}",
                f"{fe_prefix}_env_freq_std_W{w_size}_M{tf}",
            ]

            if use_cudf:
                import cupy as cp

                array = cp.asarray(df[raw_features].values)
            else:
                array = df[raw_features].to_numpy()

            res = cal_window_max(array, w_size, sampling_rate, use_cudf=use_cudf, use_wavelet=use_wavelet)

            if use_cudf:
                import cudf

                # Append the calculated results to the new_columns list as DataFrames
                new_columns.append(cudf.DataFrame(res[:, 0:10].round(round_to), columns=col_fft_magnitudes, index=df.index))
                new_columns.append(cudf.DataFrame(res[:, 10:20].round(round_to), columns=col_fft_frequencies, index=df.index))
                new_columns.append(cudf.DataFrame(res[:, 20:30].round(round_to), columns=col_fft_phases, index=df.index))

                new_columns.append(cudf.DataFrame(res[:, 30:40].round(round_to), columns=col_wavelet_magnitudes, index=df.index))
                new_columns.append(cudf.DataFrame(res[:, 40:50].round(round_to), columns=col_wavelet_frequencies, index=df.index))
                new_columns.append(cudf.DataFrame(res[:, 50:60].round(round_to), columns=col_wavelet_phases, index=df.index))

                new_columns.append(cudf.DataFrame(res[:, 60:70].round(round_to), columns=col_envelope_magnitudes, index=df.index))
                new_columns.append(cudf.DataFrame(res[:, 70:80].round(round_to), columns=col_envelope_frequencies, index=df.index))
                new_columns.append(cudf.DataFrame(res[:, 80:90].round(round_to), columns=col_envelope_phases, index=df.index))
                new_columns.append(cudf.DataFrame(res[:, 90:100].round(round_to), columns=col_instant_freq, index=df.index))

                new_columns.append(cudf.DataFrame(res[:, 100:110].round(round_to), columns=col_cepstrum_amplitudes, index=df.index))
                new_columns.append(cudf.DataFrame(res[:, 110:120].round(round_to), columns=col_cepstrum_quefrencies, index=df.index))

                new_columns.append(cudf.DataFrame(res[:, 120:126].round(round_to), columns=col_stats, index=df.index))
            else:
                # Append the calculated results to the new_columns list as DataFrames
                new_columns.append(pd.DataFrame(res[:, 0:10].round(round_to), columns=col_fft_magnitudes, index=df.index))
                new_columns.append(pd.DataFrame(res[:, 10:20].round(round_to), columns=col_fft_frequencies, index=df.index))
                new_columns.append(pd.DataFrame(res[:, 20:30].round(round_to), columns=col_fft_phases, index=df.index))

                new_columns.append(pd.DataFrame(res[:, 30:40].round(round_to), columns=col_wavelet_magnitudes, index=df.index))
                new_columns.append(pd.DataFrame(res[:, 40:50].round(round_to), columns=col_wavelet_frequencies, index=df.index))
                new_columns.append(pd.DataFrame(res[:, 50:60].round(round_to), columns=col_wavelet_phases, index=df.index))

                new_columns.append(pd.DataFrame(res[:, 60:70].round(round_to), columns=col_envelope_magnitudes, index=df.index))
                new_columns.append(pd.DataFrame(res[:, 70:80].round(round_to), columns=col_envelope_frequencies, index=df.index))
                new_columns.append(pd.DataFrame(res[:, 80:90].round(round_to), columns=col_envelope_phases, index=df.index))
                new_columns.append(pd.DataFrame(res[:, 90:100].round(round_to), columns=col_instant_freq, index=df.index))

                new_columns.append(pd.DataFrame(res[:, 100:110].round(round_to), columns=col_cepstrum_amplitudes, index=df.index))
                new_columns.append(pd.DataFrame(res[:, 110:120].round(round_to), columns=col_cepstrum_quefrencies, index=df.index))

                new_columns.append(pd.DataFrame(res[:, 120:126].round(round_to), columns=col_stats, index=df.index))

    if use_cudf:
        df = cudf.concat([df] + new_columns, axis=1)
    else:
        df = pd.concat([df] + new_columns, axis=1)

    return df


def history_fe_WIN_features_FREQ(feature_config, use_cudf=False, use_wavelet=True, logger=default_logger):
    logger.info("- " * 25)
    logger.info("--> Start history_fe_WIN_FREQ_features function:")
    try:
        tic = time.time()
        fe_prefix = "fe_WIN_FREQ"
        features_folder_path = f"{root_path}/data/features/{fe_prefix}/"
        Path(features_folder_path).mkdir(parents=True, exist_ok=True)

        base_candle_folder_path = f"{root_path}/data/features/fe_FFD/" # addres fe_FFD parquet
        round_to = 6
        sampling_rate = 1 / 300  # Assumed sampling rate in Hz; adjust if necessary

        for symbol in feature_config.keys():
            logger.info(f"---> Symbol: {symbol}")
            logger.info("= " * 40)

            # base_cols = feature_config[symbol][fe_prefix]["base_columns"]
            # raw_features = [rf"fe_FFD-M5_{base_col}.*" for base_col in base_cols]

            file_name = base_candle_folder_path + f"fe_FFD_{symbol}.parquet"

            # Read the data using Pandas (or CuDF)
            if use_cudf:
                import cudf

                df = cudf.read_parquet(file_name)
            else:
                df = pd.read_parquet(file_name)

            raw_features = df.columns[1]  # Get the name of the second column
            needed_columns = ["_time", raw_features]

            if use_cudf:
                df = cudf.read_parquet(file_name, columns=needed_columns).sort_values("_time")
            else:
                df = pd.read_parquet(file_name, columns=needed_columns).sort_values("_time")

            # Ensure `_time` column is a datetime type
            if use_cudf:
                if not df["_time"].dtype.name.startswith("datetime64"):
                    df["_time"] = cudf.to_datetime(df["_time"], format="%Y-%m-%d %H:%M:%S")
            else:
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
                use_cudf=use_cudf,
                use_wavelet=use_wavelet,
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
