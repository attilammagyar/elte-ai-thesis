# challenge='distortion-sqrt-unspec', generator='gemini-2.5-pro-preview-05-06', temperature=0.01
# --- BEGIN GENERATED CODE ---

import numpy as np
from scipy import signal

def distort(samples, gain_db=12.0, sample_rate=44100.0, channels=2): # channels parameter from signature
    # Initialize the output array as specified in the problem template
    distorted = np.zeros_like(samples)

    # Internal representation for processing, ensuring it's 2D (N, C)
    # The problem states samples is (N, channels), implying samples.ndim == 2.
    # This logic handles if samples might be (N,) for mono, ensuring _samples_internal is 2D.
    if samples.ndim == 1:
        # This implies mono. samples.shape[0] is N.
        # Reshape to (N, 1) for consistent processing.
        _samples_internal = samples.reshape(-1, 1)
    else:
        _samples_internal = samples
    
    # --- Configuration for Anti-Aliasing ---
    # Oversampling factor. 4x is a common choice for a good balance of
    # aliasing reduction and computational cost for this type of non-linearity.
    OS_FACTOR = 4 
    # Order of the Butterworth low-pass filter.
    # 8th order provides a reasonably steep rolloff to remove frequencies above original Nyquist.
    FILTER_ORDER = 8 

    # --- Step 1: Apply Gain ---
    # Convert dB gain to linear gain
    linear_gain = 10**(gain_db / 20.0)
    signal_with_gain = _samples_internal * linear_gain

    num_original_samples = signal_with_gain.shape[0]

    # If no samples, return the initialized empty/zeros array
    if num_original_samples == 0:
        return distorted # This is np.zeros_like(samples)

    # --- Anti-aliasing process using oversampling ---
    # OS_FACTOR is set > 1 to perform oversampling-based anti-aliasing.
    
    # --- Step 2: Oversample ---
    # Calculate the number of samples for the oversampled signal.
    num_oversampled_samples = num_original_samples * OS_FACTOR
    
    # Resample along the time axis (axis 0).
    # signal.resample uses FFT-based resampling.
    # It handles multi-channel (N,C) arrays correctly if axis=0 is specified.
    oversampled_signal = signal.resample(signal_with_gain, num_oversampled_samples, axis=0)
    
    # --- Step 3: Apply Non-linear Distortion ---
    # Distortion function: f(x) = x / sqrt(1 + x^2)
    # This function is numerically stable for real x, as the denominator (sqrt(1 + x^2)) is always >= 1.
    # np.square(x) computes x^2 element-wise.
    distorted_oversampled_signal = oversampled_signal / np.sqrt(1.0 + np.square(oversampled_signal))

    # --- Step 4: Low-pass Filtering (Anti-aliasing filter) ---
    # Design a Butterworth low-pass filter.
    # The cutoff frequency for this filter should be the Nyquist frequency of the original signal
    # (i.e., sample_rate / 2).
    # For scipy.signal.butter, the normalized cutoff frequency Wn is relative to
    # the Nyquist frequency of the signal being filtered (which is the oversampled signal).
    # Oversampled Nyquist = (sample_rate * OS_FACTOR) / 2.
    # So, Wn = (original_nyquist) / (oversampled_nyquist)
    #         = (sample_rate / 2) / ((sample_rate * OS_FACTOR) / 2)
    #         = 1.0 / OS_FACTOR
    
    # This Wn is valid for signal.butter as long as OS_FACTOR > 1 (making Wn < 1.0).
    # E.g., if OS_FACTOR = 4, Wn = 0.25.
    normalized_cutoff = 1.0 / OS_FACTOR
    
    # Create filter coefficients.
    b, a = signal.butter(FILTER_ORDER, normalized_cutoff, btype='low', analog=False)
    
    # Apply the filter using filtfilt for zero-phase response.
    # filtfilt operates along axis 0 (time axis).
    # It includes padding mechanisms to handle signal edges and is generally robust for typical signal lengths.
    filtered_distorted_oversampled_signal = signal.filtfilt(b, a, distorted_oversampled_signal, axis=0)

    # --- Step 5: Downsample ---
    # Resample back to the original number of samples.
    # This also uses FFT-based resampling.
    processed_signal_final_shape = signal.resample(filtered_distorted_oversampled_signal, num_original_samples, axis=0)
    
    # --- Finalize output ---
    # Ensure float32 type as per problem (input samples are float32).
    # Intermediate scipy operations might use float64 for precision.
    processed_signal_final_shape = processed_signal_final_shape.astype(np.float32)

    # Store the result in the 'distorted' array, matching original 'samples' shape.
    if samples.ndim == 1:
        # If original 'samples' was 1D (e.g., (N,)), result should be 1D.
        # At this point, processed_signal_final_shape is (N, 1) due to _samples_internal.
        # .ravel() converts it to (N,).
        distorted[:] = processed_signal_final_shape.ravel()
    else:
        # If original 'samples' was 2D (N, C), result is (N, C).
        distorted[:] = processed_signal_final_shape
    
    return distorted


# --- END GENERATED CODE ---



import numpy as np

def run_test():
    import json
    import time
    import wave

    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.signal as spsg

    def linear_to_db(linear):
        return 20.0 * np.log10(np.clip(linear, 1e-6, 32.0))

    def db_to_linear(db):
        return 10.0 ** (db / 20.0)

    def delta_cents(freq_1, freq_2):
        return 1200.0 * np.log2(freq_2 / freq_1)

    def find_noise(data, fundamental_freq, sample_rate, prominence=2.0, freq_same_threshold_cents=50.0):
        nyquist = sample_rate / 2
        harmonics = fundamental_freq * np.array(
            list(range(1, int(nyquist / fundamental_freq) + 1))
        )

        fft = np.fft.fft(data)
        freqs = np.fft.fftfreq(len(data), d=1.0 / sample_rate)

        crop = np.argmax(freqs < 0.0)
        freqs = freqs[0:crop]
        fft = fft[0:crop]

        fft = np.abs(fft) / len(data)
        fft_db = linear_to_db(fft)

        peaks = spsg.find_peaks(fft_db, prominence=prominence)
        signal_max = {}
        noise_max = {}

        for peak in peaks[0]:
            peak_freq = freqs[peak]
            closest_harmonic = harmonics[np.argmin(np.abs(harmonics - peak_freq))]
            delta = delta_cents(closest_harmonic, peak_freq)

            if np.abs(delta) > freq_same_threshold_cents:
                closest_noise_freq = None
                closest_distance = 999999.0

                for noise_freq in noise_max.keys():
                    distance = 999999.0 if closest_noise_freq is None else delta_cents(noise_freq, peak_freq)

                    if closest_noise_freq is None or distance < closest_distance:
                        closest_distance = distance
                        closest_noise_freq = noise_freq

                if closest_distance > freq_same_threshold_cents:
                    noise_max[peak_freq] = fft[peak]
                elif fft[peak] > noise_max[closest_noise_freq]:
                    noise_max[closest_noise_freq] += fft[peak]

            else:
                signal_max.setdefault(closest_harmonic, 0.0)

                if fft[peak] > signal_max[closest_harmonic]:
                    signal_max[closest_harmonic] += fft[peak]

        # plot_fft(freqs, fft_db, signal_max, noise_max)

        signal = sum(v for v in signal_max.values())
        noise = sum(v for v in noise_max.values())

        return linear_to_db(signal), linear_to_db(noise)

    def plot_fft(freqs, fft_db, signal_max, noise_max):
        for x, v in signal_max.items():
            v = linear_to_db(v)
            label = f"S {x:7.1f} {v:>6.1f}"
            plt.axvline(
                x=x,
                dashes=(1, 7),
                color="green",
                label=label,
            )
            print(label)

        for x, v in noise_max.items():
            v = linear_to_db(v)
            label = f"N {x:7.1f} {v:>6.1f}"
            plt.axvline(
                x=x,
                dashes=(1, 7),
                color="red",
                label=label,
            )
            print(label)

        plt.plot(freqs, fft_db)
        plt.xlabel("Frequency")
        # plt.legend()
        plt.show()

    def write_wav(filename, buffer, num_channels, sample_width, sample_rate, sample_norm):
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(num_channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes((buffer * sample_norm).astype(np.int16).tobytes())

    sample_rate = 44100
    num_channels = 2
    tone_freq = 3600.0
    length_sec = 30
    num_samples = length_sec * sample_rate
    fade_sec = 0.05
    fade_spl = int(fade_sec * sample_rate) + 1
    amp = 0.99

    silence = np.zeros((num_samples, num_channels))
    t = np.linspace(
        [0.0] * num_channels,
        [length_sec] * num_channels,
        length_sec * sample_rate
    )
    sine = amp * np.sin(2.0 * np.pi * tone_freq * t)
    sine[:fade_spl] *= np.linspace(0.0, 1.0, fade_spl)[:, np.newaxis]
    sine[-fade_spl:] *= np.linspace(1.0, 0.0, fade_spl)[:, np.newaxis]

    buffer = np.vstack([sine, silence]).astype(np.float32)

    begin = time.time()
    distorted = distort(buffer, gain_db=12.0, sample_rate=sample_rate, channels=num_channels)
    end = time.time()

    # write_wav("/tmp/distorted.wav", distorted, num_channels, 2, sample_rate, 32767.0)

    signal, noise = find_noise(distorted[:, 0], tone_freq, sample_rate)
    distorted_abs = np.abs(distorted[:, 0])
    clipping_count = np.sum(distorted_abs > 1.0)
    clipping = distorted_abs - np.clip(distorted_abs, 0.0, 1.0)
    clipping = clipping[clipping > 0.0]
    nans_infs = np.sum(~np.isfinite(distorted[:, 0]))

    extreme = distort(buffer, gain_db=50.0, sample_rate=sample_rate, channels=num_channels)
    nans_infs_extr = np.sum(~np.isfinite(extreme[:, 0]))

    result = {
        "samples": num_samples,
        "signal": float(signal),
        "noise": float(noise),
        "clipping_mean": 0.0 if clipping_count < 1 else float(clipping.mean()),
        "clipping_std": 0.0 if clipping_count < 1 else float(clipping.std()),
        "clipping_count": int(clipping_count),
        "perf": end - begin,
        "nans_infs": int(nans_infs),
        "nans_infs_extr": int(nans_infs_extr),
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run_test()

