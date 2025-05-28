# challenge='distortion-sqrt-adaa', generator='gemini-2.5-pro-preview-05-06', temperature=1.0
# --- BEGIN GENERATED CODE ---

import numpy as np

# Helper function: Non-linear function f(x) = x / sqrt(1 + x^2)
def _f_nonlinear_for_adaa(x_val):
    # Ensure floating point arithmetic; 1.0 forces float promotion.
    # Denominator is np.sqrt(1.0 + x_val**2), which is >= 1.0 for real x_val,
    # so no division by zero from sqrt result.
    # Handles x_val = 0 correctly (returns 0.0).
    # Handles array inputs correctly due to NumPy's ufuncs.
    return x_val / np.sqrt(1.0 + x_val**2)

# Helper function: Antiderivative F(x) of f(x), which is sqrt(1 + x^2)
def _F_antiderivative_for_adaa(x_val):
    # Ensure floating point arithmetic.
    # Handles array inputs correctly.
    return np.sqrt(1.0 + x_val**2)

def distort(samples, gain_db=12.0, sample_rate=44100.0, channels=2):
    """
    Applies gain increase and f(x)=x/sqrt(1+x^2) non-linearity 
    to an audio signal using 1st order antiderivative anti-aliasing (ADAA).

    Parameters:
    samples (np.ndarray): Input audio samples, shape (N, channels), dtype float32.
    gain_db (float): Gain to apply in decibels.
    sample_rate (float): Sampling rate of the audio. Not used in 1st order ADAA.
    channels (int): Number of audio channels in the 'samples' array.

    Returns:
    np.ndarray: Distorted audio samples, same shape and dtype as input 'samples'.
    """
    
    # 1. Convert gain from dB to linear factor
    # Python's float type is typically a 64-bit double-precision float.
    gain_linear = 10.0**(gain_db / 20.0) 

    # 2. Apply gain to samples
    # samples (float32) * gain_linear (float64) results in gained_samples (float64).
    # Using float64 for intermediate calculations improves precision.
    gained_samples = samples * gain_linear 
    
    # Initialize the output array with the same shape as input and original dtype (float32)
    distorted = np.zeros_like(samples, dtype=np.float32)

    # Handle the edge case of empty input samples
    if samples.shape[0] == 0:
        return distorted

    # 3. Process each channel independently using ADAA
    # This loop iterates over channels (e.g., 1 for mono, 2 for stereo).
    # Operations within the loop are vectorized across samples of a single channel.
    for c in range(channels):
        # Extract current channel's samples (these are float64 due to previous operations)
        x_n = gained_samples[:, c] 
        
        # Create x_{n-1} (previous sample series) for the current channel.
        # Initialize with zeros, ensuring dtype matches x_n (float64) for consistency.
        x_n_minus_1 = np.zeros_like(x_n, dtype=x_n.dtype) 
        
        # For the first sample (x_n[0]), its "previous" sample (x_{n-1}[0])
        # is assumed to be 0.0. This is a common initial condition.
        # This access is safe as samples.shape[0] > 0 is guaranteed at this point.
        x_n_minus_1[0] = 0.0 
        
        # For subsequent samples (i > 0), x_{n-1}[i] is the actual previous sample x_n[i-1].
        if x_n.shape[0] > 1: # Check if there is more than one sample in the channel
            x_n_minus_1[1:] = x_n[:-1] # Vectorized assignment for all other elements
            
        # Calculate the difference: delta_x = x_n - x_{n-1}
        delta_x = x_n - x_n_minus_1 # This will be a float64 array
        
        # Initialize the output array for the current channel (float64 for precision)
        y_n_channel = np.zeros_like(x_n, dtype=x_n.dtype)
        
        # Create boolean masks to identify where delta_x is zero or non-zero.
        # This is for applying the two different cases of the ADAA formula.
        
        # Condition 1: delta_x == 0.0 (i.e., x_n and x_{n-1} are numerically identical)
        idx_delta_zero = (delta_x == 0.0)
        
        # Condition 2: delta_x != 0.0 (i.e., x_n and x_{n-1} are different)
        idx_delta_not_zero = ~idx_delta_zero # Efficiently computes (delta_x != 0.0)
        
        # --- Apply 1st Order ADAA formulas ---
        
        # Case 1: delta_x == 0.0
        # The output y[n] is simply f(x[n]).
        # _f_nonlinear_for_adaa is called with x_n[idx_delta_zero], which are the elements of x_n
        # where the corresponding delta_x is zero.
        # If no elements satisfy idx_delta_zero, an empty array is passed and processed, which is valid.
        y_n_channel[idx_delta_zero] = _f_nonlinear_for_adaa(x_n[idx_delta_zero])
            
        # Case 2: delta_x != 0.0
        # The output y[n] = (F(x[n]) - F(x[n-1])) / (x[n] - x[n-1]).
        # We sub-select only the elements where delta_x is non-zero.
        # This is crucial to avoid division by zero if we were to use the full delta_x array.
        x_n_nz = x_n[idx_delta_not_zero]
        x_n_minus_1_nz = x_n_minus_1[idx_delta_not_zero]
        delta_x_nz = delta_x[idx_delta_not_zero] # These delta_x values are guaranteed non-zero.
        
        # Calculate numerator: F(x_n) - F(x_{n-1}) for the non-zero delta_x parts
        numerator = _F_antiderivative_for_adaa(x_n_nz) - _F_antiderivative_for_adaa(x_n_minus_1_nz)
        
        # Division is safe as all elements in delta_x_nz are non-zero.
        y_n_channel[idx_delta_not_zero] = numerator / delta_x_nz
            
        # Assign the processed channel's samples (float64) back to the corresponding
        # column in the main 'distorted' array.
        # This involves an implicit cast from float64 (in y_n_channel) to float32 (dtype of 'distorted').
        distorted[:, c] = y_n_channel
        
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

