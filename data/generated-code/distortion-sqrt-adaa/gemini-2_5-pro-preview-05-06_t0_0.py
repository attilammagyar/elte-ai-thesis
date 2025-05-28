# challenge='distortion-sqrt-adaa', generator='gemini-2.5-pro-preview-05-06', temperature=0.0
# --- BEGIN GENERATED CODE ---

import numpy as np

def distort(samples, gain_db=12.0, sample_rate=44100.0, channels=2):
    """
    Applies gain increase and f(x)=x/sqrt(1+x^2) non-linearity to an audio signal
    while keeping aliasing to a minimum by using 1st order antiderivative anti-aliasing.

    Parameters:
    samples (np.ndarray): Input audio samples of shape (N, channels), float32.
    gain_db (float): Gain to be applied in dB.
    sample_rate (float): Sampling rate (not used in this specific implementation).
    channels (int): Number of audio channels (not used directly as samples.shape[1] is authoritative,
                    but part of the function signature as per requirements).

    Returns:
    np.ndarray: Distorted audio samples of the same shape and type as input.
    """

    num_samples = samples.shape[0]

    if num_samples == 0:
        # Return an empty array with the correct shape and type if input is empty
        return np.zeros_like(samples, dtype=np.float32)

    # 1. Convert gain from dB to linear scale
    # Ensure gain_linear is float32 to maintain precision if samples is float32
    gain_linear = np.float32(10**(gain_db / 20.0))

    # 2. Apply gain to samples
    # g will be float32 if samples (float32) and gain_linear (float32) are float32
    g = samples * gain_linear

    # 3. Initialize output array with the same shape and type as input samples
    # (Problem states input is float32, so output will also be float32)
    distorted = np.zeros_like(samples, dtype=np.float32)

    # 4. Define the non-linear function f(x) and its antiderivative F(x)
    # These functions are designed to work element-wise on NumPy arrays
    # and maintain float32 precision.
    
    # Non-linear function: f(x) = x / sqrt(1 + x^2)
    def f_x_vec(x_val):
        # Using np.float32(1.0) to encourage float32 arithmetic
        return x_val / np.sqrt(np.float32(1.0) + x_val**2)

    # Antiderivative of f(x): F(x) = integral of f(x) dx = sqrt(1 + x^2)
    def F_x_vec(x_val):
        return np.sqrt(np.float32(1.0) + x_val**2)

    # 5. Apply non-linearity to the first sample directly
    # For n=0, there is no previous sample g[n-1], so ADAA formula cannot be used.
    # Standard practice is to use f(g[0]) for the first sample.
    # This applies f(g[0]) for each channel.
    distorted[0, :] = f_x_vec(g[0, :])

    # 6. Apply 1st order ADAA for subsequent samples (n > 0)
    if num_samples > 1:
        # g_n represents the current samples g[n], g_n_minus_1 represents g[n-1]
        # These slices result in arrays of shape (num_samples-1, num_channels_in_sample)
        g_n = g[1:, :]          # Samples from index 1 to N-1
        g_n_minus_1 = g[:-1, :] # Samples from index 0 to N-2

        # Difference between current and previous gain-applied samples
        delta_g = g_n - g_n_minus_1

        # Create a temporary array to store results for samples from n=1 onwards.
        # Initialize with float32 type.
        y_n_adaa_values = np.zeros_like(g_n, dtype=np.float32)

        # Mask for cases where g[n] == g[n-1] (i.e., delta_g is zero)
        # Comparison with np.float32(0.0) for type consistency.
        zero_delta_mask = (delta_g == np.float32(0.0))
        
        # Mask for cases where g[n] != g[n-1] (i.e., delta_g is non-zero)
        nonzero_delta_mask = ~zero_delta_mask
        # Alternative: nonzero_delta_mask = (delta_g != np.float32(0.0))

        # Case 1: delta_g is non-zero. Apply the ADAA formula.
        # y[n] = (F(g[n]) - F(g[n-1])) / (g[n] - g[n-1])
        # We operate on masked selections to ensure calculations are only for relevant elements.
        # g_n[nonzero_delta_mask] extracts elements from g_n where delta_g is non-zero.
        # This results in a 1D array if g_n is 2D. Operations are element-wise.
        # Assignment back using y_n_adaa_values[nonzero_delta_mask] correctly maps
        # these 1D results to their original positions in the 2D array.
        
        # Extract values for the non-zero delta path
        g_n_nz = g_n[nonzero_delta_mask]
        g_n_minus_1_nz = g_n_minus_1[nonzero_delta_mask]
        delta_g_nz = delta_g[nonzero_delta_mask]
        
        # Compute ADAA for non-zero delta_g. This avoids division by zero.
        # Check if there are any elements to process to prevent empty array warnings if all deltas are zero.
        if np.any(nonzero_delta_mask): # Or check if delta_g_nz.size > 0
            y_n_adaa_values[nonzero_delta_mask] = \
                (F_x_vec(g_n_nz) - F_x_vec(g_n_minus_1_nz)) / delta_g_nz
        
        # Case 2: delta_g is zero. Apply the direct non-linearity f(g[n]).
        # y[n] = f(g[n])
        # (since g[n] = g[n-1], f(g[n-1]) would also be correct and numerically identical)
        
        # Extract values of g[n] for the zero delta path
        g_n_z = g_n[zero_delta_mask]
        
        # Check if there are any elements to process.
        if np.any(zero_delta_mask): # Or check if g_n_z.size > 0
            y_n_adaa_values[zero_delta_mask] = f_x_vec(g_n_z)
        
        # Assign the computed ADAA values to the corresponding part of the output array
        distorted[1:, :] = y_n_adaa_values
        
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

