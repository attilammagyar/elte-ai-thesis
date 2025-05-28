# challenge='distortion-tanh-noupspl', generator='gemini-2.5-pro-preview-05-06', temperature=0.0
# --- BEGIN GENERATED CODE ---

import numpy as np

def distort(samples, gain_db=12.0, sample_rate=44100.0, channels=2):
    # The `sample_rate` and `channels` parameters are not explicitly used in this
    # particular ADAA implementation method after initial setup, as the method adapts
    # to signal changes rather than fixed frequencies, and NumPy handles channel
    # operations implicitly. `channels` is implicitly handled by `samples.shape`.

    # Ensure consistent float32 type for samples, as per problem specification
    # (float32 numbers) and typical audio processing pipelines.
    samples = samples.astype(np.float32)

    # Define constants needed, ensuring they are float32
    _LOG_2_FLOAT32 = np.log(np.float32(2.0))
    # Threshold for switching to approximation for log(cosh(x)) to avoid overflow.
    # np.cosh(x_f32) overflows for x_f32 > ~88.7. We use 88.0 as a safe threshold.
    _COSH_OVERFLOW_THRESHOLD_FLOAT32 = np.float32(88.0)
    # Epsilon for checking if delta_x is close to zero.
    # Float32 machine epsilon is ~1.19e-7. A slightly larger value is robust.
    _EPSILON_FLOAT32 = np.float32(1e-7)

    # Helper function for stable calculation of log(cosh(x)) using float32 arithmetic.
    # Defined as a nested function to keep it within the scope of `distort`
    # and to allow it to capture constants from the outer scope.
    def _stable_log_cosh_f32_impl(x_input):
        # x_input is a NumPy array, expected to be float32.
        abs_x = np.abs(x_input)
        
        # Initialize result array with the same shape and dtype as x_input.
        res = np.zeros_like(x_input) 
        
        # Create masks for regions based on |x_input|.
        large_x_mask = abs_x >= _COSH_OVERFLOW_THRESHOLD_FLOAT32
        small_x_mask = ~large_x_mask # Equivalent to abs_x < _COSH_OVERFLOW_THRESHOLD_FLOAT32
        
        # For large |x|, use the approximation: log(cosh(x)) approx |x| - log(2)
        # This avoids overflow of cosh(x) and maintains precision.
        res[large_x_mask] = abs_x[large_x_mask] - _LOG_2_FLOAT32
        
        # For smaller |x|, direct computation of log(cosh(x)) is safe and accurate.
        # np.cosh and np.log on float32 inputs will produce float32 outputs.
        res[small_x_mask] = np.log(np.cosh(x_input[small_x_mask]))
        
        return res

    # Handle empty input array (N=0 samples).
    if samples.shape[0] == 0:
        # np.zeros_like correctly handles empty arrays, returning an empty array
        # of the same shape and specified dtype.
        return np.zeros_like(samples, dtype=np.float32)

    # 1. Convert gain from dB to a linear factor. Ensure it's float32.
    gain_linear = np.float32(10.0**(gain_db / 20.0))

    # 2. Apply gain to the input samples.
    # Since `samples` is float32 and `gain_linear` is float32, `gained_samples` will be float32.
    gained_samples = samples * gain_linear

    # 3. Prepare for 1st order Antiderivative Antialiasing (ADAA).
    # x_n represents the current gained sample values.
    x_n = gained_samples 
    
    # x_nm1 represents the previous gained sample values.
    # Initialize x_nm1 for the first sample (n=0). A common choice is 0.0,
    # representing a signal starting from a state of rest.
    x_nm1 = np.empty_like(x_n, dtype=np.float32)
    x_nm1[0, :] = np.float32(0.0) # Set previous value for the first actual sample to 0.
    if x_n.shape[0] > 1: # If there's more than one sample, fill the rest by shifting.
        x_nm1[1:, :] = x_n[:-1, :]

    # 4. Calculate delta_x = x_n - x_nm1. This is the change in signal value.
    delta_x = x_n - x_nm1

    # 5. Initialize the output array for distorted samples.
    distorted = np.zeros_like(x_n, dtype=np.float32)

    # 6. Create a boolean mask for samples where delta_x is very small (close to zero).
    mask_small_delta = np.abs(delta_x) < _EPSILON_FLOAT32
    
    # 7. Apply the ADAA formula based on the magnitude of delta_x.
    
    # Case 1: delta_x is small.
    # The formula (F(x_n) - F(x_nm1)) / delta_x becomes numerically unstable or undefined.
    # Approximation: y[n] = f(x[n]) = tanh(x[n]).
    # np.tanh on float32 input returns float32.
    distorted[mask_small_delta] = np.tanh(x_n[mask_small_delta])

    # Case 2: delta_x is not small.
    # Use the standard 1st order ADAA formula:
    # y[n] = (F(x[n]) - F(x[n-1])) / (x[n] - x[n-1]),
    # where F is the antiderivative of f. For f(x) = tanh(x), F(x) = log(cosh(x)).
    mask_large_delta = ~mask_small_delta
    
    # Perform calculations only if there are elements satisfying this condition.
    if np.any(mask_large_delta):
        # Select only the elements where delta_x is "large" for these calculations.
        x_n_large_delta = x_n[mask_large_delta]
        x_nm1_large_delta = x_nm1[mask_large_delta]
        # Get the actual delta_x values for division for these specific elements.
        delta_x_values_for_division = delta_x[mask_large_delta]
        
        # Calculate F(x_n) and F(x_nm1) using the stable log(cosh) implementation.
        F_xn = _stable_log_cosh_f32_impl(x_n_large_delta)
        F_xnm1 = _stable_log_cosh_f32_impl(x_nm1_large_delta)
        
        # Compute the distorted values for these elements.
        distorted[mask_large_delta] = (F_xn - F_xnm1) / delta_x_values_for_division
        
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

