# challenge='distortion-adaa', generator='gemini-2.5-pro-preview-05-06', temperature=0.3
# --- BEGIN GENERATED CODE ---

import numpy as np

def distort(samples, gain_db=12.0, sample_rate=44100.0, channels=2):
    """
    Applies gain increase and tanh non-linearity to an audio signal
    using 1st order antiderivative anti-aliasing (ADAA).

    Parameters:
    samples (np.ndarray): Input audio samples, shape (N, channels), float32.
    gain_db (float): Gain to apply in decibels.
    sample_rate (float): Sampling rate (unused in this 1st order ADAA for tanh,
                         as tanh is a memoryless non-linearity).
    channels (int): Number of audio channels (primarily for signature consistency;
                    actual channel count is derived from samples.shape[1]).

    Returns:
    np.ndarray: Distorted audio samples, same shape and type as input.
    """

    # 1. Convert gain_db to linear gain
    gain_linear = 10**(gain_db / 20.0)

    # 2. Apply gain to input samples. This is the signal x_g[n] entering the non-linearity.
    # The input `samples` array is expected to be float32. NumPy operations
    # will maintain precision or promote to float64 for intermediate calculations
    # if necessary, then results can be cast back if assigned to a float32 array.
    x_g = samples * gain_linear

    # 3. Get previous samples x_g[n-1]
    # For the first sample (n=0), x_g_prev[0] is the initial state (typically 0.0).
    x_g_prev = np.zeros_like(x_g) # Initializes with zeros, matching dtype of x_g.
    
    # If there are samples to process (N > 0)
    if x_g.shape[0] > 0:
        # x_g_prev[n] = x_g[n-1] for n > 0
        x_g_prev[1:, :] = x_g[:-1, :] 
        # x_g_prev[0, :] remains 0.0, serving as the initial condition x_g[-1] = 0.

    # 4. Calculate the difference in input to non-linearity: delta_x_g = x_g[n] - x_g[n-1]
    delta_x_g = x_g - x_g_prev

    # 5. Define the non-linear function f(u) = tanh(u)
    #    and its antiderivative F(u) = ln(cosh(u)).
    #    A numerically stable way to compute F(u) = ln(cosh(u)) is needed.
    
    def F_stable_log_cosh(u):
        """Numerically stable computation of F(u) = ln(cosh(u))."""
        # ln(cosh(u)) = ln((e^u + e^-u)/2) = ln(e^u + e^-u) - ln(2)
        # np.logaddexp(u, -u) computes log(e^u + e^-u) stably, avoiding overflow for large |u|.
        return np.logaddexp(u, -u) - np.log(2.0) # Use 2.0 for float context

    # 6. Calculate F(x_g[n]) and F(x_g[n-1]) using the stable function
    F_x_g = F_stable_log_cosh(x_g)
    F_x_g_prev = F_stable_log_cosh(x_g_prev)
    
    # Calculate the difference of the antiderivative values: delta_F = F(x_g[n]) - F(x_g[n-1])
    delta_F = F_x_g - F_x_g_prev

    # 7. Compute distorted samples using the 1st order ADAA formula.
    # Initialize the output with the case where delta_x_g is very small (or zero).
    # In this limiting case, y[n] = f(x_g[n]) = tanh(x_g[n]).
    # np.tanh will produce an array of the same dtype as x_g if x_g is float,
    # or float64 if x_g is int (though x_g here is float due to gain_linear).
    distorted_output = np.tanh(x_g)

    # Define a small epsilon for checking if delta_x_g is close to zero.
    # This threshold determines when to switch from the f(x_g[n]) approximation
    # to the full ADAA formula.
    epsilon = 1e-8 # A common choice for such thresholds. Values much smaller
                   # (e.g., < float32.eps) can be problematic.

    # Create a boolean mask for elements where delta_x_g is large enough for division.
    # This avoids division by zero or by numerically insignificant (and potentially noisy) small numbers.
    condition_mask = np.abs(delta_x_g) > epsilon

    # Where condition_mask is True (i.e., |delta_x_g| > epsilon),
    # apply the ADAA formula: y[n] = delta_F / delta_x_g.
    # This calculation is only performed for elements satisfying the condition.
    
    # Check if any elements satisfy the condition. This avoids attempting to index
    # with an all-False mask if that could lead to issues, though NumPy handles this.
    # More importantly, it avoids division if the denominator array slice would be empty.
    if np.any(condition_mask):
        # Retrieve the delta_F and delta_x_g values only for the masked elements
        masked_delta_F = delta_F[condition_mask]
        masked_delta_x_g = delta_x_g[condition_mask]
        
        # Compute the ADAA term for these selected elements
        adaa_values = masked_delta_F / masked_delta_x_g
        
        # Assign these computed values back to the corresponding positions in the output array.
        # If distorted_output is float32, and adaa_values is float64 (due to division),
        # this assignment will cast adaa_values to float32.
        distorted_output[condition_mask] = adaa_values
    
    # The 'distorted_output' array's dtype is determined by np.tanh(x_g).
    # If 'samples' (and thus 'x_g') is float32, 'distorted_output' will be float32.
    # This matches the typical requirement for audio samples.
    
    return distorted_output


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

