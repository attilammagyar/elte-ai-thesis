# challenge='distortion-tanh-adaa', generator='gemini-2.5-pro-preview-05-06', temperature=0.01
# --- BEGIN GENERATED CODE ---

import numpy as np

def distort(samples, gain_db=12.0, sample_rate=44100.0, channels=2):
    # Store original dtype to cast back at the end. Input 'samples' are float32.
    original_dtype = samples.dtype

    # Convert gain from dB to linear.
    # Perform calculations in float64 for precision, as intermediate values (e.g., in F(x))
    # can become very large or require high precision.
    gain_lin = 10.0**(gain_db / 20.0)
    
    # Apply gain to samples. Cast samples to float64 for calculation.
    x = samples.astype(np.float64) * gain_lin

    # Create x_prev (previous values of x, i.e., x[n-1]).
    # For the first sample (n=0), x_prev is initialized to 0.
    x_prev = np.zeros_like(x)  # x is of shape (N, channels), type float64
    # x_prev[0,:] is already 0 due to np.zeros_like.
    # For n > 0, set x_prev[n,:] = x[n-1,:]
    # This slicing works correctly even if x.shape[0] is 0 or 1.
    x_prev[1:, :] = x[:-1, :]

    # The non-linearity is f(x) = tanh(x).
    # Its antiderivative is F(x) = ∫ tanh(x) dx = ln(cosh(x)).

    # We need a numerically stable implementation for F(x) = ln(cosh(x)).
    # Standard np.log(np.cosh(x)) can overflow for large |x| (e.g., |x| > 709 for float64).
    # A stable form is F(val) = np.logaddexp(val, -val) - np.log(2.0).
    # np.logaddexp(a, b) computes log(e^a + e^b) stably.
    # So, F(val) = log(e^val + e^(-val)) - log(2) = log((e^val + e^(-val))/2) = log(cosh(val)).
    log_2 = np.log(2.0)  # Precompute log(2) as float64
    
    F_x = np.logaddexp(x, -x) - log_2
    F_x_prev = np.logaddexp(x_prev, -x_prev) - log_2

    # Difference between current and previous input samples to the non-linearity
    delta_x = x - x_prev

    # Epsilon for checking if delta_x is close to zero.
    # This is to handle the case where x_n = x_{n-1}, where the ADAA formula is undefined (0/0)
    # and should take its limiting value f(x_n).
    # A common threshold for float64 differences in audio DSP is around 1e-9.
    epsilon = 1e-9

    # Case 1: delta_x is close to zero (i.e., x_n approx x_{n-1}).
    # The 1st order ADAA output is f(x_n) = tanh(x_n).
    y_if_delta_zero = np.tanh(x)

    # Case 2: delta_x is not close to zero.
    # The 1st order ADAA output is (F(x_n) - F(x_{n-1})) / (x_n - x_{n-1}).
    # We use np.errstate to suppress warnings (e.g., division by zero or invalid value for 0/0)
    # for elements where delta_x is extremely small (or zero) but not caught by the epsilon threshold,
    # or for elements that are correctly handled by the `condition` in `np.where`.
    # The results of these problematic divisions will be masked out by np.where.
    with np.errstate(divide='ignore', invalid='ignore'):
        quotient = (F_x - F_x_prev) / delta_x
    
    # Combine cases using np.where.
    # If abs(delta_x) < epsilon, use y_if_delta_zero. Otherwise, use the quotient.
    condition = np.abs(delta_x) < epsilon
    distorted_float64 = np.where(condition, y_if_delta_zero, quotient)
    
    # If the input `samples` contained NaNs, they will propagate through calculations:
    # e.g., tanh(NaN) is NaN, logaddexp(NaN, ...) is NaN.
    # The output `distorted` will correctly have NaNs at corresponding positions.

    # Cast the result back to the original dtype of the input samples (float32).
    distorted = distorted_float64.astype(original_dtype)
    
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

