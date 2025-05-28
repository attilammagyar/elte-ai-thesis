# challenge='distortion-sqrt-adaa', generator='gemini-2.5-pro-preview-05-06', temperature=0.01
# --- BEGIN GENERATED CODE ---

import numpy as np

def distort(samples, gain_db=12.0, sample_rate=44100.0, channels=2):
    """
    Applies gain increase and f(x)=x/sqrt(1+x^2) non-linearity to an audio signal
    using 1st order antiderivative anti-aliasing (ADAA).

    Parameters:
    samples (np.ndarray): Input audio samples, shape (N, channels), float32.
    gain_db (float): Gain to apply in dB.
    sample_rate (float): Sampling rate (unused in this specific implementation).
    channels (int): Number of audio channels (unused, derived from samples.shape[1]).

    Returns:
    np.ndarray: Distorted audio samples, shape (N, channels), float32.
    """

    # 0. Handle empty input array
    if samples.shape[0] == 0:
        return np.zeros_like(samples, dtype=np.float32)

    # 1. Convert gain_db to linear gain factor
    linear_gain = 10.0**(gain_db / 20.0)

    # 2. Apply gain to samples
    # samples (float32) * linear_gain (float64) results in a float64 array.
    # Intermediate calculations are done in float64 for precision.
    x_g = samples * linear_gain

    # 3. Prepare current samples (x_n) and previous samples (x_prev)
    x_n = x_g
    
    # x_prev holds x_g[i-1]. For i=0, x_prev[0, c] = 0.0.
    x_prev = np.zeros_like(x_g)  # Inherits dtype from x_g (float64)
    if x_g.shape[0] > 1:
        x_prev[1:, :] = x_g[:-1, :]

    # 4. Define terms for ADAA
    # Non-linear function: f(x) = x / sqrt(1 + x^2)
    # Antiderivative: F(x) = sqrt(1 + x^2)

    # Calculate F(x_n) and F(x_prev)
    F_xn = np.sqrt(1.0 + x_n**2)
    F_xprev = np.sqrt(1.0 + x_prev**2)

    # Calculate differences delta_F and delta_x
    delta_F = F_xn - F_xprev
    delta_x = x_n - x_prev

    # 5. Apply 1st order ADAA formula
    # y[n] = (F(x[n]) - F(x[n-1])) / (x[n] - x[n-1])  if x[n] != x[n-1]
    # y[n] = f(x[n])                                   if x[n] == x[n-1] (or delta_x is small)
    distorted_signal = np.zeros_like(x_g) # Output array, float64 initially

    # Epsilon for comparing delta_x to zero
    epsilon = 1e-9

    # Case 1: delta_x is "close to zero" (x_n approx x_prev) -> y_n = f(x_n)
    zero_delta_indices = np.abs(delta_x) < epsilon
    
    # f(x_n) = x_n / sqrt(1 + x_n^2)
    # Denominator sqrt(1.0 + x_n**2) is always >= 1.0.
    fx_n = x_n / np.sqrt(1.0 + x_n**2) 
    distorted_signal[zero_delta_indices] = fx_n[zero_delta_indices]

    # Case 2: delta_x is "not close to zero" -> y_n = delta_F / delta_x
    non_zero_delta_indices = ~zero_delta_indices
    
    # Perform division only on elements where delta_x is not close to zero.
    distorted_signal[non_zero_delta_indices] = (
        delta_F[non_zero_delta_indices] / delta_x[non_zero_delta_indices]
    )

    # 6. Convert the output signal back to float32
    distorted_signal = distorted_signal.astype(np.float32)

    return distorted_signal


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

