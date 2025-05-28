# challenge='distortion-sqrt-adaa', generator='gemini-2.5-pro-preview-05-06', temperature=0.7
# --- BEGIN GENERATED CODE ---

import numpy as np

def distort(samples, gain_db=12.0, sample_rate=44100.0, channels=2):
    # The skeleton initializes `distorted` and returns it.
    # My code will calculate the actual distorted signal and assign it to `distorted`.
    # `sample_rate` and `channels` parameters are not used in this specific implementation
    # as the number of channels is inferred from `samples.shape` and sample rate is not
    # needed for 1st order ADAA of a memoryless nonlinearity with an analytical integral.

    # Ensure calculations use float32, consistent with input `samples` type.
    # Python floats are float64 by default, so explicit casting of constants is good practice.
    gain_db_f32 = np.float32(gain_db)
    
    # 1. Convert gain from dB to linear scale
    linear_gain = np.float32(10.0)**(gain_db_f32 / np.float32(20.0))

    # 2. Apply gain to input samples
    # `samples` is (N, channels) float32. `x` will also be (N, channels) float32.
    x = samples * linear_gain

    # 3. Prepare the `x_prev` array (previous samples)
    # `x_prev[i]` should be `x[i-1]`.
    # For the first sample (index 0), `x_prev[0]` is the initial condition, typically 0.0.
    x_prev = np.zeros_like(x, dtype=np.float32) # Initialize all previous samples to 0.0.
    
    # Populate `x_prev` for samples from index 1 onwards.
    # `x_prev[0, :]` remains 0.0 for all channels.
    # If N=0, x.shape[0] is 0. x[:-1] is empty, x_prev[1:] is an empty slice. NumPy handles this.
    x_prev[1:] = x[:-1] # x_prev[i,c] = x[i-1,c] for i >= 1

    # 4. Define the antiderivative F(v) of the non-linear function f(v).
    # The non-linear function is f(v) = v / sqrt(1 + v^2).
    # Its antiderivative is F(v) = sqrt(1 + v^2).
    # This will be applied element-wise to NumPy arrays.
    def F_antiderivative_func(v_arr):
        # Using np.float32(1.0) ensures intermediate calculations also tend towards float32.
        # np.sqrt maintains the dtype of its argument (if real).
        return np.sqrt(np.float32(1.0) + v_arr**2)

    # 5. Apply the 1st order Antiderivative Anti-Aliasing (ADAA) formula.
    # For F(v) = sqrt(C + v^2), the standard ADAA formula:
    #   y_n = (F(x_n) - F(x_{n-1})) / (x_n - x_{n-1})  (if x_n != x_{n-1})
    #   y_n = f(x_n)                                   (if x_n == x_{n-1})
    # can be algebraically simplified to a more numerically robust form that
    # also inherently handles the x_n == x_{n-1} case:
    #   y_n = (x_n + x_{n-1}) / (F(x_n) + F(x_{n-1}))
    
    numerator = x + x_prev
    
    # Calculate F(x_n) and F(x_{n-1}) using the element-wise function
    Fx_n = F_antiderivative_func(x)
    Fx_prev = F_antiderivative_func(x_prev)
    
    denominator = Fx_n + Fx_prev
    
    # Numerical stability of the denominator:
    # F(v) = sqrt(1 + v^2) is always >= 1 for real v.
    # Therefore, denominator = F(x_n) + F(x_{n-1}) is always >= 2.0.
    # This prevents division by zero.
    
    # Calculate the distorted signal. Result will be float32.
    # This assignment will overwrite the `distorted = np.zeros_like(samples)` from the skeleton.
    distorted = numerator / denominator
    
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

