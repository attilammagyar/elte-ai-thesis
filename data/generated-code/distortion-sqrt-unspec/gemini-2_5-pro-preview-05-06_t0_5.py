# challenge='distortion-sqrt-unspec', generator='gemini-2.5-pro-preview-05-06', temperature=0.5
# --- BEGIN GENERATED CODE ---

import numpy as np
from scipy import signal

def distort(samples, gain_db=12.0, sample_rate=44100.0, channels=2):
    """
    Applies gain increase and f(x)=x/sqrt(1+x^2) non-linearity to an audio signal
    while keeping aliasing to a minimum using oversampling.
    """

    # Handle empty input array
    if samples.shape[0] == 0:
        return samples.astype(np.float32) # Return empty array of the correct type

    # Define oversampling factor. 4x is a common choice for a good balance
    # between antialiasing quality and computational cost.
    oversample_factor = 4

    # 1. Convert gain from dB to linear scale
    # gain_linear will be float64 due to 10.0 and 20.0
    gain_linear = 10.0**(gain_db / 20.0)

    # Input `samples` are float32. Intermediate calculations might promote to float64.
    # The final output must be float32.

    # --- Stage 1: Upsampling ---
    if oversample_factor > 1:
        # Use resample_poly for efficient integer-factor resampling.
        # It applies an anti-aliasing/anti-imaging filter.
        # If `samples` (input) is float32, `resample_poly` output is also float32.
        current_signal_processed = signal.resample_poly(samples, oversample_factor, 1, axis=0)
    else:
        # If no oversampling, work directly on the input samples (or a copy if modification was in-place)
        # Since subsequent operations create new arrays, no copy is strictly needed here.
        current_signal_processed = samples

    # --- Stage 2: Apply Gain ---
    # `current_signal_processed` (float32) * `gain_linear` (float64) results in a float64 array.
    current_signal_processed = current_signal_processed * gain_linear

    # --- Stage 3: Apply Non-linear Distortion ---
    # f(x) = x / sqrt(1 + x^2)
    # This operation is performed on the float64 signal.
    # The term (1.0 + x^2) is always >= 1.0, so its square root is always >= 1.0.
    # This prevents division by zero or issues with sqrt of negative numbers.
    numerator = current_signal_processed
    denominator = np.sqrt(1.0 + current_signal_processed**2)
    distorted_signal_oversampled = numerator / denominator # Result is float64

    # --- Stage 4: Downsampling ---
    if oversample_factor > 1:
        # Downsample back to the original sample rate.
        # `distorted_signal_oversampled` is float64.
        # `resample_poly` output will also be float64.
        output_signal = signal.resample_poly(distorted_signal_oversampled, 1, oversample_factor, axis=0)
    else:
        # If no oversampling was done, the distorted signal is already at the original sample rate.
        output_signal = distorted_signal_oversampled # This is float64

    # --- Stage 5: Ensure Output Type ---
    # Cast the output signal back to float32, as specified for the input `samples`
    # and matching the type of the placeholder `distorted = np.zeros_like(samples)`
    # from the problem description.
    output_signal = output_signal.astype(np.float32)

    return output_signal



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

