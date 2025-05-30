# challenge='distortion-tanh-unspec', generator='gemini-2.5-pro-preview-05-06', temperature=0.0
# --- BEGIN GENERATED CODE ---

import numpy as np
import scipy.signal

def distort(samples, gain_db=12.0, sample_rate=44100.0, channels=2):
    """
    Applies gain increase and tanh non-linearity to an audio signal
    with anti-aliasing.

    Parameters:
    samples (np.ndarray): Input audio signal of shape (N, channels), float32.
    gain_db (float): Gain to apply in decibels.
    sample_rate (float): Sampling rate of the audio signal.
    channels (int): Number of audio channels.

    Returns:
    np.ndarray: Distorted audio signal of the same shape and dtype as input.
    """

    # Note: The `sample_rate` and `channels` parameters from the function signature
    # are not explicitly used in this implementation logic, as the operations
    # are based on the properties of the `samples` array (shape, dtype) and
    # a fixed oversampling factor. They are included to match the required signature.

    # If samples is empty (e.g. shape is (0, 2)), scipy.signal.resample_poly
    # correctly returns an empty array. Processing will proceed and result
    # in an appropriately shaped empty array of the original dtype.
    # Thus, no special handling for empty input `samples` is strictly needed here.

    # 1. Define oversampling factor
    # A common choice for balancing anti-aliasing effectiveness and computational cost.
    # Higher factors (e.g., 8x, 16x) provide better aliasing reduction but are slower.
    oversampling_factor = 4

    # 2. Convert gain from dB to linear scale
    # gain_db = 20 * log10(gain_linear) => gain_linear = 10^(gain_db / 20)
    gain_linear = 10**(gain_db / 20.0)

    # Store the original data type of the input samples (e.g., np.float32)
    # to ensure the output matches it.
    original_dtype = samples.dtype

    # scipy.signal.resample_poly typically uses float64 for its internal calculations
    # to maintain precision. If `samples` is float32, it will be upcast.

    # 3. Upsample the signal
    # This increases the effective sampling rate, pushing the Nyquist frequency higher,
    # which provides more "space" for harmonics generated by the non-linearity.
    # `axis=0` ensures that resampling is performed along the time dimension (rows)
    # independently for each channel (columns).
    upsampled_samples = scipy.signal.resample_poly(samples, up=oversampling_factor, down=1, axis=0)

    # 4. Apply gain
    # This multiplication is an element-wise operation.
    boosted_upsampled_samples = upsampled_samples * gain_linear

    # 5. Apply tanh non-linearity (soft clipping)
    # np.tanh is a common function for simulating distortion. It's applied element-wise.
    # This non-linear process generates new harmonic content.
    distorted_upsampled_samples = np.tanh(boosted_upsampled_samples)

    # 6. Downsample the signal back to the original sampling rate
    # This step is crucial and includes an anti-aliasing low-pass filter.
    # The filter removes harmonics that fall above the original Nyquist frequency
    # (sample_rate / 2), preventing them from aliasing into the audible band.
    distorted_signal_processed = scipy.signal.resample_poly(distorted_upsampled_samples, up=1, down=oversampling_factor, axis=0)

    # The length of the output from resample_poly should generally match the original
    # number of samples (N) due to the nature of the up/down factors.
    # Specifically, for an input of N samples:
    # Length after upsampling: N_up = ceil(N * oversampling_factor / 1)
    # Length after downsampling: N_down = ceil(N_up * 1 / oversampling_factor)
    # If N * oversampling_factor is an integer, N_up = N * oversampling_factor.
    # Then N_down = ceil((N * oversampling_factor) / oversampling_factor) = N.

    # 7. Convert the processed signal back to the original data type
    distorted = distorted_signal_processed.astype(original_dtype)

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

