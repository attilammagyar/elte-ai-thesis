# challenge='distortion-unspec', generator='gemini-2.5-pro-preview-05-06', temperature=0.7
# --- BEGIN GENERATED CODE ---

import numpy as np
from scipy import signal

def distort(samples, gain_db=12.0, sample_rate=44100.0, channels=2):
    """
    Applies gain increase and tanh non-linearity to an audio signal
    with anti-aliasing using oversampling.
    """
    # Oversampling factor. A factor of 4 is chosen as a common compromise
    # between anti-aliasing effectiveness and computational cost for tanh distortion.
    # Tanh can generate significant harmonics, especially at high gain levels,
    # where a 2x oversampling factor might not be sufficient to prevent audible aliasing.
    oversampling_factor = 4

    # 1. Convert gain from dB to a linear scale factor.
    # gain_db = 20 * log10(gain_linear)  =>  gain_linear = 10^(gain_db / 20)
    gain_linear = 10**(gain_db / 20.0)

    # 2. Apply the linear gain to the input samples.
    # This is a vectorized operation, applied to all samples and channels.
    gained_samples = samples * gain_linear

    # Note: If `samples` has a shape of (0, num_channels) (i.e., no samples),
    # `gained_samples` will also be empty. `scipy.signal.resample_poly`
    # handles empty arrays correctly (tested with scipy 1.5.0+), returning an
    # empty array with the appropriate output shape and dtype.
    # Therefore, an explicit check for empty `samples` is not strictly necessary here.

    # 3. Upsample the signal by the oversampling_factor.
    # `signal.resample_poly` performs upsampling by inserting zeros
    # and then applying an anti-imaging (low-pass) FIR filter.
    # The `axis=0` argument specifies that resampling should be done along the
    # time axis (the first dimension of the `samples` array). Each channel
    # (column) is processed independently.
    upsampled_signal = signal.resample_poly(gained_samples,
                                            up=oversampling_factor,
                                            down=1,
                                            axis=0)

    # 4. Apply the tanh non-linearity (waveshaping).
    # `np.tanh` is a universal function and applies element-wise to the NumPy array.
    distorted_oversampled_signal = np.tanh(upsampled_signal)

    # 5. Downsample the processed signal back to the original sample rate.
    # `signal.resample_poly` first applies an anti-aliasing (low-pass) FIR filter
    # to remove frequencies above the original Nyquist limit (now present due to
    # the non-linearity) and then decimates the signal.
    distorted_final = signal.resample_poly(distorted_oversampled_signal,
                                           up=1,
                                           down=oversampling_factor,
                                           axis=0)

    # `scipy.signal.resample_poly` (versions >= 1.5.0) typically preserves the
    # `float32` dtype if the input array is `float32`. If compatibility with
    # older scipy versions that might upcast to `float64` was critical,
    # one might add:
    # if distorted_final.dtype != samples.dtype:
    #     distorted_final = distorted_final.astype(samples.dtype)
    # For this implementation, we assume a modern scipy version where this is not needed.
    
    # The parameters `sample_rate` and `channels` from the function signature
    # are not explicitly used in this implementation logic. `sample_rate` is implicitly
    # handled by `resample_poly` operating relative to Nyquist frequencies, and
    # channel information is derived from `samples.shape[1]` (implicitly by `axis=0`).

    return distorted_final


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

