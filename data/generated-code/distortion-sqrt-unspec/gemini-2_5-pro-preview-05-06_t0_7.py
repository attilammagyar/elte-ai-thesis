# challenge='distortion-sqrt-unspec', generator='gemini-2.5-pro-preview-05-06', temperature=0.7
# --- BEGIN GENERATED CODE ---

import numpy as np
import scipy.signal

def distort(samples, gain_db=12.0, sample_rate=44100.0, channels=2):
    # The 'distorted' array is initialized as per the problem description.
    # It will have the same shape and dtype as 'samples' (e.g., float32).
    distorted = np.zeros_like(samples)

    # 1. Convert gain from dB to a linear factor
    # The formula for dB to linear gain is: linear_gain = 10^(gain_db / 20)
    linear_gain = 10**(gain_db / 20.0)

    # 2. Define the oversampling factor.
    # Oversampling is used to reduce aliasing caused by the non-linear distortion.
    # A factor of 4 is chosen as a common compromise providing a good balance
    # between anti-aliasing effectiveness and computational cost.
    oversample_factor = 4

    # Process each audio channel independently.
    # The number of channels is determined by the input 'samples' array's shape.
    # The 'channels' parameter in the function signature is descriptive of
    # samples.shape[1] as per the problem statement `samples is (N, channels)`.
    num_actual_channels = samples.shape[1]

    for i in range(num_actual_channels):
        # Extract the current channel's data. This will be a 1D NumPy array.
        channel_data = samples[:, i]

        # 3. Upsample the channel data.
        # `scipy.signal.resample_poly` resamples the signal using a polyphase filter,
        # which includes appropriate anti-imaging filtering during upsampling.
        # Internally, SciPy functions often work with float64 for precision,
        # regardless of the input dtype if it's float.
        upsampled_channel = scipy.signal.resample_poly(channel_data, up=oversample_factor, down=1)

        # 4. Apply the specified gain to the upsampled signal.
        # This is a simple element-wise multiplication.
        gained_upsampled_channel = upsampled_channel * linear_gain

        # 5. Apply the non-linear distortion function: f(x) = x / sqrt(1 + x^2).
        # This function provides soft clipping, smoothly saturating the signal
        # and mapping input values to the output range [-1, 1].
        # It is numerically stable for standard floating-point inputs.
        x = gained_upsampled_channel
        distorted_upsampled_channel = x / np.sqrt(1.0 + x**2)

        # 6. Downsample the distorted channel data back to the original sample rate.
        # `scipy.signal.resample_poly` also handles the necessary anti-aliasing
        # (low-pass) filtering before decimation to prevent aliasing of the
        # newly generated harmonics.
        # The output of resample_poly will typically be float64.
        final_channel_data_f64 = scipy.signal.resample_poly(distorted_upsampled_channel, up=1, down=oversample_factor)
        
        # The length of `final_channel_data_f64` should match the original `channel_data` length
        # because upsampling by `oversample_factor` and then downsampling by the same factor
        # (i.e., `N_out = ceil( (N_in * up_factor) / down_factor )`) preserves the number of samples.

        # 7. Store the processed channel data in the corresponding channel of the
        # 'distorted' output array. It's important to cast the data back to the
        # original data type of 'samples' (e.g., float32) as `resample_poly`
        # might return float64.
        distorted[:, i] = final_channel_data_f64.astype(samples.dtype)
            
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

