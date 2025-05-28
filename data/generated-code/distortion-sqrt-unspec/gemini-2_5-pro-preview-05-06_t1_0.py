# challenge='distortion-sqrt-unspec', generator='gemini-2.5-pro-preview-05-06', temperature=1.0
# --- BEGIN GENERATED CODE ---

import numpy as np
from scipy import signal

def distort(samples, gain_db=12.0, sample_rate=44100.0, channels=2):
    distorted = np.zeros_like(samples) # Provided, must be populated

    # If N=0 (number of frames is zero), return the empty zeros array
    if samples.shape[0] == 0:
        return distorted

    # Store the data type of the output array (which is same as input array)
    output_dtype = distorted.dtype 

    # 1. Set oversampling factor for anti-aliasing.
    # 4x is a common choice for a good balance between quality and computation.
    # Higher factors (e.g., 8x, 16x) improve anti-aliasing further but increase CPU load.
    oversample_factor = 4

    # 2. Convert gain from dB to linear scale.
    # Python floats are typically float64, so gain_linear will be float64.
    gain_linear = 10.0**(gain_db / 20.0)

    # 3. Apply gain. Convert samples to float64 for precision during intermediate processing.
    # samples.astype(np.float64) ensures that the multiplication with float64 gain_linear
    # results in a float64 array. This helps maintain precision for filtering and other operations.
    processed_samples_float64 = samples.astype(np.float64) * gain_linear
    
    # 4. Upsample the signal.
    # resample_poly operates along axis=0 by default. If samples is (N, C), 
    # it processes C columns (channels) independently.
    # Inputting float64 data ensures filter calculations within resample_poly are likely
    # done with high precision (some scipy functions internally use float64 regardless of input type,
    # but being explicit with float64 input is safer for precision critical steps).
    upsampled_samples = signal.resample_poly(processed_samples_float64, 
                                             up=oversample_factor, 
                                             down=1, 
                                             axis=0) # axis=0 processes each channel

    # 5. Apply the non-linear distortion function: f(x) = x / sqrt(1 + x^2)
    # This function is a form of soft clipping; it's well-behaved and saturates towards +/-1 for large |x|.
    x = upsampled_samples
    # The denominator sqrt(1.0 + x**2) is always >= 1.0 for real x, so no division by zero.
    distorted_upsampled = x / np.sqrt(1.0 + x**2)

    # 6. Downsample the signal back to the original sample rate.
    # This step includes an anti-aliasing low-pass filter, which is automatically
    # applied by resample_poly before decimation to prevent aliasing.
    distorted_original_rate_float64 = signal.resample_poly(distorted_upsampled, 
                                                           up=1, 
                                                           down=oversample_factor, 
                                                           axis=0) # axis=0 processes each channel

    # 7. Convert the result back to the original data type (e.g., float32)
    # and populate the `distorted` array provided in the function skeleton.
    
    # The length of the output from resample_poly using integer up/down factors
    # (N * up / down) is typically N, matching the input length.
    # Output length calculation: num_output_samples = ceil(num_input_samples * up / down).
    # After upsampling: N_up = ceil(N * oversample_factor / 1) = N * oversample_factor.
    # After downsampling: N_final = ceil(N_up * 1 / oversample_factor) 
    #                            = ceil((N * oversample_factor) / oversample_factor) = N.
    # So, distorted_original_rate_float64.shape should ideally be equal to samples.shape.
    
    if distorted_original_rate_float64.shape[0] == distorted.shape[0]:
        # If lengths match, directly assign after type conversion.
        distorted[:] = distorted_original_rate_float64.astype(output_dtype)
    else:
        # Fallback for rare cases of length mismatch (e.g., very short signals,
        # or if specific resampler parameters caused slight deviations).
        # This truncates or leaves trailing zeros in `distorted` if `distorted_original_rate_float64` is shorter.
        min_len = min(distorted_original_rate_float64.shape[0], distorted.shape[0])
        distorted[:min_len, :] = distorted_original_rate_float64[:min_len, :].astype(output_dtype)
        # If distorted_original_rate_float64 was shorter, the rest of `distorted` (pre-filled with zeros) remains zero.
        # If distorted_original_rate_float64 was longer, its end is truncated.

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

