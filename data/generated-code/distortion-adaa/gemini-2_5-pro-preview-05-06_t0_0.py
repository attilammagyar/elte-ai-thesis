# challenge='distortion-adaa', generator='gemini-2.5-pro-preview-05-06', temperature=0.0
# --- BEGIN GENERATED CODE ---

import numpy as np

def distort(samples, gain_db=12.0, sample_rate=44100.0, channels=2): # sample_rate and channels are not strictly needed for this specific method
    # 1. Convert gain from dB to linear
    # Python floats (gain_db, 20.0, 10.0) are typically float64.
    # gain_linear will be float64.
    gain_linear = 10.0**(gain_db / 20.0)

    # 2. Apply gain to input samples
    # If `samples` is float32 and `gain_linear` is float64, `x_gained` becomes float64.
    # This use of float64 for intermediate calculations is generally good for precision.
    x_gained = samples * gain_linear

    # Handle empty input array (shape[0] == 0)
    if x_gained.shape[0] == 0:
        # Return an empty array with the same dtype as the input samples
        return np.zeros_like(samples) 

    # 3. Prepare current (x_curr) and previous (x_prev) gained samples
    # x_curr is simply x_gained
    x_curr = x_gained # dtype is float64 if samples is float32 and gain_linear is float64

    # x_prev is x_gained shifted by one sample.
    # The first sample's previous value is assumed to be 0.0 for each channel.
    # x_gained.shape[1] correctly gives the number of channels from the data.
    # dtype=x_curr.dtype ensures x_prev has the same dtype as x_curr (e.g., float64).
    num_input_channels = x_gained.shape[1]
    x_prev = np.concatenate(
        (np.zeros((1, num_input_channels), dtype=x_curr.dtype), x_curr[:-1, :]),
        axis=0
    )

    # 4. Compute antiderivative F(x) = log(cosh(x)) for x_curr and x_prev
    # We use the numerically stable version: F(x) = |x| - log(2) + log1p(exp(-2*|x|))
    # All calculations here will be float64 if x_curr, x_prev are float64,
    # as Python float constants (2.0) and NumPy functions (log, exp, log1p) default to float64.
    
    abs_x_curr = np.abs(x_curr)
    # np.log(2.0) is a float64 constant. 
    # np.exp(-2.0 * abs_x_curr) involves float64 constant, result is float64. 
    # np.log1p result is float64.
    F_x_curr = abs_x_curr - np.log(2.0) + np.log1p(np.exp(-2.0 * abs_x_curr))

    abs_x_prev = np.abs(x_prev)
    F_x_prev = abs_x_prev - np.log(2.0) + np.log1p(np.exp(-2.0 * abs_x_prev))
    
    # 5. Calculate the difference delta_x = x_curr - x_prev
    delta_x = x_curr - x_prev # dtype will be same as x_curr and x_prev (e.g., float64)

    # 6. Apply the 1st order ADAA formula
    # Initialize the output array with the same shape and dtype as x_curr (e.g., float64)
    distorted_intermediate = np.zeros_like(x_curr)

    # Create a mask for samples where delta_x is exactly zero
    # Floating point comparisons for exact zero can be tricky, but this is standard for ADAA.
    mask_delta_zero = (delta_x == 0.0)
    
    # Case 1: delta_x == 0
    # y_adaa[n] = f(x_curr[n]) = tanh(x_curr[n])
    # np.tanh on float64 input gives float64 output.
    distorted_intermediate[mask_delta_zero] = np.tanh(x_curr[mask_delta_zero])

    # Case 2: delta_x != 0
    # y_adaa[n] = (F(x_curr[n]) - F(x_prev[n])) / delta_x[n]
    mask_delta_nonzero = ~mask_delta_zero # Invert the mask for non-zero delta_x
    
    # Calculate numerator and denominator for the non-zero delta_x case.
    # These will be float64 arrays if intermediate precision is float64.
    numerator = F_x_curr[mask_delta_nonzero] - F_x_prev[mask_delta_nonzero]
    denominator = delta_x[mask_delta_nonzero]
    
    # Perform the division.
    # The mask_delta_nonzero ensures that denominator elements are not exactly zero.
    distorted_intermediate[mask_delta_nonzero] = numerator / denominator

    # Cast the final result back to the original dtype of `samples` (e.g., float32)
    distorted = distorted_intermediate.astype(samples.dtype)
    
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

