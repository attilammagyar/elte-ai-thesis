# challenge='distortion-sqrt-adaa', generator='gemini-2.5-pro-preview-05-06', temperature=0.3
# --- BEGIN GENERATED CODE ---

import numpy as np

def distort(samples, gain_db=12.0, sample_rate=44100.0, channels=2): # sample_rate and channels are part of signature, but effectively inferred or unused.
    """
    Applies gain increase and f(x)=x/sqrt(1+x^2) non-linearity to an audio signal
    using 1st order antiderivative anti-aliasing (ADAA).
    """
    
    # Determine the data type from input samples (e.g., float32)
    dtype = samples.dtype

    # Handle the edge case of empty input samples
    if samples.shape[0] == 0:
        return np.zeros_like(samples) # Returns an empty array with correct shape and dtype

    # 1. Convert gain from dB to a linear factor
    # Cast to the samples' dtype to maintain precision throughout calculations.
    linear_gain = np.array(10**(gain_db / 20.0), dtype=dtype)

    # 2. Apply gain to the input samples
    # x_g represents the gain-applied signal x_g[n]
    x_g = samples * linear_gain

    # Define a constant '1.0' with the target dtype for use in f_nonlinear and F_antiderivative
    # This ensures that operations like (const_one + x_val**2) maintain the original dtype.
    const_one = np.array(1.0, dtype=dtype)

    # 3. Define the non-linear function f(x) and its antiderivative F(x)
    # These functions are designed to operate element-wise on NumPy arrays.
    def f_nonlinear(x_val):
        # Non-linear function: f(x) = x / sqrt(1 + x^2)
        # This function is well-behaved for real x_val since 1 + x_val^2 >= 1.
        return x_val / np.sqrt(const_one + x_val**2)

    def F_antiderivative(x_val):
        # Antiderivative of f(x): F(x) = sqrt(1 + x^2)
        return np.sqrt(const_one + x_val**2)

    # 4. Prepare the array of previous gain-applied samples (x_g[n-1])
    # Initialize x_g_prev_all with zeros. For the first sample (n=0), x_g[n-1] (i.e., x_g[-1]) is taken as 0.
    x_g_prev_all = np.zeros_like(x_g) # Shape and dtype match x_g
    # For n > 0, x_g_prev_all[n, :] = x_g[n-1, :]
    if x_g.shape[0] > 1: # Check if there's more than one sample frame
        x_g_prev_all[1:, :] = x_g[:-1, :]
    
    # 5. Calculate differences required for the ADAA formula
    # dx = x_g[n] - x_g[n-1]
    dx = x_g - x_g_prev_all

    # dF = F(x_g[n]) - F(x_g[n-1])
    F_x_g = F_antiderivative(x_g)
    F_x_g_prev = F_antiderivative(x_g_prev_all)
    dF = F_x_g - F_x_g_prev
    
    # 6. Initialize the output array for distorted samples
    distorted = np.zeros_like(samples, dtype=dtype) # Shape and dtype match input samples

    # 7. Apply the 1st order ADAA formula
    # The formula depends on whether dx is zero or non-zero.

    # Create boolean masks to identify where dx is zero and non-zero
    mask_dx_is_zero = (dx == 0)
    mask_dx_not_zero = ~mask_dx_is_zero # Equivalent to (dx != 0)

    # Case 1: dx != 0
    # y_ADAA[n] = (F(x_g[n]) - F(x_g[n-1])) / (x_g[n] - x_g[n-1]) = dF[n] / dx[n]
    # Perform this calculation only for elements where dx is not zero to avoid division by zero.
    if np.any(mask_dx_not_zero):
        # Extract relevant dF and dx values using the mask
        relevant_dF = dF[mask_dx_not_zero]
        relevant_dx = dx[mask_dx_not_zero]
        # Perform the division and assign to the corresponding elements in 'distorted'
        distorted[mask_dx_not_zero] = relevant_dF / relevant_dx

    # Case 2: dx == 0
    # y_ADAA[n] = f(x_g[n]) (the limit of the ADAA formula as dx -> 0)
    # Perform this calculation only for elements where dx is zero.
    if np.any(mask_dx_is_zero):
        # Extract relevant x_g values using the mask
        relevant_x_g = x_g[mask_dx_is_zero]
        # Apply f_nonlinear and assign to the corresponding elements in 'distorted'
        distorted[mask_dx_is_zero] = f_nonlinear(relevant_x_g)
        
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

