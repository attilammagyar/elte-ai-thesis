# challenge='distortion-sqrt-adaa', generator='gemini-2.5-pro-preview-05-06', temperature=0.5
# --- BEGIN GENERATED CODE ---

import numpy as np

def distort(samples, gain_db=12.0, sample_rate=44100.0, channels=2):
    """
    Applies gain increase and f(x)=x/sqrt(1+x^2) non-linearity to an audio signal
    using 1st order antiderivative anti-aliasing (ADAA).

    Parameters:
    samples (np.ndarray): Input audio signal of shape (N, channels), float32.
    gain_db (float): Gain to apply in decibels.
    sample_rate (float): Sampling rate of the audio signal. (Not used in this 1st order ADAA implementation)
    channels (int): Number of audio channels. (Can also be inferred from samples.shape[1])

    Returns:
    np.ndarray: Distorted audio signal of the same shape and dtype as samples.
    """

    # Ensure gain_db is float32 for calculations, to maintain precision consistency.
    gain_db_f32 = np.float32(gain_db)

    # Handle N=0 case: if samples array is empty (no frames), 
    # return an empty array of the same shape and type.
    if samples.shape[0] == 0:
        return np.zeros_like(samples, dtype=np.float32)

    # 1. Convert gain_db to a linear gain factor
    gain_linear = np.float32(10.0)**(gain_db_f32 / np.float32(20.0))

    # 2. Apply gain to input samples
    # Input 'samples' are float32 as per problem specification.
    # The result 'x_gained' will also be float32.
    x_gained = samples * gain_linear

    # 3. Prepare current samples (x_n) and previous samples (x_n-1)
    # x_n is the array of current samples (post-gain).
    x_n = x_gained

    # x_prev is x_n shifted by one sample.
    # For the first sample (at index 0), x_prev[0, c] is initialized to 0.0 for all channels c.
    # This choice implies that the signal effectively starts from a state of 0.0 
    # for the purpose of calculating the first derivative approximation.
    x_prev = np.zeros_like(x_n, dtype=np.float32)
    if samples.shape[0] > 1:  # Only attempt to shift if there's more than one sample frame.
        # x_prev[1:] correctly handles multi-channel arrays: it's equivalent to x_prev[1:, :]
        # This assigns x_n[0] to x_prev[1], x_n[1] to x_prev[2], and so on for each channel.
        x_prev[1:] = x_n[:-1]
    # If samples.shape[0] is 1, x_prev remains all zeros.
    # Then delta_x[0] will be x_n[0] - 0.0. This is handled correctly by the logic below.

    # 4. Define the non-linear function f(x) and its antiderivative F(x)
    # The non-linear function is f(x) = x / sqrt(1 + x^2).
    # Its antiderivative (indefinite integral) is F(x) = sqrt(1 + x^2) (+ C, constant C cancels out).
    # These helper functions are designed to operate element-wise on NumPy arrays.
    # Constants are specified as np.float32 to ensure float32 arithmetic.

    def f_x_vectorized(val_array):
        # For val_array = 0, this correctly returns 0.
        # The denominator sqrt(1 + val_array^2) is always >= 1.0, so no division by zero from sqrt part.
        return val_array / np.sqrt(np.float32(1.0) + val_array**2)

    def F_x_vectorized(val_array):
        # For val_array = 0, this correctly returns sqrt(1) = 1.
        return np.sqrt(np.float32(1.0) + val_array**2)

    # 5. Calculate the difference in input samples: delta_x = x_n - x_prev
    delta_x = x_n - x_prev

    # 6. Apply 1st order Antiderivative Anti-Aliasing (ADAA)
    # The 1st order ADAA formula is:
    #   y_adaa[n] = (F(x_n) - F(x_prev)) / (x_n - x_prev),  if x_n != x_prev
    #   y_adaa[n] = f(x_n),                                if x_n == x_prev (or delta_x is very small)

    # Calculate F(x_n) and F(x_prev) for the numerator of the ADAA formula
    F_of_xn = F_x_vectorized(x_n)
    F_of_xprev = F_x_vectorized(x_prev)
    numerator_adaa = F_of_xn - F_of_xprev

    # Calculate f(x_n) for the cases where delta_x is (close to) zero
    fx_n_for_small_delta_case = f_x_vectorized(x_n)

    # Define a small epsilon for comparing delta_x to zero in floating point arithmetic.
    # This handles cases where delta_x should ideally be zero but isn't due to precision,
    # or when delta_x is so small that direct division in the ADAA formula might lead to precision loss or instability.
    epsilon = np.float32(1e-8)  # A common threshold for float32 comparisons.

    # Initialize the output array for the distorted signal.
    # It will be filled based on the conditions for delta_x.
    distorted_output = np.zeros_like(x_n, dtype=np.float32)

    # Identify indices where delta_x is effectively zero (i.e., its absolute value is less than epsilon).
    condition_delta_is_small = np.abs(delta_x) < epsilon

    # Case 1: delta_x is (close to) zero. Use f(x_n) as the output.
    # This assignment uses boolean array indexing to modify only the relevant elements.
    distorted_output[condition_delta_is_small] = fx_n_for_small_delta_case[condition_delta_is_small]

    # Case 2: delta_x is not close to zero. Use the full ADAA formula.
    # Identify indices for this case (this is the logical negation of the previous condition).
    condition_delta_is_not_small = ~condition_delta_is_small
    
    # Apply the ADAA formula for these elements.
    # The division is safe here: delta_x[condition_delta_is_not_small] will contain values
    # with magnitudes greater than or equal to epsilon, thus not zero.
    distorted_output[condition_delta_is_not_small] = \
        numerator_adaa[condition_delta_is_not_small] / delta_x[condition_delta_is_not_small]
        
    return distorted_output


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

