# challenge='distortion-adaa', generator='gemini-2.5-pro-preview-05-06', temperature=0.5
# --- BEGIN GENERATED CODE ---

import numpy as np

# Helper function for numerically stable log(cosh(x))
# log(cosh(x)) = log((e^x + e^-x)/2)
# For large |x|, cosh(x) approx e^|x|/2, so log(cosh(x)) approx |x| - log(2)
# The numerically stable formula is: |x| + log(1 + exp(-2*|x|)) - log(2)
# np.log1p(y) computes log(1+y) accurately for small y.
def _log_cosh_stable(x_input):
    abs_x = np.abs(x_input)
    # np.exp(-2 * abs_x) can underflow to 0 for large abs_x. np.log1p(0) is 0. This is correct.
    # The constant np.log(2.0) is float64. If x_input is float32, calculations involving
    # this constant will promote the result to float64. This enhances precision.
    return abs_x + np.log1p(np.exp(-2 * abs_x)) - np.log(2.0)

def distort(samples, gain_db=12.0, sample_rate=44100.0, channels=2):
    # sample_rate and channels parameters are part of the function signature as requested.
    # The number of channels is implicitly handled by NumPy's operations on the input array shape.
    # sample_rate is not directly used in this specific 1st order ADAA formulation for tanh.
    
    # Initialize output array with the same shape and dtype as input samples.
    # The problem states "samples are float32 numbers", so distorted will be float32.
    distorted = np.zeros_like(samples)

    # 1. Convert gain from dB to linear.
    # Standard Python floats (64-bit) are used for 10.0 and 20.0, so gain_linear will be float64.
    gain_linear = 10.0**(gain_db / 20.0)

    # 2. Apply gain to input samples.
    # If `samples` is float32 and `gain_linear` is float64, `x` will be promoted to float64.
    # This is generally beneficial for the precision of intermediate calculations.
    x = samples * gain_linear

    # 3. Prepare previous samples (x_prev).
    # `x_prev` will have the same dtype as `x` (i.e., float64 in this setup).
    # Initialize with zeros; for the first sample of each channel, the previous sample is effectively 0.
    x_prev = np.zeros_like(x) 
    
    # Only perform the shift if there's more than one sample frame.
    # If x.shape[0] is 0 or 1, x_prev remains all zeros.
    # This correctly handles the first sample (index 0) or single-sample inputs.
    if x.shape[0] > 1:
        x_prev[1:, :] = x[:-1, :]

    # 4. Calculate dx = x_n - x_{n-1}.
    # `dx` will have the same dtype as `x` (float64).
    dx = x - x_prev

    # 5. Define epsilon for comparing dx against zero.
    # The choice of epsilon depends on the precision of `x`.
    if x.dtype == np.float32: # This branch is unlikely if x is promoted by float64 gain_linear
        epsilon = np.float32(1e-6)
    elif x.dtype == np.float64:
        epsilon = np.float64(1e-9) # A common epsilon for float64 comparisons
    else: 
        # Fallback for unexpected dtypes. Assumes float64-like precision.
        epsilon = 1e-9 

    # 6. Apply 1st order Antiderivative Anti-Aliasing (ADAA) formula.
    # The non-linear function is f(x) = tanh(x).
    # Its antiderivative is F(x) = log(cosh(x)).
    # ADAA output y_n:
    #   if |dx| > epsilon:  y_n = (F(x_n) - F(x_{n-1})) / dx
    #   if |dx| <= epsilon: y_n = f(x_n) = tanh(x_n)

    # Create boolean masks to identify where dx is non-zero or zero (approximately).
    mask_dx_nonzero = np.abs(dx) > epsilon
    mask_dx_zero = ~mask_dx_nonzero # Equivalent to np.abs(dx) <= epsilon

    # Case 1: dx is significantly non-zero.
    # All calculations in this branch will use the precision of x (float64).
    if np.any(mask_dx_nonzero):
        # Extract slices corresponding to non-zero dx.
        x_nz = x[mask_dx_nonzero]
        x_prev_nz = x_prev[mask_dx_nonzero]
        dx_nz = dx[mask_dx_nonzero]

        # Calculate antiderivative values using the stable helper function.
        # These will be float64 if x_nz, x_prev_nz are float64.
        F_x_n = _log_cosh_stable(x_nz)
        F_x_prev = _log_cosh_stable(x_prev_nz)
        
        # Compute the ADAA output for these elements. Result is float64.
        # Assignment to `distorted` (float32) will perform downcasting.
        distorted[mask_dx_nonzero] = (F_x_n - F_x_prev) / dx_nz
    
    # Case 2: dx is zero or very close to zero.
    # Calculations use precision of x (float64).
    if np.any(mask_dx_zero):
        # Extract slice corresponding to zero dx.
        x_z = x[mask_dx_zero]
        
        # Apply the original non-linearity f(x_n) = tanh(x_n). Result is float64.
        # Assignment to `distorted` (float32) will perform downcasting.
        distorted[mask_dx_zero] = np.tanh(x_z)
        
    # The `distorted` array was initialized with `samples.dtype` (float32).
    # Values assigned (possibly from float64 calculations) are cast to float32.
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

