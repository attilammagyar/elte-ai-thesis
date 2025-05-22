# challenge='distortion-adaa', generator='gemini-2.5-pro-preview-05-06', temperature=0.7
# --- BEGIN GENERATED CODE ---

import numpy as np

def distort(samples, gain_db=12.0, sample_rate=44100.0, channels=2): # sample_rate and channels parameters are not strictly used by this implementation if samples array defines these.
    
    # Helper function for numerically stable log(cosh(x))
    # Antiderivative of tanh(x) is log(cosh(x)).
    # This function computes log(cosh(x)) stably.
    # Input val_in is expected to be a NumPy array (typically float32 from audio samples).
    # Output is a float64 NumPy array for precision.
    def stable_log_cosh(val_in):
        # Promote to float64 for internal calculations to maintain precision and avoid overflow
        # with intermediate cosh values.
        val_f64 = val_in.astype(np.float64)
        abs_val_f64 = np.abs(val_f64)
        
        # For float64, cosh(x) overflows if x is approximately > 709.78.
        # We use a threshold safely below that to switch to an approximation.
        # For large |x|, cosh(x) approx. e^|x| / 2.
        # So, log(cosh(x)) approx. log(e^|x| / 2) = |x| - log(2).
        threshold = 700.0 
        
        out_f64 = np.zeros_like(val_f64, dtype=np.float64)
        
        # Mask for values where |x| is large enough to use the approximation
        mask_large = abs_val_f64 >= threshold
        # Mask for values where |x| is small enough for direct computation
        mask_small = ~mask_large
        
        # For "small" values, compute log(cosh(x)) directly
        out_f64[mask_small] = np.log(np.cosh(val_f64[mask_small]))
        
        # For "large" values, use the approximation: |x| - log(2)
        out_f64[mask_large] = abs_val_f64[mask_large] - np.log(2.0) # np.log(2.0) is a float64 constant
        
        return out_f64

    # Convert gain from dB to linear scale
    gain_linear = 10.0**(gain_db / 20.0)
    
    # Apply gain to input samples.
    # Input 'samples' are float32, so 'x' will be float32.
    x = samples * gain_linear 
    
    # Prepare x_prev: array of previous samples (x[n-1]).
    # Initialize with zeros. For the first sample of the block (x[0]),
    # its corresponding x_prev[0] will be 0.0, representing an initial state.
    x_prev = np.zeros_like(x, dtype=x.dtype) # x_prev will match x's dtype (float32)
    
    # Populate x_prev by shifting x. x_prev[i] = x[i-1].
    # NumPy's slicing handles edge cases (e.g., empty 'samples' or single sample) correctly.
    if x.shape[0] > 0: # Ensure not attempting to slice an empty dimension
        x_prev[1:] = x[:-1] 
    
    # Calculate dx = x[n] - x[n-1]. This will be float32.
    dx = x - x_prev
    
    # Compute the antiderivative F(x)=log(cosh(x)) for current (x) and previous (x_prev) samples.
    # Results F_x and F_x_prev are float64 arrays due to stable_log_cosh implementation.
    F_x = stable_log_cosh(x)
    F_x_prev = stable_log_cosh(x_prev)
    
    # Initialize the output array for distorted samples.
    # Use float64 for intermediate calculations to preserve precision from F_x, F_x_prev.
    distorted_intermediate = np.zeros_like(x, dtype=np.float64)
    
    # Define a small epsilon to check if dx is effectively zero.
    # This handles the limit case for the ADAA formula.
    # If dx is very small, ADAA formula (F(x) - F(x_prev))/dx approaches f(x) = tanh(x).
    # This threshold is primarily to avoid division by literal zero if x == x_prev.
    epsilon = 1e-8 # An absolute tolerance for floating point comparisons.

    # Create a boolean mask for samples where dx is close to zero.
    mask_dx_is_small = np.abs(dx) < epsilon
    
    # Case 1: dx is small (or zero).
    # Output y[n] = f(x[n]) = tanh(x[n]).
    # We cast x to float64 before np.tanh to maintain float64 pipeline.
    distorted_intermediate[mask_dx_is_small] = np.tanh(x[mask_dx_is_small].astype(np.float64))
    
    # Case 2: dx is not small.
    # Output y[n] = (F(x[n]) - F(x[n-1])) / (x[n] - x[n-1]). This is the 1st order ADAA.
    mask_dx_is_not_small = ~mask_dx_is_small
    
    # Extract dx values for division, corresponding to where dx is not small.
    # dx is float32. (F_x - F_x_prev) is float64. Division promotes to float64.
    # Only perform division where mask_dx_is_not_small is True
    if np.any(mask_dx_is_not_small): # Check if there are any elements to process for this case
        dx_for_division = dx[mask_dx_is_not_small]
        numerator = F_x[mask_dx_is_not_small] - F_x_prev[mask_dx_is_not_small]
        
        distorted_intermediate[mask_dx_is_not_small] = numerator / dx_for_division
    
    # Cast the result back to the original dtype of 'samples' (expected to be float32).
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

