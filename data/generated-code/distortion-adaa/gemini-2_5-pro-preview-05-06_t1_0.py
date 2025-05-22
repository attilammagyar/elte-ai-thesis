# challenge='distortion-adaa', generator='gemini-2.5-pro-preview-05-06', temperature=1.0
# --- BEGIN GENERATED CODE ---

import numpy as np

def distort(samples, gain_db=12.0, sample_rate=44100.0, channels=2):
    # Constants for float32 operations
    # Using np.log(2.0) (float64) and casting to np.float32 for LOG2_F32
    LOG2_F32 = np.float32(np.log(2.0))
    # Epsilon for checking if delta_x is close to zero. Using a small absolute value.
    EPS_F32 = np.float32(1e-9)

    # --- Helper functions (operating on float32 NumPy arrays) ---
    
    # Non-linearity: f(u) = tanh(u)
    def f_tanh_f32(u_f32):
        return np.tanh(u_f32)

    # Antiderivative of tanh: F(u) = log(cosh(u))
    # Numerically stable version: F(u) = abs(u) + log1p(exp(-2*abs(u))) - log(2)
    def F_logcosh_stable_f32(u_f32):
        abs_u_f32 = np.abs(u_f32)
        # Intermediate calculations should maintain float32 precision
        exp_arg = np.float32(-2.0) * abs_u_f32
        exp_term = np.exp(exp_arg)
        log1p_term = np.log1p(exp_term)
        return abs_u_f32 + log1p_term - LOG2_F32

    # --- Main 1st Order ADAA Distortion Logic ---

    # Handle empty input: return an empty array of the same shape and type
    if samples.shape[0] == 0:
        return np.zeros_like(samples, dtype=np.float32)

    # 1. Apply linear gain to input samples
    # Ensure gain_linear is float32 for consistent arithmetic with float32 samples
    gain_linear = np.float32(10.0**(gain_db / 20.0))
    # 'samples' is specified as float32.
    x_gained = samples * gain_linear

    # 2. Get previous gained sample values (x_gained_prev)
    # For the first sample (n=0), its "previous" value (x_gained_prev[0]) is taken as 0.0.
    x_gained_prev = np.zeros_like(x_gained, dtype=np.float32)
    if samples.shape[0] > 1: # If there's more than one sample in the array
        x_gained_prev[1:, :] = x_gained[:-1, :] # x_prev[n] = x[n-1] for n > 0

    # 3. Calculate difference: delta_x_gained[n] = x_gained[n] - x_gained_prev[n]
    delta_x_gained = x_gained - x_gained_prev

    # 4. Initialize the output array for distorted samples
    distorted_samples = np.zeros_like(samples, dtype=np.float32)

    # 5. Create a mask for elements where delta_x_gained is (close to) zero
    zero_delta_mask = np.abs(delta_x_gained) < EPS_F32

    # 6. Case 1: delta_x_gained is (close to) zero
    # Apply direct nonlinearity: y[n] = f(x_gained[n])
    # Compute f_tanh_f32 for all elements, then selectively copy using the mask.
    # This avoids indexing into x_gained with a mask before function application.
    direct_eval_f = f_tanh_f32(x_gained)
    np.copyto(distorted_samples, direct_eval_f, where=zero_delta_mask)

    # 7. Case 2: delta_x_gained is not zero
    # Apply ADAA formula: y[n] = (F(x_gained[n]) - F(x_gained_prev[n])) / delta_x_gained[n]
    non_zero_delta_mask = ~zero_delta_mask
    
    # Proceed with ADAA calculation only if there are elements that fall into this case
    if np.any(non_zero_delta_mask):
        # Calculate antiderivative F for current and previous samples
        F_x = F_logcosh_stable_f32(x_gained)
        F_x_prev = F_logcosh_stable_f32(x_gained_prev)
        
        numerator_adaa = F_x - F_x_prev
        
        # Temporary array to store results of ADAA division.
        # Values in this array where non_zero_delta_mask is False are undefined/irrelevant
        # as they won't be copied to the final output.
        adaa_eval_results = np.empty_like(distorted_samples, dtype=np.float32)
        
        # Perform division safely, only for elements where non_zero_delta_mask is true.
        # The 'out' argument receives results; 'where' controls which elements are computed.
        np.divide(numerator_adaa, delta_x_gained, out=adaa_eval_results, where=non_zero_delta_mask)
        
        # Copy the computed ADAA results to the final output array for relevant elements.
        np.copyto(distorted_samples, adaa_eval_results, where=non_zero_delta_mask)

    return distorted_samples


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

