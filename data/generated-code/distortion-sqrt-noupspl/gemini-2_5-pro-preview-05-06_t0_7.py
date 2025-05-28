# challenge='distortion-sqrt-noupspl', generator='gemini-2.5-pro-preview-05-06', temperature=0.7
# --- BEGIN GENERATED CODE ---

import numpy as np

def distort(samples, gain_db=12.0, sample_rate=44100.0, channels=2): # sample_rate and channels args are not strictly used by this ADAA method but are part of the signature.
    """
    Applies gain increase and f(x)=x/sqrt(1+x^2) non-linearity to an audio signal
    while keeping aliasing to a minimum without oversampling, using 1st order ADAA.
    """

    # Handle empty input array: if N=0.
    # samples.shape[0] would be number of frames N.
    # samples.shape[1] would be number of channels based on input.
    if samples.shape[0] == 0:
        return samples.astype(np.float32) # Return empty array with correct dtype

    # 1. Convert gain from dB to linear scale and apply to samples
    linear_gain = 10.0**(gain_db / 20.0)
    
    # x will be promoted to float64 if samples is float32 and linear_gain is float64 (default for 10.0**...),
    # which is beneficial for precision during intermediate calculations.
    x = samples * linear_gain 

    # 2. Antiderivative Antialiasing (ADAA) - 1st order
    # The non-linear function is f(x) = x / sqrt(1 + x^2).
    # Its first antiderivative (integral) is F(x) = sqrt(1 + x^2).

    # Prepare x_prev = x_{n-1} (previous sample array)
    # For the first sample (n=0), x_prev[0] is set to x[0]. This makes dx[0] = 0.
    # For subsequent samples (n>0), x_prev[n] = x[n-1].
    x_prev = np.empty_like(x)
    x_prev[0, :] = x[0, :]  # Handles the boundary condition for the first sample.
    if x.shape[0] > 1: # Check if there's more than one sample to avoid issues with x[:-1,:] on single sample arrays
        x_prev[1:, :] = x[:-1, :] # Vectorized way to set x_prev[n,c] = x[n-1,c] for n > 0.
                                  # If x.shape[0] == 1, this line is skipped. x_prev[1:,:] would be an empty slice.
    
    # Calculate dx = x_n - x_{n-1}
    # For the first sample, dx[0,:] will be x[0,:] - x_prev[0,:] = x[0,:] - x[0,:] = 0.
    dx = x - x_prev

    # Calculate F(x_n) and F(x_{n-1}) using the antiderivative F(val) = sqrt(1 + val^2).
    # np.sqrt(1.0 + arr**2) is always well-defined as 1.0 + arr**2 >= 1.0.
    Fx_n = np.sqrt(1.0 + x**2)
    Fx_n_minus_1 = np.sqrt(1.0 + x_prev**2) # Due to x_prev[0,:], Fx_n_minus_1[0,:] = F(x[0,:]).

    # Calculate dF = F(x_n) - F(x_{n-1})
    # For the first sample, dF[0,:] will be F(x[0,:]) - F(x[0,:]) = 0.
    dF = Fx_n - Fx_n_minus_1

    # Define a small epsilon for numerically stable comparison of dx with zero.
    epsilon = 1e-9 # A common choice. Adjust if specific precision issues arise.

    # Case 1: Direct application of the non-linear function f(x_n).
    # This is used when dx is (close to) zero, as lim (dF/dx) as dx->0 is f(x_n).
    fx_n_direct = x / np.sqrt(1.0 + x**2)

    # Case 2: ADAA formula (F(x_n) - F(x_{n-1})) / (x_n - x_{n-1})
    # This is dF/dx. This term is used when dx is not close to zero.
    # np.divide is used implicitly by '/', can warn on 0/0; np.where avoids using this result if dx is small.
    
    # Use np.where to select between fx_n_direct and dF/dx based on dx.
    # If abs(dx) < epsilon, dx is considered effectively zero, use fx_n_direct.
    # Otherwise, use the ADAA estimate dF/dx.
    distorted_samples = np.where(
        np.abs(dx) < epsilon,
        fx_n_direct,
        dF / dx
    )

    # Ensure the output array has the float32 dtype as specified.
    return distorted_samples.astype(np.float32)


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

