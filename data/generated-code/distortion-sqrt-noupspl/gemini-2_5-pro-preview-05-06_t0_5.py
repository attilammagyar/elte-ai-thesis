# challenge='distortion-sqrt-noupspl', generator='gemini-2.5-pro-preview-05-06', temperature=0.5
# --- BEGIN GENERATED CODE ---

import numpy as np
from scipy import signal

def distort(samples, gain_db=12.0, sample_rate=44100.0, channels=2):
    """
    Applies gain increase and f(x)=x/sqrt(1+x^2) non-linearity to an audio signal
    while keeping aliasing to a minimum without oversampling.

    Parameters:
    samples (np.ndarray): Input audio samples of shape (N, channels), float32.
    gain_db (float): Gain to be applied in dB.
    sample_rate (float): Sampling rate of the audio signal.
    channels (int): Number of audio channels (inferred from samples.shape if possible).

    Returns:
    np.ndarray: Distorted audio samples of the same shape and dtype as input.
    """

    # 1. Convert gain_db to linear scale
    # gain_lin = 10^(gain_db / 20)
    gain_linear = 10.0**(gain_db / 20.0)

    # 2. Apply gain to the samples
    # samples is expected to be a NumPy array of shape (N, channels)
    # This operation is vectorized.
    gained_samples = samples * gain_linear

    # 3. Apply the non-linear distortion function f(x) = x / sqrt(1 + x^2)
    # This function is a soft clipper, output is bounded between -1 and 1.
    # The operation 1.0 + gained_samples**2 ensures the argument to sqrt is always >= 1.0,
    # so np.sqrt will not produce NaNs or warnings for typical float inputs.
    # This operation is vectorized and preserves dtype if inputs are floats.
    distorted_nonlinear = gained_samples / np.sqrt(1.0 + gained_samples**2)

    # 4. Apply an anti-aliasing filter (low-pass filter)
    # This is done post-distortion to remove high-frequency components generated
    # by the non-linearity, which could include aliased frequencies.

    # Filter design parameters:
    # Using a 2nd order Butterworth filter as a good balance between
    # effectiveness (decent rolloff) and computational cost.
    filter_order = 2

    # Cutoff frequency for the low-pass filter.
    # Nyquist frequency is half the sample rate.
    nyquist = sample_rate / 2.0
    
    # Set the cutoff frequency to a fraction of Nyquist.
    # A common choice is between 0.8 to 0.95 of Nyquist.
    # E.g., for 44.1kHz sample rate, Nyquist = 22050 Hz.
    # 0.85 * Nyquist = 18742.5 Hz.
    # This helps to attenuate frequencies near Nyquist that might have been
    # created by the distortion and could alias back into the audible spectrum.
    # This value can be tuned depending on desired aggressiveness of filtering.
    cutoff_ratio = 0.85 
    cutoff_freq = nyquist * cutoff_ratio

    # Ensure cutoff_freq is strictly less than Nyquist for stable filter design.
    # This is guaranteed if cutoff_ratio < 1.0 and nyquist > 0.
    # If cutoff_freq were to equal or exceed nyquist, signal.butter might behave
    # unexpectedly or create an ineffective filter. Our current setup is safe.
    # We must also ensure cutoff_freq > 0. If sample_rate is pathologically low,
    # this could be an issue, but typical audio sample rates are fine.
    if cutoff_freq <= 0:
        # If cutoff is zero or negative (e.g. sample_rate is zero or negative, or extremely low)
        # filtering is not meaningful or possible. Return the undistorted signal or handle error.
        # For simplicity here, we'll pass through the non-linear signal if filtering isn't feasible.
        # However, sample_rate is expected to be a positive audio rate.
        # A more robust implementation might raise an error or log a warning.
        # Given the problem constraints, assume valid sample_rate.
        pass


    # Design the Butterworth filter coefficients (numerator `b` and denominator `a`).
    # `fs=sample_rate` specifies that `cutoff_freq` is in Hz.
    # `output='ba'` provides coefficients for use with `lfilter`.
    # For higher orders or critical applications, `output='sos'` (second-order sections)
    # with `sosfilt` is generally more numerically stable, but for 2nd order, 'ba' is often fine.
    b, a = signal.butter(filter_order, cutoff_freq, btype='low', analog=False, fs=sample_rate, output='ba')

    # Apply the filter to each channel of the audio.
    # `distorted_nonlinear` has shape (N, channels).
    # `signal.lfilter` applies the filter along `axis=0` (the time dimension) for each channel.
    # If `distorted_nonlinear` is float32, `lfilter` typically uses float64 for computation
    # and returns a float64 array.
    if distorted_nonlinear.shape[0] == 0: # Handle empty input array
        distorted_filtered = np.array([], dtype=distorted_nonlinear.dtype).reshape(0, distorted_nonlinear.shape[1])
    else:
        distorted_filtered = signal.lfilter(b, a, distorted_nonlinear, axis=0)
    
    # Ensure the output array has the same data type as the input `samples` array (float32).
    distorted = distorted_filtered.astype(samples.dtype)
    
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

