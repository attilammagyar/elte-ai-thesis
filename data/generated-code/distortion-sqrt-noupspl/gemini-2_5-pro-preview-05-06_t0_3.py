# challenge='distortion-sqrt-noupspl', generator='gemini-2.5-pro-preview-05-06', temperature=0.3
# --- BEGIN GENERATED CODE ---

import numpy as np
from scipy import signal

def distort(samples, gain_db=12.0, sample_rate=44100.0, channels=2):
    """
    Applies gain increase and f(x)=x/sqrt(1+x^2) non-linearity to an audio signal
    while keeping aliasing to a minimum without oversampling.

    Parameters:
    samples (np.ndarray): Input audio samples of shape (N, channels), float32.
    gain_db (float): Gain to apply in decibels.
    sample_rate (float): Sampling rate of the audio signal in Hz.
    channels (int): Number of audio channels. (Note: This parameter is implicitly
                     handled by samples.shape in this implementation).

    Returns:
    np.ndarray: Distorted audio samples, same shape and dtype as input.
    """

    # 1. Convert gain from dB to linear scale
    # gain_db = 20 * log10(linear_gain) => linear_gain = 10^(gain_db / 20)
    linear_gain = 10**(gain_db / 20.0)

    # 2. Apply gain to the input samples
    # Input samples are float32. NumPy operations might promote to float64.
    # This is fine for intermediate calculations; will cast back at the end.
    gained_samples = samples * linear_gain

    # 3. Apply the non-linear distortion function: f(x) = x / sqrt(1 + x^2)
    # This function is a form of soft clipping that smoothly saturates towards +/-1.
    # It's numerically stable as (1 + x^2) is always >= 1 for real x.
    # np.errstate can be used to manage warnings if x_gained could be extremely large,
    # but for typical audio float ranges, this should be fine.
    # The denominator is np.sqrt(1.0 + gained_samples**2)
    # The numerator is gained_samples
    distorted_signal = gained_samples / np.sqrt(1.0 + gained_samples**2)

    # 4. Antialiasing: Apply a low-pass filter to the distorted signal.
    # This step is crucial for minimizing aliasing caused by the non-linearity
    # generating harmonics above the Nyquist frequency. Since oversampling is not
    # allowed, a post-distortion filter is a common approach.
    # A 2nd order Butterworth filter offers a good balance between effectiveness
    # (decent rolloff) and computational speed.

    # Define filter parameters:
    filter_order = 2  # Order of the Butterworth filter
    
    # Nyquist frequency is half the sample rate.
    nyquist_freq = sample_rate / 2.0

    # Normalized cutoff frequency (Wn) for the Butterworth filter.
    # Wn is specified as a fraction of the Nyquist frequency (0 < Wn < 1).
    # A common choice is to set the cutoff somewhat below Nyquist to attenuate
    # aliased components effectively.
    # e.g., 0.85 * nyquist_freq means Wn = 0.85.
    # This value provides a trade-off: preserves high-frequency content while
    # reducing aliasing. Too high (e.g., 0.99) is less effective for anti-aliasing.
    # Too low (e.g., 0.5) might make the distortion sound muffled.
    # We must ensure Wn < 1.0 for scipy.signal.butter.
    # Assuming sample_rate > 0, nyquist_freq > 0.
    if nyquist_freq <= 0:
        # This case should ideally not occur with valid inputs.
        # If it does, filtering is not possible or meaningful.
        # Return the distorted signal without filtering, or handle as an error.
        # For robustness, if sample_rate is invalid, skip filtering.
        # However, the problem implies standard audio processing conditions.
        # We'll proceed assuming sample_rate allows for valid nyquist_freq.
        # If nyquist_freq is extremely small, filter design might be ill-conditioned.
        # scipy.signal.butter handles Wn close to 0 or 1, but 0 < Wn < 1 is required.
        pass # Assuming valid sample_rate leads to nyquist_freq > 0.

    # Wn_ratio defines the cutoff relative to Nyquist. E.g. 0.85 means 85% of Nyquist.
    Wn_ratio = 0.85 
    
    # Design the Butterworth low-pass filter coefficients (numerator 'b', denominator 'a')
    # output='ba' is for use with lfilter.
    b_coeffs, a_coeffs = signal.butter(filter_order, Wn_ratio, btype='low', analog=False, output='ba')

    # Apply the filter.
    # The input `samples` (and thus `distorted_signal`) has shape (N, channels).
    # We need to apply the filter independently to each channel along the time axis (N).
    # In NumPy, for an array of shape (N, channels), axis 0 is the time axis.
    filtered_signal = signal.lfilter(b_coeffs, a_coeffs, distorted_signal, axis=0)
    
    # Ensure the output array has the same data type as the input `samples` (float32).
    # `signal.lfilter` typically returns float64 if input is float, as coefficients are float64.
    distorted_output = filtered_signal.astype(samples.dtype)
    
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

