# challenge='distortion-sqrt-noupspl', generator='gemini-2.5-pro-preview-05-06', temperature=1.0
# --- BEGIN GENERATED CODE ---

import numpy as np
from scipy.signal import butter, sosfilt

def distort(samples, gain_db=12.0, sample_rate=44100.0, channels=2):
    """
    Applies gain increase and f(x)=x/sqrt(1+x^2) non-linearity to an audio signal
    while keeping aliasing to a minimum without oversampling.

    Parameters:
    samples (np.ndarray): Input audio samples of shape (N, channels), float32.
    gain_db (float): Gain to apply in decibels.
    sample_rate (float): Sampling rate of the audio signal.
    channels (int): Number of audio channels (inferred from samples.shape if possible).

    Returns:
    np.ndarray: Distorted audio samples of the same shape and dtype as input.
    """

    # The 'distorted' variable is initialized in the problem's template.
    # We will calculate the result and assign it to a variable, then return it.
    # If the template implies modifying a pre-allocated array, that's a minor adjustment.
    # For now, assume standard NumPy practice of returning a new array.

    # 1. Convert gain from dB to linear
    # Ensure linear_gain is float32 to maintain dtype consistency with samples.
    linear_gain = np.float32(10.0**(gain_db / 20.0))

    # 2. Apply gain
    # samples (float32) * linear_gain (float32) results in float32
    x_gained = samples * linear_gain

    # 3. Anti-aliasing: Low-pass filter the gained signal BEFORE non-linearity
    # This helps to remove high-frequency content that could alias after distortion.
    
    # Design a 2nd order Butterworth low-pass filter.
    # Cutoff frequency: chosen as a compromise, e.g., 0.8 * Nyquist frequency.
    # Nyquist frequency = sample_rate / 2.0
    # Wn_cutoff is the normalized cutoff frequency for scipy.signal.butter:
    # Wn_cutoff = (desired_cutoff_hz) / (sample_rate / 2.0)
    # Here, Wn_cutoff = 0.8 means cutoff is at 0.8 * Nyquist.
    filter_order = 2 
    Wn_cutoff = np.float32(0.8) # Using 0.8 as a general-purpose value.

    # Check if sample_rate is high enough for the chosen Wn. Wn must be < 1.
    # Wn_cutoff = 0.8 is always valid.
    
    # Create filter coefficients (Second-Order Sections format)
    # `butter` typically returns float64 coefficients.
    sos = butter(filter_order, Wn_cutoff, btype='low', analog=False, output='sos')

    # Apply the filter along axis 0 (time axis for each channel).
    # `sosfilt` preserves float32 dtype if input is float32.
    # Check if there are enough samples for filtering.
    # sos.shape[0] is the number of second-order sections.
    # Minimal length for sosfilt is related to filter order.
    # If samples.shape[0] is very small, sosfilt might produce largely transient output or warn/error.
    # Assuming N (samples.shape[0]) is generally sufficient for typical audio buffers.
    if samples.shape[0] == 0:
        # If input is empty, return an empty array matching input structure.
        # This also handles the `distorted = np.zeros_like(samples)` template line correctly.
        return np.zeros_like(samples)
    
    # For very short, non-empty samples, filtering might still be applied.
    # `sosfilt` will handle short sequences, though the result might be mostly filter startup transient.
    x_filtered = sosfilt(sos, x_gained, axis=0)

    # 4. Apply the non-linear distortion function: f(x) = x / sqrt(1 + x^2)
    # Use np.hypot(1.0, x) for sqrt(1 + x^2) for better numerical stability.
    # Ensure the constant 1.0 is float32 to keep calculations in float32.
    denominator = np.hypot(np.float32(1.0), x_filtered)
    
    # np.hypot(1.0, x) is always >= 1.0, so no division by zero.
    distorted_signal = x_filtered / denominator

    # Ensure the output dtype matches the input 'samples' dtype (float32).
    # The operations chosen (float32 inputs, specific constant dtypes)
    # should result in float32 output. An explicit cast can be added if necessary.
    # distorted_signal = distorted_signal.astype(samples.dtype)

    return distorted_signal


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

