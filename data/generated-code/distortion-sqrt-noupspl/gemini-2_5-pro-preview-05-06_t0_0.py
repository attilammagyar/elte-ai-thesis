# challenge='distortion-sqrt-noupspl', generator='gemini-2.5-pro-preview-05-06', temperature=0.0
# --- BEGIN GENERATED CODE ---

import numpy as np
from scipy import signal

def distort(samples, gain_db=12.0, sample_rate=44100.0, channels=2): # Note: 'channels' param is unused, inferred from samples.shape
    """
    Applies gain increase and f(x)=x/sqrt(1+x^2) non-linearity to an audio signal
    while keeping aliasing to a minimum without oversampling.

    Parameters:
    samples (np.ndarray): Input audio samples of shape (N, channels), float32.
    gain_db (float): Gain to apply in decibels.
    sample_rate (float): Sampling rate of the audio.
    channels (int): Number of audio channels. (Currently inferred from samples.shape)

    Returns:
    np.ndarray: Distorted audio samples, same shape and dtype as input.
    """

    # 1. Convert gain_db to linear gain
    # Using np.float32 to keep calculations in float32 if possible,
    # as input 'samples' are float32.
    gain_linear = np.float32(10.0**(gain_db / 20.0))

    # 2. Apply gain
    # If 'samples' is float32, 'samples_gained' will also be float32.
    samples_gained = samples * gain_linear

    # 3. Antialiasing Pre-filter
    # Design a Butterworth low-pass filter to apply before the non-linearity.
    # This reduces aliasing by attenuating high frequencies that would generate
    # harmonics above the Nyquist frequency after distortion.
    
    nyquist_freq = sample_rate / 2.0
    
    # Filter parameters
    filter_order = 4  # A 4th order Butterworth filter offers a good balance of effectiveness and speed.
    
    # Normalized cutoff frequency (as a fraction of the Nyquist frequency).
    # Example: 0.85 means the cutoff is at 0.85 * Nyquist frequency.
    # This value is a trade-off:
    # - Lower values provide stronger antialiasing but may dull the sound.
    # - Higher values preserve more high frequencies but risk more aliasing.
    # The value must be in the range (0, 1) for scipy.signal.butter.
    normalized_cutoff = 0.85 # A common choice for mild pre-filtering.

    # Proceed with filtering only if sample_rate and cutoff are valid for filter design.
    if nyquist_freq > 0 and 0 < normalized_cutoff < 1:
        # Create Second-Order Sections (SOS) representation of the filter.
        # 'sos' output is preferred for numerical stability over 'ba' (numerator/denominator).
        sos_coefficients = signal.butter(filter_order, 
                                         normalized_cutoff, 
                                         btype='low', 
                                         analog=False, 
                                         output='sos')
        
        # Apply the filter along the time axis (axis 0).
        # 'samples_gained' has shape (N, channels) according to the problem description.
        # sosfilt processes each channel (column) independently when axis=0.
        # The output dtype of sosfilt matches the input dtype (float32 in this case).
        filtered_gained_samples = signal.sosfilt(sos_coefficients, samples_gained, axis=0)
    else:
        # If filtering is not possible (e.g., due to invalid sample_rate or cutoff parameters),
        # pass the gained samples through without filtering.
        # This might lead to aliasing in such edge cases.
        # For typical audio parameters, this branch should not be taken.
        filtered_gained_samples = samples_gained

    # 4. Apply the non-linear distortion function: f(x) = x / sqrt(1 + x^2)
    # Let y be the filtered, gain-boosted samples.
    y = filtered_gained_samples
    
    # Use np.float32(1.0) to ensure the expression (1 + y^2) is computed in float32
    # if y is float32, maintaining the precision of the input samples.
    one_f32 = np.float32(1.0)
    
    # The denominator np.sqrt(one_f32 + y**2) is always >= 1.0 for real y,
    # so division by zero is not a concern here.
    # This operation is applied element-wise thanks to NumPy's vectorization.
    distorted_signal = y / np.sqrt(one_f32 + y**2)

    # The function is expected to return the distorted samples.
    # The dtype should remain float32 if the input 'samples' was float32,
    # due to the consistent use of np.float32 for constants and dtype preservation
    # by SciPy/NumPy operations.
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

