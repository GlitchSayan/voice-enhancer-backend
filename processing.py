import numpy as np
from scipy.signal import stft, istft
import soundfile as sf
from scipy.signal import iirpeak, lfilter

# ----- Soft ANC -----
def soft_anc(audio, sr, noise_frames=20, alpha=1.5, beta=0.002):
    f, t, Zxx = stft(audio, fs=sr, nperseg=1024)
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)

    noise_mag = np.mean(mag[:, :noise_frames], axis=1, keepdims=True)
    subtracted = mag - alpha * noise_mag
    subtracted = np.maximum(subtracted, beta * noise_mag)

    Zxx_clean = subtracted * np.exp(1j * phase)
    _, enhanced = istft(Zxx_clean, fs=sr)
    return enhanced


# ----- Compressor -----
def smooth_compress(audio, threshold=0.1, ratio=3, attack=0.01, release=0.1):
    out = np.zeros_like(audio)
    gain = 1.0
    for i, sample in enumerate(audio):
        abs_sample = abs(sample)
        if abs_sample > threshold:
            target_gain = threshold + (abs_sample - threshold) / ratio
            target_gain /= abs_sample
        else:
            target_gain = 1.0

        if target_gain < gain:
            gain = gain - attack * (gain - target_gain)
        else:
            gain = gain + release * (target_gain - gain)

        out[i] = sample * gain
    return out


# ----- EQ -----
def apply_eq(audio, sr):
    def boost(freq, Q, gain_db):
        gain = 10 ** (gain_db / 20)
        w0 = freq / (sr / 2)
        b, a = iirpeak(w0, Q)
        return lfilter(b, a, audio) * gain

    out = audio.copy()
    out += boost(250, 2, 1.5)
    out += boost(3000, 2, 2.0)
    return out


# ----- Pipeline -----
def process_audio(audio, sr):
    anc = soft_anc(audio, sr)
    compressed = smooth_compress(anc)
    enhanced = apply_eq(compressed, sr)
    return enhanced


# ----- File-based wrapper for API -----
def process_audio_file(in_path, out_path):
    audio, sr = sf.read(in_path)

    if len(audio.shape) == 2:
        audio = audio.mean(axis=1)

    enhanced = process_audio(audio, sr)
    sf.write(out_path, enhanced, sr)
