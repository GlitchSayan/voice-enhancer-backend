import numpy as np
from scipy.signal import stft, istft, iirpeak, lfilter
import soundfile as sf
from pydub import AudioSegment
import os

# --------------------------
#   LOAD ANY AUDIO FORMAT
# --------------------------
def load_audio_any_format(path):
    """
    Loads ANY audio (mp3, m4a, wav, flac, ogg, aac...) using pydub,
    converts internally to WAV and returns PCM numpy array + samplerate.
    """

    # Convert using pydub
    audio = AudioSegment.from_file(path)
    wav_path = path + "_temp.wav"
    audio.export(wav_path, format="wav")

    # Read WAV safely
    data, sr = sf.read(wav_path)

    # Cleanup temp file
    os.remove(wav_path)

    # Convert stereo → mono
    if len(data.shape) == 2:
        data = data.mean(axis=1)

    return data.astype(np.float32), sr


# --------------------------
#       SOFT ANC
# --------------------------
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


# --------------------------
#       COMPRESSOR
# --------------------------
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

        # smooth gain
        if target_gain < gain:
            gain -= attack * (gain - target_gain)
        else:
            gain += release * (target_gain - gain)

        out[i] = sample * gain

    return out


# --------------------------
#       EQ BOOST
# --------------------------
def apply_eq(audio, sr):
    def boost(freq, Q, gain_db):
        gain = 10 ** (gain_db / 20)
        w0 = freq / (sr / 2)
        b, a = iirpeak(w0, Q)
        return lfilter(b, a, audio) * gain

    out = audio.copy()
    out += boost(250, 2, 1.5)   # Warmth
    out += boost(3000, 2, 2.0)  # Clarity
    return out


# --------------------------
#       FULL PIPELINE
# --------------------------
def process_audio(audio, sr):
    anc = soft_anc(audio, sr)
    compressed = smooth_compress(anc)
    enhanced = apply_eq(compressed, sr)
    return enhanced


# --------------------------
#   FILE-LEVEL API WRAPPER
# --------------------------
def process_audio_file(in_path, out_path):
    """
    Safe processing of ANY audio file uploaded via API.
    """

    # Load ANY audio format using pydub
    audio, sr = load_audio_any_format(in_path)

    # DSP pipeline
    enhanced = process_audio(audio, sr)

    # Export final WAV
    sf.write(out_path, enhanced, sr)
