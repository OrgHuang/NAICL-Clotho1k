import os
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter

SR = 16000
DURATION = 1.0
N = int(SR * DURATION)

OUT_DIR = "noise"
os.makedirs(OUT_DIR, exist_ok=True)


def save(wav, name):
    wav = wav / np.max(np.abs(wav) + 1e-8)
    sf.write(os.path.join(OUT_DIR, name), wav, SR)


# 5️⃣ Bubble Noise
def bubble_noise():
    noise = np.random.randn(N)
    mod = np.abs(np.sin(np.cumsum(np.random.randn(N) * 0.01)))
    wav = noise * mod
    save(wav, "bubble_noise_1s.wav")


# 6️⃣ Silence + Device Hum
def silence_device_hum():
    hum_freq = 50  # or 60
    t = np.arange(N) / SR
    hum = 0.002 * np.sin(2 * np.pi * hum_freq * t)
    silence = 0.0005 * np.random.randn(N)
    wav = hum + silence
    save(wav, "silence_device_hum_1s.wav")


# 7️⃣ Pink Noise (1/f)
def pink_noise():
    white = np.random.randn(N)
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(N, 1 / SR)
    fft /= np.sqrt(freqs + 1e-6)
    wav = np.fft.irfft(fft)
    save(wav, "pink_noise_1s.wav")


# 8️⃣ Band-pass Filtered Noise
def bandpass_noise(low=2000, high=3000):
    white = np.random.randn(N)
    b, a = butter(4, [low / (SR / 2), high / (SR / 2)], btype="band")
    wav = lfilter(b, a, white)
    save(wav, "bandpass_noise_1s.wav")


# 9️⃣ Amplitude Modulated Noise
def modulated_noise():
    noise = np.random.randn(N)
    t = np.arange(N) / SR
    mod = 0.5 * (1 + np.sin(2 * np.pi * 4 * t))  # slow AM
    wav = noise * mod
    save(wav, "modulated_noise_1s.wav")


# 🔟 Synthetic Glitch Noise
def glitch_noise():
    wav = np.zeros(N)
    for _ in range(20):
        start = np.random.randint(0, N - 200)
        length = np.random.randint(50, 200)
        wav[start:start + length] += np.random.randn(length) * 0.8
    save(wav, "glitch_noise_1s.wav")


if __name__ == "__main__":
    bubble_noise()
    silence_device_hum()
    pink_noise()
    bandpass_noise()
    modulated_noise()
    glitch_noise()
