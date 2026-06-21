from __future__ import annotations

import io
import wave
from pathlib import Path
import miniaudio
import numpy as np
from scipy.fft import dct
from scipy.signal import resample_poly

TARGET_SAMPLE_RATE = 16_000
N_FFT, FRAME_LENGTH, HOP_LENGTH, N_MELS, N_MFCC = 512, 400, 160, 40, 20


def _read_wav(path: str | Path, target_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    with wave.open(io.BytesIO(Path(path).read_bytes()), "rb") as source:
        channels, width, rate = source.getnchannels(), source.getsampwidth(), source.getframerate()
        raw = source.readframes(source.getnframes())
    if width == 1:
        signal = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        signal = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    elif width == 4:
        signal = np.frombuffer(raw, "<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"不支持的 WAV 位深：{width * 8} bit")
    if channels > 1:
        signal = signal.reshape(-1, channels).mean(axis=1)
    if not len(signal):
        raise ValueError("音频内容为空")
    if rate != target_rate:
        divisor = int(np.gcd(rate, target_rate))
        signal = resample_poly(signal, target_rate // divisor, rate // divisor).astype(np.float32)
    peak = float(np.max(np.abs(signal)))
    return signal / peak if peak > 1e-8 else signal


def _read_audio(path: str | Path, target_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    candidate = Path(path)
    if candidate.suffix.lower() == ".wav":
        return _read_wav(candidate, target_rate)
    if candidate.suffix.lower() == ".mp3":
        decoded = miniaudio.decode_file(
            str(candidate), output_format=miniaudio.SampleFormat.FLOAT32,
            nchannels=1, sample_rate=target_rate
        )
        signal = np.frombuffer(decoded.samples, dtype=np.float32).copy()
        if not len(signal):
            raise ValueError("音频内容为空")
        peak = float(np.max(np.abs(signal)))
        return signal / peak if peak > 1e-8 else signal
    raise ValueError("仅支持 WAV 或 MP3 音频")


def _mel_filterbank(sample_rate: int) -> np.ndarray:
    hz_to_mel = lambda hz: 2595.0 * np.log10(1.0 + hz / 700.0)
    mel_to_hz = lambda mel: 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    points = np.linspace(hz_to_mel(20.0), hz_to_mel(sample_rate / 2), N_MELS + 2)
    bins = np.floor((N_FFT + 1) * mel_to_hz(points) / sample_rate).astype(int)
    bank = np.zeros((N_MELS, N_FFT // 2 + 1), dtype=np.float32)
    for index in range(N_MELS):
        left, center, right = bins[index:index + 3]
        center, right = max(center, left + 1), max(right, center + 1)
        bank[index, left:center] = np.arange(left, center) / (center - left)
        bank[index, center:right] = (right - np.arange(center, right)) / (right - center)
    return bank


_MEL_BANK = _mel_filterbank(TARGET_SAMPLE_RATE)


def extract_audio_features(path: str | Path) -> np.ndarray:
    """Extract fixed-length MFCC, spectral, energy and duration statistics."""
    signal = _read_audio(path)
    if len(signal) < FRAME_LENGTH:
        signal = np.pad(signal, (0, FRAME_LENGTH - len(signal)))
    count = 1 + (len(signal) - FRAME_LENGTH) // HOP_LENGTH
    indices = np.arange(FRAME_LENGTH)[None, :] + HOP_LENGTH * np.arange(count)[:, None]
    frames = signal[indices]
    spectrum = np.abs(np.fft.rfft(frames * np.hanning(FRAME_LENGTH)[None, :], n=N_FFT, axis=1))
    power = spectrum ** 2 / N_FFT
    log_mel = np.log(np.maximum(power @ _MEL_BANK.T, 1e-10))
    mfcc = dct(log_mel, type=2, axis=1, norm="ortho")[:, :N_MFCC]
    delta = np.diff(mfcc, axis=0, prepend=mfcc[:1])
    zcr = np.mean(frames[:, 1:] * frames[:, :-1] < 0, axis=1, keepdims=True)
    rms = np.sqrt(np.mean(frames ** 2, axis=1, keepdims=True) + 1e-10)
    frequencies = np.fft.rfftfreq(N_FFT, 1.0 / TARGET_SAMPLE_RATE)
    magnitude_sum = np.maximum(spectrum.sum(axis=1, keepdims=True), 1e-10)
    centroid = ((spectrum * frequencies).sum(axis=1, keepdims=True) / magnitude_sum) / (TARGET_SAMPLE_RATE / 2)
    cumulative = np.cumsum(power, axis=1)
    rolloff = (cumulative < cumulative[:, -1:] * .85).sum(axis=1, keepdims=True) / (N_FFT // 2)
    descriptors = np.concatenate([mfcc, delta, zcr, rms, centroid, rolloff], axis=1)
    stats = np.concatenate([descriptors.mean(0), descriptors.std(0),
                            np.percentile(descriptors, 10, axis=0),
                            np.percentile(descriptors, 90, axis=0)])
    return np.concatenate([stats.astype(np.float32), [len(signal) / TARGET_SAMPLE_RATE]]).astype(np.float32)




def select_speaker_reduced_features(features: np.ndarray) -> np.ndarray:
    """Suppress coefficients dominated by speaker/channel identity.

    The full vector contains mean, standard deviation, 10th/90th percentiles
    for 44 frame descriptors plus duration. We drop coefficient 0 from every
    block and replace absolute percentiles with their range.
    """
    values = np.asarray(features)
    if values.shape[-1] != 177:
        raise ValueError(f"语音特征维度应为 177，实际为 {values.shape[-1]}")
    return np.concatenate([
        values[..., 1:44],
        values[..., 45:88],
        values[..., 133:176] - values[..., 89:132],
    ], axis=-1).astype(np.float32)
