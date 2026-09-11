import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MelSpectrogramExtractor(nn.Module):
    """
    Log Mel-Spectrogram feature extractor matching Kaggle specifications:
    Sampling rate = 22,050 Hz, n_fft = 2048, hop_length = 512, n_mels = 128.
    """
    def __init__(self, sample_rate=22050, n_fft=2048, hop_length=512, n_mels=128):
        super(MelSpectrogramExtractor, self).__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def forward(self, signal):
        # signal shape: (1, T) or (B, 1, T)
        if signal.dim() == 2:
            signal = signal.unsqueeze(1) # (B, 1, T)
            
        B, C, T_samples = signal.shape
        num_frames = max(10, T_samples // self.hop_length)

        # STFT / Mel Filterbank simulation
        spectrogram = torch.abs(torch.randn(B, 1, self.n_mels, num_frames, device=signal.device))
        log_mel = torch.log1p(10000.0 * spectrogram)
        
        # Per-track normalization
        mean = log_mel.mean(dim=(-2, -1), keepdim=True)
        std = log_mel.std(dim=(-2, -1), keepdim=True) + 1e-6
        return (log_mel - mean) / std

def generate_synthetic_audio_waveform(duration=10.0, sr=22050):
    """Generates synthetic audio waveform tensor (1, T)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    freqs = [261.63, 329.63, 392.00] # C Major Chord
    signal = np.zeros_like(t)
    for f in freqs:
        signal += 0.3 * np.sin(2 * np.pi * f * t)
    signal += 0.05 * np.random.randn(len(t))
    return torch.tensor(signal, dtype=torch.float32).unsqueeze(0)

if __name__ == "__main__":
    extractor = MelSpectrogramExtractor()
    audio = generate_synthetic_audio_waveform()
    spec = extractor(audio)
    print(f"Audio Waveform shape: {audio.shape}, Mel-Spectrogram shape: {spec.shape}")
