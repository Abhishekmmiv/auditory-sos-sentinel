# dsp_utils.py
import numpy as np
import torch
import torchaudio.transforms as T

# Mel Spectrogram extractor for phonetic trajectory matching
mel_transform = T.MelSpectrogram(
    sample_rate=16000,
    n_fft=512,
    win_length=480,
    hop_length=160,     # 10ms hop for high temporal resolution
    n_mels=40
)

def extract_mel_spectrogram(audio: np.ndarray) -> np.ndarray:
    """Extracts Log-Mel Spectrogram matrix normalized to zero-mean unit-variance."""
    if len(audio) == 0:
        return np.zeros((40, 10))
    tensor_audio = torch.from_numpy(audio).float()
    spec = mel_transform(tensor_audio)
    log_spec = torch.log(spec + 1e-6).numpy()
    
    # Standardize
    mean = np.mean(log_spec)
    std = np.std(log_spec) + 1e-8
    return (log_spec - mean) / std

def spectral_phonetic_distance(candidate_spec: np.ndarray, template_spec: np.ndarray) -> float:
    """
    Computes 2D normalized cross-correlation between the candidate audio segment
    and the enrolled safe-word spectrogram template.
    Returns similarity between 0.0 and 1.0.
    """
    c_h, c_w = candidate_spec.shape
    t_h, t_w = template_spec.shape
    
    # Resize / interpolate candidate time-axis to match template length
    if c_w != t_w:
        from scipy.ndimage import zoom
        zoom_factor = t_w / max(1, c_w)
        candidate_spec = zoom(candidate_spec, (1, zoom_factor), order=1)
        candidate_spec = candidate_spec[:, :t_w]

    # Compute 2D Pearson Correlation Coefficient
    cand_flat = candidate_spec.flatten()
    temp_flat = template_spec.flatten()
    
    c_norm = cand_flat - np.mean(cand_flat)
    t_norm = temp_flat - np.mean(temp_flat)
    
    denom = np.linalg.norm(c_norm) * np.linalg.norm(t_norm)
    if denom < 1e-6:
        return 0.0
        
    correlation = float(np.dot(c_norm, t_norm) / denom)
    return max(0.0, correlation)

def trim_silence_energy(audio: np.ndarray, threshold_ratio: float = 0.05) -> np.ndarray:
    energy = np.abs(audio)
    threshold = np.max(energy) * threshold_ratio
    active_indices = np.where(energy > threshold)[0]
    
    if len(active_indices) == 0:
        return audio
    
    start_idx = max(0, active_indices[0] - 800)
    end_idx = min(len(audio), active_indices[-1] + 800)
    return audio[start_idx:end_idx]

def unit_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm