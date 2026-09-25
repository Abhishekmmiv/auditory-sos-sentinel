# distress_detector.py
import numpy as np
from scipy.signal import spectrogram

class AcousticDistressDetector:
    def __init__(self, sample_rate: int = 16000):
        self.sr = sample_rate
        # Tunable distress thresholds
        self.min_rms_energy = 0.08        # Must be loud
        self.min_f0_pitch = 450.0         # Hz (vocal fold hyper-extension)
        self.min_spectral_centroid = 2200 # Hz (high-frequency energy shift)
        self.min_spectral_flatness = 0.15 # Noise/turbulence threshold

    def _estimate_f0_autocorr(self, frame: np.ndarray) -> float:
        """Estimates Fundamental Frequency (F0) using normalized autocorrelation."""
        if len(frame) == 0 or np.max(np.abs(frame)) < 1e-4:
            return 0.0
            
        # Standard human vocal pitch range: 80Hz - 1200Hz
        min_lag = int(self.sr / 1200)
        max_lag = int(self.sr / 80)
        
        # Autocorrelation via FFT
        n = len(frame)
        fft_frame = np.fft.rfft(frame, n=2*n)
        autocorr = np.fft.irfft(fft_frame * np.conj(fft_frame))[:n]
        
        # Normalize
        if autocorr[0] <= 0:
            return 0.0
        autocorr = autocorr / autocorr[0]
        
        # Search for peak in candidate lag window
        candidate_lags = autocorr[min_lag:max_lag]
        if len(candidate_lags) == 0:
            return 0.0
            
        peak_idx = np.argmax(candidate_lags) + min_lag
        if autocorr[peak_idx] > 0.35: # Sufficient periodicity
            return float(self.sr / peak_idx)
        return 0.0

    def analyze(self, audio_buffer: np.ndarray) -> dict:
        """
        Runs acoustic feature extraction on the window to determine distress state.
        Returns metrics and a boolean trigger flag.
        """
        rms_energy = float(np.sqrt(np.mean(audio_buffer**2)))
        
        # Gate: If audio is quiet, skip heavy spectral compute
        if rms_energy < self.min_rms_energy:
            return {
                "distress_detected": False,
                "rms_energy": rms_energy,
                "f0_pitch": 0.0,
                "spectral_centroid": 0.0,
                "reason": "energy_too_low"
            }

        # 1. Pitch Extraction (F0)
        f0 = self._estimate_f0_autocorr(audio_buffer)

        # 2. Spectral Centroid & Spectral Flatness
        freqs, _, spec = spectrogram(
            audio_buffer, 
            fs=self.sr, 
            nperseg=512, 
            noverlap=256, 
            mode='magnitude'
        )
        spec_mean = np.mean(spec, axis=1) + 1e-10
        
        # Spectral Centroid: center of mass of spectrum
        spectral_centroid = float(np.sum(freqs * spec_mean) / np.sum(spec_mean))
        
        # Spectral Flatness: geometric mean / arithmetic mean (measure of harsh noise)
        geom_mean = np.exp(np.mean(np.log(spec_mean)))
        arith_mean = np.mean(spec_mean)
        flatness = float(geom_mean / arith_mean)

        # Decision Logic: Scream / Vocal Distress criteria
        distress_score = 0
        if f0 >= self.min_f0_pitch:
            distress_score += 2
        if spectral_centroid >= self.min_spectral_centroid:
            distress_score += 1
        if flatness >= self.min_spectral_flatness:
            distress_score += 1

        # Distress triggered if high pitch + high energy OR high centroid + noise + energy
        distress_detected = (distress_score >= 2) and (rms_energy >= self.min_rms_energy)

        return {
            "distress_detected": distress_detected,
            "score": distress_score,
            "rms_energy": rms_energy,
            "f0_pitch": f0,
            "spectral_centroid": spectral_centroid,
            "spectral_flatness": flatness
        }