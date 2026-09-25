# live_sentinel.py
import sys
import time
import select
import queue
import numpy as np
import sounddevice as sd
import torch
from rich.console import Console
from rich.panel import Panel
from speechbrain.inference.speaker import EncoderClassifier

import config
from dsp_utils import unit_normalize, extract_mel_spectrogram, spectral_phonetic_distance, trim_silence_energy
from distress_detector import AcousticDistressDetector
from dispatcher import EmergencyDispatcher

console = Console()

class LiveSentinel:
    def __init__(self):
        # Two-Factor Thresholds
        self.neural_threshold = 0.65       # Biometric Voice Match
        self.phonetic_threshold = 0.42     # Syllable/Phonetic Match
        
        self.sample_rate = config.SAMPLE_RATE
        self.buffer_len = int(config.WINDOW_SIZE_SEC * self.sample_rate) # 1.5s detection window
        self.audio_ring_buffer = np.zeros(self.buffer_len, dtype=np.float32)
        
        # Extended Rolling Evidence Buffer (Stores last 2.5 seconds of context)
        self.evidence_history_len = int(2.5 * self.sample_rate)
        self.evidence_ring_buffer = np.zeros(self.evidence_history_len, dtype=np.float32)

        self.audio_queue = queue.Queue()
        self.is_running = False
        self.alert_active = False
        
        # Load Templates
        spec_path = config.PROFILES_DIR / "safe_word_spec.npy"
        if not (config.CENTROID_PATH.exists() and spec_path.exists()):
            console.print(f"[bold red]Error:[/bold red] Profile files missing. Run 'python enroll.py' first.")
            sys.exit(1)
            
        self.target_centroid = np.load(config.CENTROID_PATH)
        self.target_spec = np.load(spec_path)
        
        # Initialize Telegram Dispatcher
        console.print("[yellow]Connecting Telegram Dispatcher...[/yellow]")
        self.dispatcher = EmergencyDispatcher()
        
        # Initialize Neural Backbone & DSP Engine
        console.print("[yellow]Loading ECAPA-TDNN & DSP Engine...[/yellow]")
        self.encoder = EncoderClassifier.from_hparams(
            source=config.EMBEDDING_MODEL_SOURCE,
            savedir=config.EMBEDDING_SAVEDIR,
            run_opts={"device": "cpu"}
        )
        self.distress_detector = AcousticDistressDetector(sample_rate=self.sample_rate)
        console.print("[green]✓ 2-Factor Sentinel Armed (5-Second Evidence Capture Enabled).[/green]\n")

    def _audio_callback(self, indata, frames, time_info, status):
        if self.is_running:
            self.audio_queue.put(indata.copy())

    def _check_key_press(self) -> bool:
        """Non-blocking keyboard reader for macOS/Linux terminal."""
        dr, _, _ = select.select([sys.stdin], [], [], 0)
        if dr:
            sys.stdin.readline()
            return True
        return False

    def _trigger_emergency_protocol(self, trigger_type: str, metric_val: float):
        self.alert_active = True
        
        # Snapshot the pre-trigger historical audio (2.5 seconds before/during the safe-word)
        pre_trigger_audio = self.evidence_ring_buffer.copy()
        
        sys.stdout.write("\a\n")
        sys.stdout.flush()

        console.print(Panel(
            f"[bold white on red] 🚨 DISTRESS DETECTED ({trigger_type.upper()}) 🚨 [/bold white on red]\n\n"
            f"[bold yellow]Confidence / Pitch:[/bold yellow] {metric_val:.3f}\n"
            f"[bold white]Recording post-trigger audio...[/bold white]\n"
            f"[bold white]You have 10 SECONDS to cancel this alarm.[/bold white]\n"
            f"[bold green]Press [ENTER] to CANCEL...[/bold green]",
            border_style="red"
        ))

        cancelled = False
        timeout = 10.0
        start_time = time.time()
        
        # List to continuously accumulate post-trigger audio during the countdown
        post_trigger_chunks = []

        while (time.time() - start_time) < timeout:
            # Drain queue into post-trigger evidence accumulator
            while not self.audio_queue.empty():
                chunk = self.audio_queue.get_nowait().flatten()
                post_trigger_chunks.append(chunk)

            if self._check_key_press():
                cancelled = True
                break

            remaining = int(np.ceil(timeout - (time.time() - start_time)))
            sys.stdout.write(f"\r\033[K\033[1;31mDispatching in {remaining}s... (Press ENTER to cancel)\033[0m")
            sys.stdout.flush()
            time.sleep(0.1)

        sys.stdout.write("\r\033[K")

        if cancelled:
            console.print("[bold green]✓ ALARM CANCELLED. Resuming Sentinel monitoring...[/bold green]\n")
            # Stop background tracking if it was running
            self.dispatcher.stop_live_tracking()
        else:
            console.print("\n[bold white on red] 🔥 TIMEOUT EXPIRED: EXECUTING LIVE DISPATCH & BREADCRUMB TRACKING! 🔥 [/bold white on red]")
            
            # Assemble the complete 5.0-second incident payload:
            # (2.5 seconds Pre-Trigger + 2.5 seconds Post-Trigger)
            if post_trigger_chunks:
                post_trigger_audio = np.concatenate(post_trigger_chunks)
                # Keep first 2.5 seconds of post-trigger audio for a balanced 5.0s file
                post_trigger_audio = post_trigger_audio[:int(2.5 * self.sample_rate)]
            else:
                post_trigger_audio = np.zeros(int(2.5 * self.sample_rate), dtype=np.float32)

            full_5s_evidence = np.concatenate([pre_trigger_audio, post_trigger_audio])

            # Send Initial SOS to all contacts and launch continuous movement breadcrumbs
            self.dispatcher.dispatch_initial_sos(
                trigger_type=trigger_type, 
                metric_val=metric_val, 
                audio_buffer=full_5s_evidence
            )
            console.print("[dim]Resuming monitoring in 3 seconds...[/dim]\n")
            time.sleep(3.0)

        # Flush queues and reset ring buffers
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()
        self.audio_ring_buffer.fill(0)
        self.evidence_ring_buffer.fill(0)
        time.sleep(1.2)
        
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        self.alert_active = False

    def _evaluate_two_factor(self, buffer: np.ndarray) -> tuple[float, float]:
        """Calculates both Phonetic Spectrogram Match and Neural Biometric Match."""
        trimmed = trim_silence_energy(buffer, threshold_ratio=0.1)
        if len(trimmed) < self.sample_rate * 0.3:
            return 0.0, 0.0

        # 1. Phonetic Spectrogram Match
        cand_spec = extract_mel_spectrogram(trimmed)
        phonetic_sim = spectral_phonetic_distance(cand_spec, self.target_spec)

        # 2. Neural Embedding Biometric Match
        tensor_audio = torch.from_numpy(trimmed).unsqueeze(0)
        with torch.no_grad():
            emb = self.encoder.encode_batch(tensor_audio).squeeze().cpu().numpy()
        neural_sim = float(np.dot(unit_normalize(emb), self.target_centroid))

        return neural_sim, phonetic_sim

    def run(self):
        self.is_running = True
        block_size = int(self.sample_rate * 0.1)
        
        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=config.CHANNELS,
            dtype=config.DTYPE,
            blocksize=block_size,
            callback=self._audio_callback
        )

        console.print(Panel(
            "[bold green]Auditory SOS Sentinel ACTIVE (5-Second Circular Audio Evidence)[/bold green]\n"
            f"[dim]Neural Thresh: {self.neural_threshold:.2f} | Phonetic Thresh: {self.phonetic_threshold:.2f}[/dim]\n"
            "Press [Ctrl+C] to exit.", 
            border_style="green"
        ))

        with stream:
            try:
                while self.is_running:
                    if self.alert_active:
                        time.sleep(0.1)
                        continue

                    while not self.audio_queue.empty():
                        chunk = self.audio_queue.get_nowait().flatten()
                        # Update 1.5s detection ring buffer
                        self.audio_ring_buffer = np.roll(self.audio_ring_buffer, -len(chunk))
                        self.audio_ring_buffer[-len(chunk):] = chunk
                        
                        # Update 2.5s pre-trigger evidence history buffer
                        self.evidence_ring_buffer = np.roll(self.evidence_ring_buffer, -len(chunk))
                        self.evidence_ring_buffer[-len(chunk):] = chunk

                    # VAD Gate
                    rms = float(np.sqrt(np.mean(self.audio_ring_buffer**2)))
                    peak = float(np.max(np.abs(self.audio_ring_buffer)))

                    if rms < 0.02 or peak < 0.05:
                        sys.stdout.write(f"\r\033[K\033[90m[Standby - Quiet]\033[0m RMS: {rms:.3f}")
                        sys.stdout.flush()
                        time.sleep(0.06)
                        continue

                    # 1. Distress Check (Scream)
                    distress_result = self.distress_detector.analyze(self.audio_ring_buffer)
                    if distress_result["distress_detected"]:
                        self._trigger_emergency_protocol("Scream/Distress", distress_result["f0_pitch"])
                        continue

                    # 2. Two-Factor Safe-Word Check
                    neural_sim, phonetic_sim = self._evaluate_two_factor(self.audio_ring_buffer)

                    sys.stdout.write(
                        f"\r\033[K\033[1;37m[Speech]\033[0m Voice: \033[96m{neural_sim:.2f}\033[0m/{self.neural_threshold:.2f} | "
                        f"Phonetic: \033[93m{phonetic_sim:.2f}\033[0m/{self.phonetic_threshold:.2f}"
                    )
                    sys.stdout.flush()

                    if (neural_sim >= self.neural_threshold) and (phonetic_sim >= self.phonetic_threshold):
                        self._trigger_emergency_protocol("Safe-Word Match", (neural_sim + phonetic_sim)/2.0)

                    time.sleep(0.08)

            except KeyboardInterrupt:
                console.print("\n\n[yellow]Shutting down sentinel cleanly.[/yellow]")
                self.is_running = False

if __name__ == "__main__":
    sentinel = LiveSentinel()
    sentinel.run()