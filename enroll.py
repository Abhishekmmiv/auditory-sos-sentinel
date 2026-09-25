# enroll.py
import sys
import json
import time
import numpy as np
import sounddevice as sd
import torch
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn
from speechbrain.inference.speaker import EncoderClassifier

import config
from dsp_utils import trim_silence_energy, unit_normalize, extract_mel_spectrogram

console = Console()

class SafeWordEnroller:
    def __init__(self):
        console.print("[yellow]Initializing embedding backbone (ECAPA-TDNN)...[/yellow]")
        self.encoder = EncoderClassifier.from_hparams(
            source=config.EMBEDDING_MODEL_SOURCE,
            savedir=config.EMBEDDING_SAVEDIR,
            run_opts={"device": "cpu"}
        )
        console.print("[green]✓ Backbone loaded.[/green]\n")

    def record_utterance(self, iteration: int, total: int) -> np.ndarray:
        console.print(Panel(f"[bold cyan]Recording Pass {iteration}/{total}[/bold cyan]\n"
                            f"[dim]Speak your safe-word clearly after the countdown...[/dim]"))
        
        for countdown in reversed(range(1, 4)):
            console.print(f"[bold red]{countdown}...[/bold red]", end="\r")
            time.sleep(0.6)
            
        console.print("[bold green]● RECORDING NOW[/bold green]")
        
        audio = sd.rec(
            int(config.RECORDING_DURATION * config.SAMPLE_RATE),
            samplerate=config.SAMPLE_RATE,
            channels=config.CHANNELS,
            dtype=config.DTYPE
        )
        
        with Progress(TextColumn("[progress.description]{task.description}"), BarColumn()) as progress:
            task = progress.add_task("Capturing...", total=100)
            while not progress.finished:
                progress.update(task, advance=2)
                time.sleep(config.RECORDING_DURATION / 50.0)
                
        sd.wait()
        console.print("[dim]✓ Utterance captured[/dim]\n")
        return audio.flatten()

    def run_enrollment(self, passes: int = 4):
        embeddings = []
        specs = []
        
        console.print(Panel.fit(
            "[bold white]Auditory SOS Sentinel: Safe-Word Enrollment[/bold white]\n"
            "[dim]Choose a distinct 2-4 word phrase (e.g., 'Blue Horizon', 'Protocol Omega')[/dim]",
            border_style="blue"
        ))
        
        input("Press [ENTER] to start enrollment...")
        
        for i in range(1, passes + 1):
            while True:
                audio_data = self.record_utterance(i, passes)
                trimmed = trim_silence_energy(audio_data)
                
                if len(trimmed) < config.SAMPLE_RATE * 0.4:
                    console.print("[bold red]Audio too short. Please speak clearly.[/bold red]")
                    continue

                # 1. Neural Embedding
                tensor_audio = torch.from_numpy(trimmed).unsqueeze(0)
                with torch.no_grad():
                    emb = self.encoder.encode_batch(tensor_audio).squeeze().cpu().numpy()
                embeddings.append(unit_normalize(emb))

                # 2. Phonetic Spectrogram
                spec = extract_mel_spectrogram(trimmed)
                specs.append(spec)
                break

        # Compute neural centroid
        centroid = unit_normalize(np.mean(np.array(embeddings), axis=0))
        np.save(config.CENTROID_PATH, centroid)

        # Average and save the phonetic spectrogram template
        min_width = min(s.shape[1] for s in specs)
        aligned_specs = [s[:, :min_width] for s in specs]
        avg_spec = np.mean(aligned_specs, axis=0)
        np.save(config.PROFILES_DIR / "safe_word_spec.npy", avg_spec)

        console.print(f"[bold green]✓ 2-Factor Profile generated & saved to profiles/[/bold green]\n")

if __name__ == "__main__":
    try:
        enroller = SafeWordEnroller()
        enroller.run_enrollment(passes=4)
    except KeyboardInterrupt:
        console.print("\n[red]Enrollment aborted.[/red]")
        sys.exit(0)