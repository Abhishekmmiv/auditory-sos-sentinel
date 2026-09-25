# Auditory SOS Sentinel
> **Edge-AI Continuous Acoustic Sentinel & Dynamic Emergency Dispatch Mesh**

Auditory SOS Sentinel is an autonomous, hands-free personal safety platform designed to monitor ambient audio in real time for user-defined covert safe-words and acoustic distress signatures (screaming, high-energy panic bursts).

Traditional Personal Emergency Response Systems (PERS) rely on tactile triggers (pressing smartphone buttons, tapping screens, or pulling hardware pins). In violent confrontations, physical restraint, or sudden medical distress, victims rarely possess the physical agency to operate a device. Auditory SOS Sentinel eliminates manual interaction by continuously processing audio streams locally, evaluating vocal biometrics, providing an abortable fail-safe countdown, and dispatching multi-contact alerts with live dynamic tracking.

---

## System Architecture

[ Ambient Microphone Audio @ 16 kHz Mono ] │ ▼ [ Audio Ingestion & VAD Gate ] │
(Energy < 0.02 ──► Drop / Sleep) ▼ (Active Speech / Surge)
┌──────────────┴──────────────┐ ▼ ▼
[ Two-Factor Safe-Word Engine ] [ Acoustic Distress Detector ] •
Biometric: 192-D ECAPA-TDNN • Autocorrelation Pitch (F0 > 450 Hz) • Phonetic: 2D
Log-Mel Template • Spectral Centroid (> 2200 Hz) └──────────────┬──────────────┘
│ (Trigger Event) ▼ [ Non-Blocking 10s Fail-Safe Engine ] • Audible Warning &
Visual Terminal Countdown • Single-Key Asynchronous Abort ([ENTER]) │
(Uncancelled / Timeout) ▼ [ Adaptive Emergency Dispatch Mesh ] • 5.0s Incident
Audio Snapshot (.wav) • High-Precision Coordinate Triangulation (Apple
CoreLocation / Wi-Fi) • Multi-Recipient Telegram Broadcast • Dynamic Movement
Tracking (Haversine Distance > 25m Pings)


---

## Core Capabilities

### 1. Two-Factor Safe-Word Spotting (Zero-Retraining KWS)
Standard keyword spotters require thousands of samples to train acoustic models. This system enables arbitrary custom phrases in four enrollment passes using a dual-verification barrier:
- **Biometric Layer:** An ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation Time-Delay Neural Network) backbone extracts a 192-dimensional unit-normalized speaker vector to verify the user's vocal tract geometry.
- **Phonetic Layer:** A 40-band Log-Mel Filterbank spectrogram template captures syllable cadence via 2D Pearson Cross-Correlation, ensuring the phrase cannot be falsely triggered by conversational speech from the same user.

### 2. Heuristic Distress & Scream Detection
Operates independently of linguistic content to capture physiological panic acoustics:
- **Fundamental Frequency ($F_0$):** Spikes above normal speech registers ($450\text{ Hz} - 1100\text{ Hz}$) via autocorrelation.
- **Spectral Centroid:** Tracks high-frequency energy shifts above $2200\text{ Hz}$.
- **Spectral Flatness (Wiener Entropy):** Detects turbulent, chaotic vocal fold breakdown ($> 0.15$).

### 3. Fail-Safe Verification Window
Every acoustic trigger initiates an asynchronous 10-second cancel countdown. Users can cancel false positives by pressing `[ENTER]`. If uncancelled, the incident is confirmed and live dispatch begins.

### 4. Dynamic Breadcrumb Relocation Tracking
Instead of sending a single static GPS coordinate:
- Captures a rolling 5.0-second `.wav` file ($2.5\text{s}$ pre-trigger $+ 2.5\text{s}$ post-trigger).
- Pulls high-precision coordinates using native macOS CoreLocation (Wi-Fi BSSID triangulation).
- Broadcasts the incident package to all configured guardians via the Telegram Bot API.
- A background tracking thread calculates the Haversine geodesic displacement ($\Delta d$). If the victim is moved more than $25\text{ meters}$ (e.g., inside a vehicle), updated pins and routing links are dispatched automatically.

---

## Project Structure

```text
sentinel_mark1/
├── config.py                 # Central configurations, paths, and model hyperparameters
├── dispatcher.py             # Telegram broadcasting, CoreLocation & Haversine tracking
├── distress_detector.py      # Non-linear DSP scream/distress analysis
├── dsp_utils.py              # Log-Mel transforms, VAD, and normalization helpers
├── enroll.py                 # Interactive 4-pass safe-word enrollment routine
├── live_sentinel.py          # Real-time multi-scale streaming listening loop
├── web_server.py             # FastAPI browser control dashboard
├── requirements.txt          # Python runtime dependencies
├── settings.example.json     # Configuration template for credentials and thresholds
└── profiles/                 # Stored biometric centroids and phonetic templates

Setup & Installation

Prerequisites

  - Python 3.10 or higher
  - macOS (recommended for hardware CoreLocation support) or Linux
  - Working microphone input (headset or built-in array)

1. Clone & Environment Setup

git clone https://github.com/Abhishekmmiv/auditory-sos-sentinel.git
cd auditory-sos-sentinel

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

2. Install Dependencies

pip install --upgrade pip
pip install -r requirements.txt
pip install fastapi uvicorn requests pyobjc-framework-CoreLocation

3. Configure Credentials

Create settings.json from the provided template:

cp settings.example.json settings.json

Populate settings.json with your credentials:

{
    "telegram_bot_token": "YOUR_TELEGRAM_BOT_TOKEN",
    "contacts": [
        {
            "name": "Primary Guardian",
            "chat_id": "YOUR_CHAT_ID",
            "enabled": true
        }
    ],
    "tracking_interval_sec": 15,
    "min_movement_threshold_m": 25.0,
    "neural_threshold": 0.65,
    "phonetic_threshold": 0.42
}

4. Enroll Safe-Word Profile

Run the interactive enrollment routine to capture your baseline phrase:

python enroll.py

Follow the countdown prompts and speak your chosen phrase four times at a
consistent volume. The script will validate intra-cluster similarity and save
safe_word_centroid.npy and safe_word_spec.npy to the profiles/ directory.

5. Running the System

Run the web management console and the Sentinel detector in separate terminal
tabs:

Terminal 1 — Web Control Center:

python web_server.py

Open http://127.0.0.1:8000 to view profile status, update tracking intervals, or
add/remove emergency contacts.

Terminal 2 — Live Sentinel Listener:

python live_sentinel.py

The Sentinel will calibrate against the ambient noise floor and begin live
acoustic monitoring.

Hardware Migration Roadmap (Major Project Phase)

The current Python implementation serves as the architectural reference model
for an ultra-low-power wearable implementation planned for the next engineering
phase:

| Metric                | Host Prototype (Current)      | Wearable Target (Next Phase)                         |
| :-------------------- | :---------------------------- | :--------------------------------------------------- |
| **Platform**          | macOS / Linux Host            | Nordic Semiconductor nRF52840 SoC                    |
| **Compute Profile**   | FP32 PyTorch / SpeechBrain    | INT8 Quantized 1D-CNN (TFLite-Micro / CMSIS-NN)      |
| **RAM Footprint**     | \~24 MB                       | \< 35 KB                                             |
| **Audio Capture**     | Host Audio Subsystem (16 kHz) | Onboard PDM MEMS Microphone (MSM261)                 |
| **Power Profile**     | Standard Laptop Battery       | \< 5 µA Deep Sleep (Months on CR2032 / 150 mAh LiPo) |
| **Off-Grid Tracking** | Host Wi-Fi CoreLocation API   | Apple Find My / OpenHaystack Offline BLE Mesh        |

