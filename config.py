# config.py
import json
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent
PROFILES_DIR = ROOT_DIR / "profiles"
EVIDENCE_DIR = ROOT_DIR / "distress_evidence"
SETTINGS_FILE = ROOT_DIR / "settings.json"

PROFILES_DIR.mkdir(parents=True, exist_ok=True)
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

CENTROID_PATH = PROFILES_DIR / "safe_word_centroid.npy"
SPEC_PATH = PROFILES_DIR / "safe_word_spec.npy"
METRICS_PATH = PROFILES_DIR / "safe_word_metrics.json"

# Audio Capture Specifications
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "float32"
RECORDING_DURATION = 2.5
WINDOW_SIZE_SEC = 1.5

# Neural Model Backbone
EMBEDDING_MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
EMBEDDING_SAVEDIR = str(ROOT_DIR / "pretrained_models/spkrec-ecapa-voxceleb")

# Default Configuration Template
DEFAULT_SETTINGS = {
    "telegram_bot_token": "8896966256:AAFdRWGKC9BzOdR0tcIIngiK_0e9uSanmSo",
    "contacts": [
        {"name": "Primary Guardian", "chat_id": "8337386648", "enabled": True}
    ],
    "tracking_interval_sec": 15,     # Send GPS ping every 15s when moving
    "min_movement_threshold_m": 25.0,# Only update if moved > 25 meters
    "neural_threshold": 0.65,
    "phonetic_threshold": 0.42
}

def load_settings() -> dict:
    if not SETTINGS_FILE.exists():
        save_settings(DEFAULT_SETTINGS)
        return DEFAULT_SETTINGS
    with open(SETTINGS_FILE, "r") as f:
        return json.load(f)

def save_settings(settings: dict):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=4)