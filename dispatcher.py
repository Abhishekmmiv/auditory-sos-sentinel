# dispatcher.py
import time
import math
import threading
import requests
import numpy as np
from pathlib import Path
from scipy.io import wavfile
from rich.console import Console

import config

console = Console()

def haversine_distance_meters(lat1, lon1, lat2, lon2) -> float:
    """Calculates geodesic distance between two GPS coordinates in meters."""
    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

class EmergencyDispatcher:
    def __init__(self):
        self.evidence_dir = config.EVIDENCE_DIR
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.tracking_active = False
        self.tracker_thread = None
        self.last_coordinates = (0.0, 0.0)

    def _get_active_recipients(self) -> list[str]:
        settings = config.load_settings()
        contacts = settings.get("contacts", [])
        active = [str(c["chat_id"]).strip() for c in contacts if c.get("enabled", True) and str(c.get("chat_id", "")).strip()]
        return active

    def _get_bot_token(self) -> str:
        settings = config.load_settings()
        return str(settings.get("telegram_bot_token", "")).strip()

    def get_realtime_location(self) -> dict:
        """Fetches live public IP/triangulated coordinates."""
        try:
            res = requests.get("https://ipwho.is/", timeout=3.5)
            if res.status_code == 200:
                data = res.json()
                if data.get("success", False):
                    lat = float(data.get("latitude", 0.0))
                    lon = float(data.get("longitude", 0.0))
                    city = data.get("city", "Unknown")
                    region = data.get("region", "")
                    return {
                        "valid": True,
                        "lat": lat,
                        "lon": lon,
                        "city": f"{city}, {region}",
                        "maps_url": f"https://www.google.com/maps?q={lat},{lon}"
                    }
        except Exception as e:
            console.print(f"[dim yellow]Location lookup notice: {e}[/dim yellow]")
            
        return {
            "valid": False,
            "lat": 0.0,
            "lon": 0.0,
            "city": "Location Unavailable",
            "maps_url": "https://maps.google.com"
        }

    def save_audio_evidence(self, audio_buffer: np.ndarray) -> Path:
        timestamp = int(time.time())
        file_path = self.evidence_dir / f"sos_capture_{timestamp}.wav"
        scaled_audio = np.int16(np.clip(audio_buffer, -1.0, 1.0) * 32767)
        wavfile.write(file_path, config.SAMPLE_RATE, scaled_audio)
        return file_path

    def _send_telegram_broadcast(self, text: str, audio_file: Path = None):
        """Broadcasts text and optional audio file to all enabled emergency contacts."""
        token = self._get_bot_token()
        recipients = self._get_active_recipients()

        console.print(f"[dim]Dispatching to {len(recipients)} recipient(s)...[/dim]")

        if not token:
            console.print("[bold red]❌ Dispatch failed: 'telegram_bot_token' is empty in settings.json![/bold red]")
            return
            
        if not recipients:
            console.print("[bold red]❌ Dispatch failed: No enabled contacts found in settings.json! Add contacts via Web UI.[/bold red]")
            return

        for chat_id in recipients:
            console.print(f"[cyan]Sending alert to Chat ID: {chat_id}...[/cyan]")
            
            # 1. Send Text Message
            try:
                msg_res = requests.post(
                    f"https://api.telegram.org/bot{token}/sendMessage",
                    json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
                    timeout=10.0
                )
                if msg_res.status_code == 200:
                    console.print(f"[green]✓ Text alert delivered to {chat_id}[/green]")
                else:
                    console.print(f"[bold red]❌ Telegram text error ({msg_res.status_code}): {msg_res.text}[/bold red]")
            except Exception as e:
                console.print(f"[bold red]❌ Network error sending text to {chat_id}: {e}[/bold red]")

            # 2. Send Audio (if provided)
            if audio_file and audio_file.exists():
                try:
                    with open(audio_file, "rb") as fp:
                        audio_res = requests.post(
                            f"https://api.telegram.org/bot{token}/sendAudio",
                            data={"chat_id": chat_id, "caption": "🎙 SOS Incident Audio Recording"},
                            files={"audio": (audio_file.name, fp, "audio/wav")},
                            timeout=15.0
                        )
                        if audio_res.status_code == 200:
                            console.print(f"[green]✓ Audio evidence delivered to {chat_id}[/green]")
                        else:
                            console.print(f"[bold red]❌ Telegram audio error ({audio_res.status_code}): {audio_res.text}[/bold red]")
                except Exception as e:
                    console.print(f"[bold red]❌ Network error sending audio to {chat_id}: {e}[/bold red]")

    def dispatch_initial_sos(self, trigger_type: str, metric_val: float, audio_buffer: np.ndarray):
        """Initial SOS alert trigger: notifies contacts and starts continuous tracking."""
        console.print("[bold red]⚡ EXECUTING INITIAL SOS BROADCAST...[/bold red]")
        
        loc = self.get_realtime_location()
        self.last_coordinates = (loc["lat"], loc["lon"])
        audio_file = self.save_audio_evidence(audio_buffer)

        sos_text = (
            f"🚨 <b>EMERGENCY DISTRESS ALERT TRIGGERED</b> 🚨\n\n"
            f"<b>Trigger Type:</b> {trigger_type.upper()}\n"
            f"<b>Confidence:</b> {metric_val:.3f}\n"
            f"<b>Initial Location:</b> {loc['city']}\n"
            f"<b>Coordinates:</b> <code>{loc['lat']}, {loc['lon']}</code>\n\n"
            f"📍 <b><a href='{loc['maps_url']}'>Open Initial Location in Maps</a></b>\n"
            f"🛰 <i>Live dynamic tracking is now ACTIVE. Updates will follow if movement is detected.</i>"
        )
        self._send_telegram_broadcast(sos_text, audio_file)
        
        # Start Live Movement Tracking Thread
        self.start_live_tracking()

    def start_live_tracking(self):
        """Launches continuous background location polling."""
        if self.tracking_active:
            return
            
        self.tracking_active = True
        self.tracker_thread = threading.Thread(target=self._tracking_worker, daemon=True)
        self.tracker_thread.start()
        console.print("[bold cyan]🛰 Dynamic GPS breadcrumb tracking thread started.[/bold cyan]")

    def stop_live_tracking(self):
        """Stops location breadcrumb updates."""
        if self.tracking_active:
            self.tracking_active = False
            self._send_telegram_broadcast("🟢 <b>UPDATE: User has marked themselves as SAFE. Live tracking stopped.</b>")
            console.print("[green]Live tracking stopped.[/green]")

    def _tracking_worker(self):
        """Background daemon: checks position every N seconds and pings if moved."""
        while self.tracking_active:
            settings = config.load_settings()
            interval = settings.get("tracking_interval_sec", 15)
            min_move = settings.get("min_movement_threshold_m", 25.0)

            time.sleep(interval)
            
            if not self.tracking_active:
                break

            loc = self.get_realtime_location()
            if not loc["valid"]:
                continue

            last_lat, last_lon = self.last_coordinates
            dist_moved = haversine_distance_meters(last_lat, last_lon, loc["lat"], loc["lon"])

            if dist_moved >= min_move:
                self.last_coordinates = (loc["lat"], loc["lon"])
                update_text = (
                    f"🚗 <b>LIVE MOVEMENT ALERT: Victim is Moving</b>\n\n"
                    f"<b>Relocated:</b> ~{int(dist_moved)} meters from last ping\n"
                    f"<b>Current Area:</b> {loc['city']}\n"
                    f"<b>New Coordinates:</b> <code>{loc['lat']}, {loc['lon']}</code>\n\n"
                    f"📍 <b><a href='{loc['maps_url']}'>Open Live Updated Position</a></b>\n"
                    f"⏱ <i>{time.strftime('%H:%M:%S UTC')}</i>"
                )
                self._send_telegram_broadcast(update_text)
                console.print(f"[cyan]Relocation ping sent (Moved {dist_moved:.1f}m)[/cyan]")