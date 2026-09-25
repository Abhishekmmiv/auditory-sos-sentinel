# web_server.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import subprocess
import os
import signal

import config

app = FastAPI(title="Auditory SOS Sentinel Control Center")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Auditory SOS Sentinel - Control Center</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
</head>
<body class="bg-slate-900 text-slate-100 font-sans min-h-screen">
    <div class="max-w-4xl mx-auto p-6 space-y-8">
        <!-- Header -->
        <header class="flex justify-between items-center border-b border-slate-800 pb-5">
            <div class="flex items-center space-x-3">
                <div class="w-10 h-10 bg-red-600 rounded-xl flex items-center justify-center shadow-lg shadow-red-500/30">
                    <i class="fa-solid fa-shield-halved text-xl text-white"></i>
                </div>
                <div>
                    <h1 class="text-2xl font-bold tracking-tight">Auditory SOS Sentinel</h1>
                    <p class="text-sm text-slate-400">Edge-AI Acoustic Safety Mesh Control Center</p>
                </div>
            </div>
            <div id="status-badge" class="px-3 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-xs font-semibold flex items-center gap-2">
                <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span> Profile Active
            </div>
        </header>

        <!-- Main Grid -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            
            <!-- Safe-Word Profile Manager -->
            <div class="bg-slate-800/60 backdrop-blur border border-slate-700/60 rounded-2xl p-6 space-y-4">
                <div class="flex items-center justify-between">
                    <h2 class="text-lg font-semibold flex items-center gap-2">
                        <i class="fa-solid fa-key text-yellow-400"></i> Safe-Word Profile
                    </h2>
                </div>
                <p class="text-sm text-slate-400">
                    Your personalized safe-word vector profile. You can wipe and re-record it anytime.
                </p>
                <div class="bg-slate-900/80 rounded-xl p-4 border border-slate-700/50 space-y-2 text-sm">
                    <div class="flex justify-between text-slate-400">
                        <span>Profile Status:</span>
                        <span id="profile-exists" class="font-medium text-emerald-400">Checking...</span>
                    </div>
                    <div class="flex justify-between text-slate-400">
                        <span>Vector Size:</span>
                        <span class="text-slate-300">192-D (ECAPA-TDNN)</span>
                    </div>
                </div>
                <div class="flex gap-3 pt-2">
                    <button onclick="triggerEnrollment()" class="flex-1 bg-blue-600 hover:bg-blue-500 py-2.5 rounded-xl font-medium text-sm transition flex items-center justify-center gap-2">
                        <i class="fa-solid fa-microphone"></i> Re-Enroll Voice
                    </button>
                    <button onclick="deleteProfile()" class="bg-red-500/20 hover:bg-red-500/30 text-red-400 border border-red-500/30 px-4 py-2.5 rounded-xl font-medium text-sm transition">
                        <i class="fa-solid fa-trash"></i>
                    </button>
                </div>
            </div>

            <!-- Tracking & GPS Tuning -->
            <div class="bg-slate-800/60 backdrop-blur border border-slate-700/60 rounded-2xl p-6 space-y-4">
                <h2 class="text-lg font-semibold flex items-center gap-2">
                    <i class="fa-solid fa-location-crosshairs text-cyan-400"></i> Moving GPS Breadcrumbs
                </h2>
                <p class="text-sm text-slate-400">
                    Fine-tune movement thresholds for dynamic cab/transit tracking.
                </p>
                <div class="space-y-3">
                    <div>
                        <label class="text-xs text-slate-400">GPS Ping Interval (Seconds)</label>
                        <input id="tracking-interval" type="number" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-blue-500" value="15">
                    </div>
                    <div>
                        <label class="text-xs text-slate-400">Min Relocation Distance (Meters)</label>
                        <input id="min-distance" type="number" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-blue-500" value="25">
                    </div>
                </div>
            </div>
        </div>

        <!-- Emergency Contacts Manager -->
        <div class="bg-slate-800/60 backdrop-blur border border-slate-700/60 rounded-2xl p-6 space-y-6">
            <div class="flex justify-between items-center">
                <div>
                    <h2 class="text-lg font-semibold flex items-center gap-2">
                        <i class="fa-solid fa-users text-purple-400"></i> Emergency Contacts Mesh
                    </h2>
                    <p class="text-sm text-slate-400">All enabled contacts will receive live SOS alerts, Google Maps pins, and audio proof.</p>
                </div>
            </div>

            <!-- Contacts List -->
            <div id="contacts-container" class="space-y-3">
                <!-- Dynamically Populated -->
            </div>

            <!-- Add Contact Form -->
            <div class="pt-4 border-t border-slate-700/50 flex flex-col md:flex-row gap-3">
                <input id="new-name" type="text" placeholder="Contact Name (e.g., Mom / Roommate)" class="flex-1 bg-slate-900 border border-slate-700 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-blue-500">
                <input id="new-chat-id" type="text" placeholder="Telegram Chat ID (from @userinfobot)" class="flex-1 bg-slate-900 border border-slate-700 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-blue-500">
                <button onclick="addContact()" class="bg-emerald-600 hover:bg-emerald-500 px-6 py-2.5 rounded-xl font-medium text-sm transition flex items-center justify-center gap-2">
                    <i class="fa-solid fa-plus"></i> Add
                </button>
            </div>
        </div>

        <!-- Save Button -->
        <div class="flex justify-end">
            <button onclick="saveAllSettings()" class="bg-blue-600 hover:bg-blue-500 px-8 py-3 rounded-xl font-semibold text-sm shadow-lg shadow-blue-500/20 transition">
                Save & Apply Settings
            </button>
        </div>
    </div>

    <script>
        let currentSettings = {};

        async function loadData() {
            const res = await fetch('/api/settings');
            currentSettings = await res.json();
            
            document.getElementById('tracking-interval').value = currentSettings.tracking_interval_sec || 15;
            document.getElementById('min-distance').value = currentSettings.min_movement_threshold_m || 25;
            
            const profileRes = await fetch('/api/profile-status');
            const profileStatus = await profileRes.json();
            document.getElementById('profile-exists').innerText = profileStatus.exists ? '✓ Enrolled' : '✗ Missing';
            document.getElementById('profile-exists').className = profileStatus.exists ? 'font-medium text-emerald-400' : 'font-medium text-red-400';

            renderContacts();
        }

        function renderContacts() {
            const container = document.getElementById('contacts-container');
            container.innerHTML = '';
            
            (currentSettings.contacts || []).forEach((c, idx) => {
                const div = document.createElement('div');
                div.className = "flex items-center justify-between bg-slate-900/60 p-4 rounded-xl border border-slate-700/50";
                div.innerHTML = `
                    <div class="flex items-center gap-3">
                        <div class="w-8 h-8 rounded-full bg-purple-500/20 text-purple-400 flex items-center justify-center text-xs font-bold">
                            ${c.name.charAt(0).toUpperCase()}
                        </div>
                        <div>
                            <p class="font-medium text-sm text-slate-200">${c.name}</p>
                            <p class="text-xs text-slate-400">Telegram ID: <code>${c.chat_id}</code></p>
                        </div>
                    </div>
                    <div class="flex items-center gap-3">
                        <label class="relative inline-flex items-center cursor-pointer">
                            <input type="checkbox" ${c.enabled ? 'checked' : ''} onchange="toggleContact(${idx})" class="sr-only peer">
                            <div class="w-9 h-5 bg-slate-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-emerald-500"></div>
                        </label>
                        <button onclick="removeContact(${idx})" class="text-slate-500 hover:text-red-400 transition p-1">
                            <i class="fa-solid fa-trash-can text-sm"></i>
                        </button>
                    </div>
                `;
                container.appendChild(div);
            });
        }

        function toggleContact(idx) {
            currentSettings.contacts[idx].enabled = !currentSettings.contacts[idx].enabled;
        }

        function removeContact(idx) {
            currentSettings.contacts.splice(idx, 1);
            renderContacts();
        }

        function addContact() {
            const name = document.getElementById('new-name').value.trim();
            const chatId = document.getElementById('new-chat-id').value.trim();
            if (!name || !chatId) return alert('Enter both Name and Telegram Chat ID');
            
            currentSettings.contacts.push({ name, chat_id: chatId, enabled: true });
            document.getElementById('new-name').value = '';
            document.getElementById('new-chat-id').value = '';
            renderContacts();
        }

        async function saveAllSettings() {
            currentSettings.tracking_interval_sec = parseInt(document.getElementById('tracking-interval').value);
            currentSettings.min_movement_threshold_m = parseFloat(document.getElementById('min-distance').value);
            
            const res = await fetch('/api/settings', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(currentSettings)
            });
            if (res.ok) alert('✓ Settings and Contacts Saved Successfully!');
        }

        async function deleteProfile() {
            if (!confirm('Are you sure you want to delete your enrolled safe-word voice profile?')) return;
            await fetch('/api/delete-profile', { method: 'POST' });
            loadData();
        }

        function triggerEnrollment() {
            alert('To re-enroll your voice profile with high audio fidelity, please run: python enroll.py in your terminal!');
        }

        loadData();
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return HTML_TEMPLATE

@app.get("/api/settings")
def get_settings():
    return config.load_settings()

@app.post("/api/settings")
async def update_settings(req: Request):
    data = await req.json()
    config.save_settings(data)
    return {"status": "saved"}

@app.get("/api/profile-status")
def profile_status():
    exists = config.CENTROID_PATH.exists() and config.SPEC_PATH.exists()
    return {"exists": exists}

@app.post("/api/delete-profile")
def delete_profile():
    if config.CENTROID_PATH.exists():
        os.remove(config.CENTROID_PATH)
    if config.SPEC_PATH.exists():
        os.remove(config.SPEC_PATH)
    if config.METRICS_PATH.exists():
        os.remove(config.METRICS_PATH)
    return {"status": "deleted"}



if __name__ == "__main__":
    uvicorn.run("web_server:app", host="127.0.0.1", port=8000, reload=True)