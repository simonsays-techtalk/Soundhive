# 🎵 Soundhive – Offline Media Player for Home Assistant

*A fully self-hosted, cloud-free media player with TTS, STT, and LLM integration, built for Raspberry Pi.*

Soundhive is a minimal, zero-cloud media player client that runs on a Raspberry Pi and integrates seamlessly with Home Assistant. It features speech-to-text (STT), large language model (LLM) processing, and text-to-speech (TTS) in a single pipeline — with optional offline processing using Whisper.cpp and Piper.

---

## 🚀 Features

- ✅ **Full Home Assistant Integration**: Appears as a `media_player` entity
- 🔊 **Plays TTS and Streaming Media**: Supports MP3, WAV, radio streams, and local playback
- 🗣 **Offline Voice Control**: Uses STT (Whisper.cpp) → LLM → TTS (Piper)
- 🧠 **Local LLM Integration**: Supports integration with self-hosted Ollama
- 📦 **One File**: All functionality is in `soundhive_client.py`
- ⚙️ **Automatic Hardware Detection**: Optimizes install for RPi Zero W or more powerful boards
- 🎙️ **ReSpeaker Mic 2 HAT Ready**: Optional mic hardware supported out of the box

---

## 🛠 Installation on Raspberry Pi

1. Flash and boot Raspberry Pi OS Lite
2. SSH into your Pi or use a terminal
3. Run the installer:

```bash
wget https://raw.githubusercontent.com/simonsays-techtalk/Soundhive/main/installer/install.py
python3 install.py
```

4. Follow the prompts (you’ll enter your HA token, select TTS/STT settings, etc.)

> 💡 The installer creates a virtual environment and installs only the required dependencies.

---

## 🏠 Integration in Home Assistant

### 📦 Install the Custom Integration

1. Copy the `custom_components/soundhive_media_player/` folder into your HA `custom_components/`
2. Restart Home Assistant
3. Go to **Settings > Devices & Services > Integrations**
4. Click **"+ Add Integration"** and search for `Soundhive`
5. Follow the prompts to add your Soundhive media player(s)

> 🔐 You’ll need a long-lived access token from Home Assistant for authentication

---

## 🧪 Troubleshooting

- Client not connecting? Double-check your HA token and network.
- No sound? Make sure your device supports audio output and test with `aplay`
- No TTS? Ensure Piper is installed and configured in HA (`tts.piper_2` or similar).
- Client not appearing? Restart HA and enable debug logging if needed.

---

## 🛡 Security Note

- Your HA token is stored locally on the Pi in a config file. This file is readable by users with sufficient privileges on the device. Ensure physical and SSH access is limited.

---

## 🗂 Structure Overview

```
client/soundhive_client.py           # Main logic for TTS/STT/playback
installer/install.py                # Python-based setup script for Raspberry Pi
custom_components/soundhive_media_player/  # HA integration
```

---

## 📋 License

This project is open-source and licensed under MIT.

---

Built with ❤️ by simonsays-techtalk and his local AI.


