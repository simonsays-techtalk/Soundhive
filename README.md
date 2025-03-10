# 🎵 Soundhive - Custom Home Assistant Media Player
*A self-hosted, MQTT-based media player for Home Assistant with TTS and streaming support.*

This mediaplayer was developed by me and my local AI. I want a way out of the "cloud" and created this basic mediaplayer.
Please mind that this version uses Piper as TTS. I am working on a more flexible solution. The client is developed around the raspberry pi. It runs with minimal load on a raspberry pi Zero W. The installer optionally utilizes a respeaker mic2hat, so that everything fits nicely in this case https://www.thingiverse.com/thing:4766696.

## 🚀 Features
- ✅ ** Basic Home Assistant Integration**: Appears as a `media_player` entity in HA.
- 🔊 **Supports TTS & Media Playback**: Plays music, radio, and TTS announcements.
- 🌍 **Lightweight** Optimized for Raspberry Pi Zero W and respeaker mic2hat.
- 🎛️ **Simple UI Controls**: Play, Pause, Volume, Next, and Previous.
- 🛠️ **Easy Setup**: use install.py to install the client.

*You must use Piper tts at the moment.
---
## 🏗️ Installation on Raspberry Pi
1. Install and update RPI
2. Download the installer: wget https://raw.githubusercontent.com/simonsays-techtalk/Soundhive/main/client/install.py
3. In homeassistant, create a Long-lived access token
4. Run the installer: python3 install.py
5. Follow the installer instructions, on RPI Zero W, installation can take some time!

## 🏗️ Installation in Homeassistant

### 📌 1. Install the Home Assistant Integration
1. Copy `custom_components/soundhive` into your Home Assistant `custom_components` directory.
2. Restart Home Assistant.
3. Add the Soundhive integration and add your first Soundhive mediaplayer.
4. Follow the instructions when adding a new player.

🛠️ Troubleshooting
- Client not connecting? Verify the auth token.
- Not appearing in HA? Restart Home Assistant and check logs, enable debug log if necessary.
- No TTS? Make sure you have installed a TTS engine, like piper. Use tts.<your-tts-engine> as entity.
