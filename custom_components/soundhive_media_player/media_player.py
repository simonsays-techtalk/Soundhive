# soundhive_mqtt_client.py
# Soundhive MQTT Client: Updated to Version 0.6.7 with corrected audio device handling
# Version: 0.6.7 (Fixed mpg123 and ffplay device configurations)

import logging
import json
import os
import requests
import subprocess
import paho.mqtt.client as mqtt

try:
    from mutagen.mp3 import MP3
except ImportError:
    print("⚠️ 'mutagen' not found. Attempting to install...")
    subprocess.check_call(["python3", "-m", "pip", "install", "mutagen"])
    from mutagen.mp3 import MP3

_LOGGER = logging.getLogger(__name__)

# Version Information
VERSION = "0.6.7"

# MQTT Configuration
MQTT_BROKER = "192.168.188.62"
MQTT_PORT = 1883
MQTT_USER = "test"
MQTT_PASSWORD = "test"
MQTT_TOPIC_COMMAND = "selfhosted_mediaplayer/command"

# Audio Playback Configuration
MP3_PLAYER = "mpg123"  # Primary MP3 player
MP3_FALLBACK_PLAYER = "ffplay"  # Fallback player if mpg123 fails
# Adjusted options for correct device handling
MP3_PLAYER_OPTIONS = ["--mono", "--rate", "22050", "-v", "-o", "alsa", "--audiodevice", "plughw:1,0"]
FFPLAY_OPTIONS = ["-nodisp", "-autoexit", "-loglevel", "verbose", "-ac", "1", "-ar", "22050", "-f", "alsa", "-i", "hw:1,0"]
WAV_PLAYER = "aplay"
TEMP_AUDIO_PATH = "/tmp/tts_stream"

def download_file(url, local_path):
    """Download a file from the given URL to the specified local path."""
    _LOGGER.info(f"🌐 Downloading TTS audio from URL: {url}")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        _LOGGER.info("✅ Download complete.")
        return True
    else:
        _LOGGER.error(f"❌ Failed to download file. Status code: {response.status_code}")
        return False

def validate_mp3(filepath):
    """Validate MP3 file using mutagen to ensure it has audio frames."""
    try:
        audio = MP3(filepath)
        _LOGGER.info(f"🎼 MP3 validated: {audio.info.length:.2f} seconds, {audio.info.bitrate} bps")
        return True
    except Exception as e:
        _LOGGER.error(f"❌ MP3 validation failed: {e}")
        return False

def play_audio(filepath, media_type):
    """Play the audio file based on the media type."""
    _LOGGER.info(f"🔊 Playing audio file: {filepath}")
    try:
        if media_type == "audio/mp3":
            _LOGGER.info("🎼 Using mpg123 for MP3 playback with corrected device settings.")
            _LOGGER.info(f"📝 Running command: {[MP3_PLAYER] + MP3_PLAYER_OPTIONS + [filepath]}")
            result = subprocess.run([MP3_PLAYER] + MP3_PLAYER_OPTIONS + [filepath], check=False)
            if result.returncode != 0:
                _LOGGER.warning("⚠️ mpg123 failed. Attempting playback with ffplay using corrected device settings.")
                subprocess.run([MP3_FALLBACK_PLAYER] + FFPLAY_OPTIONS + [filepath], check=True)
        else:
            _LOGGER.info("🎼 Using aplay for WAV playback.")
            subprocess.run([WAV_PLAYER, filepath], check=True)
        _LOGGER.info("✅ Audio file playback complete.")
    except subprocess.CalledProcessError as e:
        _LOGGER.error(f"❌ Playback error: {str(e)}")

def on_message(client, userdata, msg):
    """Handle incoming MQTT messages."""
    try:
        payload = json.loads(msg.payload.decode())
        command = payload.get("command")
        args = payload.get("args", {})
        _LOGGER.info(f"🎬 Action received: {command}")

        if command == "play":
            filepath = args.get("filepath")
            media_type = args.get("media_type", "audio/wav")
            temp_file = f"{TEMP_AUDIO_PATH}.{'mp3' if media_type == 'audio/mp3' else 'wav'}"

            if download_file(filepath, temp_file):
                if media_type == "audio/mp3" and not validate_mp3(temp_file):
                    _LOGGER.error("❌ Invalid MP3 file. Skipping playback.")
                else:
                    play_audio(temp_file, media_type)
                os.remove(temp_file)
            else:
                _LOGGER.error("❌ Failed to download or play audio file.")

    except Exception as e:
        _LOGGER.error(f"❌ Error handling MQTT message: {e}")

def main():
    """Initialize MQTT client and start listening for commands."""
    _LOGGER.info(f"🚀 Starting Soundhive MQTT Client (v{VERSION})...")
    client = mqtt.Client()
    client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.subscribe(MQTT_TOPIC_COMMAND)
    _LOGGER.info(f"📡 Subscribed to topic: {MQTT_TOPIC_COMMAND}")
    _LOGGER.info(f"📝 Running version: {VERSION}")
    client.loop_forever()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()

# ✅ Version 0.6.7:
# - Adjusted mpg123 and ffplay device configurations for correct headset handling.
# - Improved device targeting with 'plughw:1,0' for better compatibility.
# - Confirmed version displayed at startup for easy tracking.
