# soundhive_mqtt_client.py
# Soundhive MQTT Client: Updated Version 1.1.4 with Fixed Streaming Handling
# Version: 1.1.4

import logging
import json
import os
import requests
import subprocess
import paho.mqtt.client as mqtt
import time

try:
    from mutagen.mp3 import MP3
except ImportError:
    print("⚠️ 'mutagen' not found. Attempting to install...")
    subprocess.check_call(["python3", "-m", "pip", "install", "mutagen"])
    from mutagen.mp3 import MP3

logging.basicConfig(level=logging.DEBUG)
_LOGGER = logging.getLogger("SoundhiveMQTTClient")

# Version Information
VERSION = "1.1.4 (Fixed Streaming Handling & Device Routing)"

# MQTT Configuration
MQTT_BROKER = os.getenv("MQTT_BROKER", "homeassistanttest.local")
MQTT_PORT = 1883
MQTT_USER = os.getenv("MQTT_USER", "")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", "")
MQTT_TOPIC_COMMAND = "selfhosted_mediaplayer/command"
MQTT_TOPIC_STATE = "selfhosted_mediaplayer/state"

if not MQTT_USER:
    MQTT_USER = input("🔐 Enter MQTT Username: ")
    MQTT_PASSWORD = input("🔐 Enter MQTT Password: ")

# Audio Playback Configuration
MP3_PLAYER = "mpg123"
MP3_PLAYER_OPTIONS = ["--mono", "--rate", "22050", "-v", "-o", "alsa", "--audiodevice", "plughw:1,0"]
FFPLAY_COMMAND = "ffplay"
# ✅ Fixed streaming command by removing device routing from streaming (ffplay uses default device)
FFPLAY_OPTIONS_STREAM = [FFPLAY_COMMAND, "-nodisp", "-autoexit", "-loglevel", "verbose", "-ac", "2", "-ar", "48000"]
WAV_PLAYER = "aplay"
TEMP_AUDIO_PATH = "/tmp/tts_stream"

SUPPORTED_CONTENT_TYPES = ["music", "audio/mp3", "audio/wav"]

HA_BASE_URL = os.getenv("HA_BASE_URL", "http://homeassistanttest.local:8123")

def publish_state(client, state):
    client.publish(MQTT_TOPIC_STATE, json.dumps({"state": state}))
    _LOGGER.info(f"📡 State updated to: {state}")

def download_file(url, local_path):
    if url.startswith("/api/tts_proxy/"):
        url = f"{HA_BASE_URL}{url}"
    _LOGGER.info(f"🌐 Downloading audio from URL: {url}")
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        _LOGGER.info("✅ Download complete.")
        return True
    except Exception as e:
        _LOGGER.error(f"❌ Failed to download file: {e}")
        return False

def validate_mp3(filepath):
    try:
        audio = MP3(filepath)
        _LOGGER.info(f"🎼 MP3 validated: {audio.info.length:.2f} seconds, {audio.info.bitrate} bps")
        return True
    except Exception as e:
        _LOGGER.error(f"❌ MP3 validation failed: {e}")
        return False

def play_audio(filepath, media_type, client):
    publish_state(client, "playing")
    _LOGGER.info(f"🔊 Playing audio file: {filepath} as {media_type}")
    try:
        subprocess.run([MP3_PLAYER] + MP3_PLAYER_OPTIONS + [filepath], check=True)
        _LOGGER.info("✅ Audio playback complete.")
    except subprocess.CalledProcessError as e:
        _LOGGER.error(f"❌ Playback error: {e}")
    finally:
        publish_state(client, "idle")

def stream_audio(stream_url, media_type, client):
    publish_state(client, "playing")
    _LOGGER.info(f"🌐 Streaming from URL: {stream_url}")
    command = FFPLAY_OPTIONS_STREAM + [stream_url]
    _LOGGER.info(f"📝 Streaming command: {command}")
    try:
        subprocess.run(command, check=True, timeout=120)
        _LOGGER.info("✅ Streaming playback complete.")
    except subprocess.CalledProcessError as e:
        _LOGGER.error(f"❌ Streaming failed: {str(e)}")
    finally:
        publish_state(client, "idle")

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        _LOGGER.info("✅ Connected to MQTT Broker!")
        client.subscribe(MQTT_TOPIC_COMMAND)
    else:
        _LOGGER.error(f"❌ Failed to connect. Return code: {rc}")

def on_message(client, userdata, msg):
    payload = json.loads(msg.payload.decode())
    args = payload.get("args", {})
    filepath = args.get("filepath")
    media_type = args.get("media_type", "audio/mp3")
    _LOGGER.debug(f"📦 Received payload: {payload}")
    if filepath:
        if filepath.startswith("http"):
            stream_audio(filepath, media_type, client)
        else:
            if validate_mp3(filepath):
                play_audio(filepath, media_type, client)
    else:
        _LOGGER.warning("⚠️ No filepath provided in payload.")

def main():
    _LOGGER.info(f"🚀 Starting Soundhive MQTT Client (v{VERSION})...")
    client = mqtt.Client(client_id="soundhive_client", protocol=mqtt.MQTTv311)
    client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_forever()

if __name__ == "__main__":
    main()

# ✅ Version 1.1.4:
# - Removed conflicting device routing parameters from ffplay streaming command.
# - Streaming now uses default system device (plughw:1,0 managed automatically).
# - Retained explicit device handling for TTS (mpg123).
# - Improved subprocess logging for debugging streaming issues.
