# soundhive_mqtt_client.py
# Soundhive MQTT Client: Version 1.2.1 (Dynamic Unique ID & Multi-Instance Support)
# Version: 1.2.1

import logging
import json
import os
import requests
import subprocess
import signal
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

VERSION = "1.2.1 (Dynamic Unique ID & Multi-Instance Support)"

MQTT_BROKER = os.getenv("MQTT_BROKER", "homeassistanttest.local")
MQTT_PORT = 1883
MQTT_USER = os.getenv("MQTT_USER", "")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", "")

if not MQTT_USER:
    MQTT_USER = input("🔐 Enter MQTT Username: ")
    MQTT_PASSWORD = input("🔐 Enter MQTT Password: ")

# Prompt for unique media player name
MEDIA_PLAYER_NAME = input("📝 Enter a unique name for this Soundhive Media Player (default: Soundhive_mediaplayer): ") or "Soundhive_mediaplayer"

MQTT_TOPIC_COMMAND = f"{MEDIA_PLAYER_NAME}/command"
MQTT_TOPIC_STATE = f"{MEDIA_PLAYER_NAME}/state"
MQTT_TOPIC_NOW_PLAYING = f"{MEDIA_PLAYER_NAME}/now_playing"
MQTT_TOPIC_VOLUME = f"{MEDIA_PLAYER_NAME}/volume"

print(f"⚡ Please ensure that 'unique_id' in configuration.yaml matches: '{MEDIA_PLAYER_NAME}' for proper HA UI management.")

MP3_PLAYER = "mpg123"
MP3_PLAYER_OPTIONS = ["--mono", "--rate", "22050", "-v", "-o", "alsa", "--audiodevice", "plughw:1,0"]
FFPLAY_COMMAND = "ffplay"
FFPLAY_OPTIONS_STREAM = [FFPLAY_COMMAND, "-nodisp", "-autoexit", "-loglevel", "verbose", "-ac", "2", "-ar", "48000"]
TEMP_AUDIO_PATH = "/tmp/tts_stream"

SUPPORTED_CONTENT_TYPES = ["music", "audio/mp3", "audio/wav"]
HA_BASE_URL = os.getenv("HA_BASE_URL", "http://homeassistanttest.local:8123")

current_playing = ""
current_volume = 50
current_state = "idle"
current_process = None

def publish_state(client, state, now_playing=None):
    payload = {
        "state": state,
        "now_playing": now_playing or current_playing,
        "volume": current_volume
    }
    client.publish(MQTT_TOPIC_STATE, json.dumps(payload))
    _LOGGER.debug(f"📡 State updated to: {payload}")

def publish_now_playing(client, now_playing):
    client.publish(MQTT_TOPIC_NOW_PLAYING, json.dumps({"now_playing": now_playing}))
    _LOGGER.debug(f"🎵 Now playing: {now_playing}")

def handle_volume(client, volume_level):
    global current_volume
    current_volume = max(0, min(100, int(volume_level)))
    subprocess.call(["amixer", "set", "Playback", f"{current_volume}%"])
    _LOGGER.debug(f"🔊 Volume set to: {current_volume}%")
    client.publish(MQTT_TOPIC_VOLUME, json.dumps({"volume": current_volume}))

def stop_playback():
    global current_process, current_state
    if current_process and current_process.poll() is None:
        current_process.terminate()
        _LOGGER.debug("⏹️ Playback stopped.")
    current_state = "idle"

def pause_playback():
    global current_process, current_state
    if current_process and current_process.poll() is None:
        current_process.send_signal(signal.SIGSTOP)
        _LOGGER.debug("⏸️ Playback paused.")
        current_state = "paused"

def resume_playback():
    global current_process, current_state
    if current_process and current_process.poll() is None:
        current_process.send_signal(signal.SIGCONT)
        _LOGGER.debug("▶️ Playback resumed.")
        current_state = "playing"

def play_audio(filepath, media_type, client):
    global current_playing, current_process
    current_playing = filepath
    publish_now_playing(client, filepath)
    publish_state(client, "playing")
    _LOGGER.debug(f"🔊 Playing file: {filepath}")
    try:
        current_process = subprocess.Popen([MP3_PLAYER] + MP3_PLAYER_OPTIONS + [filepath])
        current_process.wait()
    except Exception as e:
        _LOGGER.error(f"❌ Playback error: {e}")
    finally:
        publish_state(client, "idle")

def stream_audio(stream_url, media_type, client):
    global current_playing, current_process
    current_playing = stream_url
    publish_now_playing(client, stream_url)
    publish_state(client, "playing")
    command = FFPLAY_OPTIONS_STREAM + [stream_url]
    _LOGGER.debug(f"📝 Streaming command: {command}")
    try:
        current_process = subprocess.Popen(command)
        current_process.wait()
    except Exception as e:
        _LOGGER.error(f"❌ Streaming failed: {str(e)}")
    finally:
        publish_state(client, "idle")

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        _LOGGER.info("✅ Connected to MQTT Broker!")
        client.subscribe([(MQTT_TOPIC_COMMAND, 0), (MQTT_TOPIC_VOLUME, 0)])
    else:
        _LOGGER.error(f"❌ Failed to connect. Return code: {rc}")

def on_message(client, userdata, msg):
    _LOGGER.debug(f"📨 Received MQTT message: {msg.topic} -> {msg.payload.decode()}")
    try:
        payload = json.loads(msg.payload.decode())
        if msg.topic == MQTT_TOPIC_VOLUME:
            handle_volume(client, payload.get("volume", current_volume))
        elif msg.topic == MQTT_TOPIC_COMMAND:
            command = payload.get("command")
            if command == "play":
                args = payload.get("args", {})
                filepath = args.get("filepath")
                media_type = args.get("media_type", "audio/mp3")
                if filepath:
                    if filepath.startswith("http"):
                        stream_audio(filepath, media_type, client)
                    else:
                        play_audio(filepath, media_type, client)
                else:
                    _LOGGER.warning("⚠️ No filepath provided in payload.")
            elif command == "pause":
                pause_playback()
                publish_state(client, "paused")
            elif command == "resume":
                resume_playback()
                publish_state(client, "playing")
            elif command == "stop":
                stop_playback()
                publish_state(client, "idle")
    except json.JSONDecodeError as e:
        _LOGGER.error(f"❌ Failed to decode JSON: {e}")

def main():
    _LOGGER.info(f"🚀 Starting Soundhive MQTT Client (v{VERSION})...")
    client = mqtt.Client(client_id=MEDIA_PLAYER_NAME, protocol=mqtt.MQTTv311)
    client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_forever()

if __name__ == "__main__":
    main()

# ✅ Version 1.2.1:
# - Dynamic media player naming on first start with user prompt.
# - MQTT topics now reflect unique media player name.
# - Reminder displayed to update 'unique_id' in configuration.yaml accordingly.
# - Supports multiple Soundhive media players on the same network.
