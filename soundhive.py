# Self-Hosted MQTT Media Player for Home Assistant (Standalone)
# Version: 4.0.0
# Description:
# - Implemented workaround for TTS playback by converting `play_media` action to `play` with filepath handling.
# - Bypasses reliance on unsupported `play_media` in TroyFernandes' component.
# - Streams TTS audio to a local file and plays it with `aplay`.
# - Optimized for rapid TTS playback without additional component changes.
#
# Changelog:
# v4.0.0 - Workaround for TTS playback using `play` action (Feb 20, 2025)

import paho.mqtt.client as mqtt
import logging
import subprocess
import json

try:
    import requests
except ImportError:
    print("❌ The 'requests' module is not installed. Please run 'pip install requests' and try again.")
    exit(1)

# Configuration
MQTT_BROKER = "192.168.188.62"
MQTT_PORT = 1883
MQTT_USER = "test"
MQTT_PASSWORD = "test"
MQTT_ENTITY_ID = "selfhosted_mediaplayer"
MQTT_DISCOVERY_PREFIX = "homeassistant/mqtt-mediaplayer"
MQTT_DISCOVERY_TOPIC = f"{MQTT_DISCOVERY_PREFIX}/{MQTT_ENTITY_ID}/config"
MQTT_COMMAND_TOPIC = f"{MQTT_ENTITY_ID}/command"
MQTT_STATE_TOPIC = f"{MQTT_ENTITY_ID}/state"
MQTT_AVAILABLE_TOPIC = f"{MQTT_ENTITY_ID}/available"

AUDIO_DEVICE = "plughw:CARD=MS,DEV=0"
DEFAULT_AUDIO_FILE = "/home/satellite/wyoming-satellite/sounds/awake.wav"
TTS_TEMP_FILE = "/tmp/tts_stream.wav"

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

def publish_state(client, state):
    """Publish current state to MQTT for Home Assistant visibility."""
    try:
        payload = json.dumps({"status": state})
        client.publish(MQTT_STATE_TOPIC, payload, retain=True)
        logging.info(f"📡 State updated to: {state}")
    except Exception as e:
        logging.error(f"❌ State publishing error: {e}")

def publish_discovery(client):
    """Publish MQTT discovery payload for hass-mqtt-mediaplayer integration."""
    discovery_payload = {
        "name": "Self-Hosted MQTT Media Player",
        "unique_id": MQTT_ENTITY_ID,
        "command_topic": MQTT_COMMAND_TOPIC,
        "state_topic": MQTT_STATE_TOPIC,
        "availability_topic": MQTT_AVAILABLE_TOPIC,
        "payload_available": "online",
        "payload_not_available": "offline",
        "schema": "json",
        "supported_features": ["play", "pause", "next", "previous", "volume_set"],
        "device": {
            "identifiers": [MQTT_ENTITY_ID],
            "manufacturer": "Self-Hosted Solutions",
            "model": "MQTT Media Player v4",
            "name": "MQTT Media Player"
        }
    }
    try:
        client.publish(MQTT_DISCOVERY_TOPIC, json.dumps(discovery_payload), retain=True)
        logging.info("✅ Published updated MQTT discovery payload.")
    except Exception as e:
        logging.error(f"❌ Discovery payload error: {e}")

def play_audio_file(client, filepath):
    """Play a specified audio file using aplay and update state accordingly."""
    try:
        command = f"aplay -D {AUDIO_DEVICE} {filepath}"
        logging.info(f"🔊 Playing audio file: {filepath}")
        publish_state(client, "playing")
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            logging.info("✅ Audio file playback complete.")
            publish_state(client, "idle")
        else:
            logging.error(f"❌ Playback error: {result.stderr.decode().strip()}")
            publish_state(client, "error")
    except Exception as e:
        logging.error(f"❌ File playback error: {e}")
        publish_state(client, "error")

def download_and_play_audio(client, url):
    """Download TTS audio from URL and play it using aplay."""
    try:
        logging.info(f"🌐 Downloading TTS audio from URL: {url}")
        publish_state(client, "playing")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(TTS_TEMP_FILE, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        play_audio_file(client, TTS_TEMP_FILE)
    except Exception as e:
        logging.error(f"❌ TTS streaming or playback error: {e}")
        publish_state(client, "error")

def handle_mqtt_command(client, command):
    """Process MQTT media player commands and handle TTS playback using 'play' action."""
    try:
        cmd = json.loads(command)
        action = cmd.get("command", "")
        logging.info(f"🎬 Action received: {action}")

        if action == "play":
            # Workaround: Treat 'filepath' as TTS URL if provided
            filepath_or_url = cmd.get("args", {}).get("filepath", DEFAULT_AUDIO_FILE)
            if filepath_or_url.startswith("http"):
                download_and_play_audio(client, filepath_or_url)
            else:
                play_audio_file(client, filepath_or_url)
        elif action == "pause":
            logging.info("⏸ Pause requested - no action defined for aplay, state set to paused.")
            publish_state(client, "paused")
        elif action == "next":
            logging.info("⏭ Next requested - no playlist implemented.")
            publish_state(client, "idle")
        elif action == "previous":
            logging.info("⏮ Previous requested - no playlist implemented.")
            publish_state(client, "idle")
        elif action == "volume_set":
            volume = cmd.get("args", {}).get("volume", "50")
            execute_command(client, f"amixer set 'Playback' {volume}%", "volume_set")
        else:
            logging.warning(f"⚠️ Unknown action: {action}")
            publish_state(client, "error")
    except json.JSONDecodeError as e:
        logging.error(f"❌ JSON decode error: {e}")
        publish_state(client, "error")
    except Exception as e:
        logging.error(f"❌ Command handling error: {e}")
        publish_state(client, "error")

def execute_command(client, command, state_after):
    """Execute system commands for media control and publish updated state."""
    try:
        logging.debug(f"Executing: {command}")
        subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        publish_state(client, state_after)
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ Command error: {e.stderr.decode().strip()}")
        publish_state(client, "error")

def announce_client(client):
    """Announce availability via MQTT."""
    try:
        client.publish(MQTT_AVAILABLE_TOPIC, "online", retain=True)
        logging.info("📢 Media Player announced as available.")
    except Exception as e:
        logging.error(f"❌ Announcement error: {e}")

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        logging.info("✅ Connected to MQTT broker at %s:%s.", MQTT_BROKER, MQTT_PORT)
        announce_client(client)
        publish_state(client, "idle")
        client.subscribe(MQTT_COMMAND_TOPIC)
        logging.info("📡 Subscribed to: %s", MQTT_COMMAND_TOPIC)
        publish_discovery(client)
    else:
        logging.error(f"❌ Connection failed. Return code {rc}")

def on_message(client, userdata, msg):
    logging.debug(f"📩 Received - Topic: {msg.topic}, Payload: {msg.payload.decode()}")
    handle_mqtt_command(client, msg.payload.decode())

def main():
    logging.info("🚀 Starting Self-Hosted MQTT Media Player (v4.0.0)...")
    client = mqtt.Client(client_id=MQTT_ENTITY_ID)
    client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        logging.info("🔌 Connecting to MQTT broker...")
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        logging.info("🟢 Running MQTT loop...")
        client.loop_forever()
    except Exception as e:
        logging.error(f"❌ Startup error: {e}")

if __name__ == "__main__":
    main()

