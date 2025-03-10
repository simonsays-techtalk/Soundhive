#!/usr/bin/env python3
import asyncio
import aiohttp
import json
import logging
import os
import sys
import tempfile
import time
import vlc
import sounddevice as sd
import soundfile as sf
import string
from urllib.parse import urlparse

# --- Configuration and Global Constants ---
CONFIG_FILE = "soundhive_config.json"
VERSION = "2.0 Media playback and TTS. STT alpha status"
MEDIA_PLAYER_ENTITY = "media_player.soundhive_media_player"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger("SoundhiveClient")

# --- Load Config and Set Variables ---
def load_config():
    try:
        with open(CONFIG_FILE, "r") as f:
            config_data = json.load(f)
            _LOGGER.info("🎚 Loaded config: %s", config_data)  # Log full config
            return config_data
    except Exception as e:
        _LOGGER.error("Error loading configuration file: %s", e)
        sys.exit(1)

config = load_config()
HA_BASE_URL = config.get("ha_url")
TOKEN = config.get("auth_token")
TTS_ENGINE = config.get("tts_engine", "tts.google_translate_en_com")
STT_URI = config.get("stt_uri")  # e.g. "http://192.168.188.8:10900/inference"
STT_FORMAT = config.get("stt_format", "wav")
VOLUME_SETTING = config.get("volume", 0.5)
# Keyword and timeout settings:
WAKE_KEYWORD = config.get("wake_keyword", "hey assistant")
SLEEP_KEYWORD = config.get("sleep_keyword", "goodbye")
ACTIVE_TIMEOUT = config.get("active_timeout", 5)  # seconds

WS_API_URL = f"{HA_BASE_URL}/api/websocket"
REGISTER_ENTITY_API = f"{HA_BASE_URL}/api/states/{MEDIA_PLAYER_ENTITY}"
TTS_API_URL = f"{HA_BASE_URL}/api/tts_get_url"

# --- Helper Functions ---
def normalize_text(text):
    """Normalize text: lowercase, remove punctuation, and strip extra whitespace."""
    translator = str.maketrans("", "", string.punctuation)
    return text.lower().translate(translator).strip()

# --- VLC Media Playback Setup ---
vlc_instance = vlc.Instance('--aout=alsa')
current_player = None
media_paused = False

def get_volume_control_name():
    return "VLC"

VOLUME_CONTROL = get_volume_control_name()
_LOGGER.info("🔊 Using VLC for audio playback (ALSA forced)")

MEDIA_LIBRARY = [
    {"title": "Song 1", "media_url": "http://example.com/song1.mp3"},
    {"title": "Song 2", "media_url": "http://example.com/song2.mp3"}
]

# --- REST API Functions ---
async def register_media_player(session):
    headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
    payload = {
        "state": "idle",
        "attributes": {
            "friendly_name": "Soundhive Media Player",
            "supported_features": 52735,
            "media_content_id": None,
            "media_content_type": None,
            "volume_level": VOLUME_SETTING,
            "media_library": MEDIA_LIBRARY
        }
    }
    _LOGGER.debug("Registering media player with payload: %s", json.dumps(payload))
    async with session.post(REGISTER_ENTITY_API, headers=headers, json=payload) as resp:
        _LOGGER.debug("Registration response status: %s", resp.status)
        response_text = await resp.text()
        _LOGGER.debug("Registration response body: %s", response_text)
        if resp.status in [200, 201]:
            _LOGGER.info("✅ Media player entity registered successfully.")
        else:
            _LOGGER.error("❌ Failed to register media player entity. Status: %s", resp.status)
    await update_media_state(session, "idle")

async def update_media_state(session, state, media_url=None, volume=None):
    global VOLUME_SETTING

    if volume is None:
        volume = VOLUME_SETTING  # Ensure we keep the configured volume

    headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
    payload = {
        "state": state,
        "attributes": {
            "friendly_name": "Soundhive Media Player",
            "supported_features": 52735,
            "media_content_id": media_url if media_url is not None else "",
            "media_content_type": "music" if media_url else None,
            "volume_level": volume,  # Ensure HA receives the correct volume
            "media_library": MEDIA_LIBRARY
        }
    }

    _LOGGER.info("📡 Sending state update: %s", payload)  # Debugging log

    async with session.post(REGISTER_ENTITY_API, headers=headers, json=payload) as resp:
        if resp.status in [200, 201]:
            _LOGGER.info("✅ Media player state updated: %s", state)
        else:
            _LOGGER.error("❌ Failed to update media player state. Status: %s", resp.status)

    await asyncio.sleep(1)


# --- TTS URL Resolution ---
async def resolve_tts_url(session, media_content_id):
    global config, TTS_ENGINE
    config = load_config()  # Reload configuration
    TTS_ENGINE = config.get("tts_engine", "tts.google_translate_en_com")

    if media_content_id is None:
        _LOGGER.error("❌ Received a NoneType media_content_id. Cannot resolve TTS URL.")
        return None  # Prevent crash and return None

    if media_content_id.startswith("media-source://tts/"):
        try:
            message_param = media_content_id.split("message=")[1].split("&")[0]
            message = message_param.replace("+", " ")
            async with session.post(
                TTS_API_URL,
                headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
                json={"message": message, "platform": TTS_ENGINE}
            ) as resp:
                if resp.status == 200:
                    response_json = await resp.json()
                    return response_json.get("url")
                else:
                    _LOGGER.error("❌ Failed to retrieve TTS URL. Status: %s", resp.status)
        except Exception as e:
            _LOGGER.error("❌ Error resolving TTS URL: %s", str(e))
    return media_content_id  # Return original media_content_id if it's not a TTS request


# --- Media Playback Functions ---
async def play_media(media_url):
    global current_player, media_paused

    if media_url is None:
        _LOGGER.warning("⚠️ Received a NoneType media URL. Skipping playback.")
        return

    _LOGGER.info("▶️ Playing media: %s", media_url)

    if current_player:
        current_player.stop()

    current_player = vlc_instance.media_player_new()
    media = vlc_instance.media_new(media_url)
    current_player.set_media(media)

    # Force volume from config
    volume_percentage = int(VOLUME_SETTING * 100)
    _LOGGER.info("🔊 Setting volume to: %s%%", volume_percentage)
    current_player.audio_set_volume(volume_percentage)

    current_player.play()
    media_paused = False

    async with aiohttp.ClientSession() as session:
        await update_media_state(session, "playing", media_url=media_url, volume=VOLUME_SETTING)

async def stop_media():
    global current_player, media_paused
    if current_player:
        _LOGGER.info("⏹ Stopping media playback")
        current_player.stop()
        current_player = None
        media_paused = False
    async with aiohttp.ClientSession() as session:
        await update_media_state(session, "idle")

async def pause_media():
    global current_player, media_paused

    if current_player:
        state = current_player.get_state()
        _LOGGER.info("⏸ Attempting to pause. Current VLC state: %s", state)

        if state == vlc.State.Playing:
            current_player.pause()  # This toggles pause
            media_paused = True
            async with aiohttp.ClientSession() as session:
                await update_media_state(session, "paused")
        else:
            _LOGGER.warning("⚠️ Cannot pause. Current VLC state: %s", state)
    else:
        _LOGGER.warning("⚠️ Cannot pause. No active player found.")


async def resume_media():
    global current_player, media_paused

    if current_player:
        state = current_player.get_state()
        _LOGGER.info("▶️ Attempting to resume playback. Current VLC state: %s", state)

        if state in [vlc.State.Paused, vlc.State.Stopped]:
            _LOGGER.info("▶️ Resuming media playback")
            current_player.play()
            media_paused = False

            # Force volume reset after resume
            volume_percentage = int(VOLUME_SETTING * 100)
            _LOGGER.info("🔊 Ensuring volume is set to: %s%%", volume_percentage)
            current_player.audio_set_volume(volume_percentage)

            async with aiohttp.ClientSession() as session:
                await update_media_state(session, "playing", volume=VOLUME_SETTING)
        else:
            _LOGGER.warning("⚠️ Cannot resume. Current VLC state: %s", state)


async def set_volume(level):
    global current_player
    _LOGGER.info("🔊 Setting volume to: %s", level)
    if current_player:
        current_player.audio_set_volume(int(level * 100))
    async with aiohttp.ClientSession() as session:
        state = "playing" if not media_paused else "paused"
        await update_media_state(session, state, volume=level)

# --- STT Functions (WAV-based) ---
async def record_audio(duration=3, samplerate=16000, channels=1):
    _LOGGER.info("Recording audio for %s seconds...", duration)
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels)
    sd.wait()
    # Instead of writing to disk, use an in-memory bytes buffer
    import io
    buffer = io.BytesIO()
    import soundfile as sf
    sf.write(buffer, audio_data, samplerate, format='WAV')
    buffer.seek(0)
    _LOGGER.info("Audio recorded into in-memory buffer")
    return buffer

async def send_audio_to_stt(audio_buffer):
    if not config.get("stt_uri"):
        _LOGGER.error("STT URI not configured in config.")
        return None
    stt_uri = config.get("stt_uri")
    _LOGGER.info("Sending WAV audio via HTTP POST to STT server: %s", stt_uri)
    try:
        async with aiohttp.ClientSession() as session:
            form = aiohttp.FormData()
            form.add_field("file", audio_buffer.read(),
                           filename="audio.wav",
                           content_type="audio/wav")
            async with session.post(stt_uri, data=form) as resp:
                if resp.status != 200:
                    _LOGGER.error("Error sending audio to STT server, status: %s", resp.status)
                    return None
                result = await resp.text()
                _LOGGER.info("Received transcription from STT server: %s", result)
                return result
    except Exception as e:
        _LOGGER.error("Error sending audio to STT server: %s", e)
        return None

async def perform_stt(duration=3):
    audio_buffer = await record_audio(duration=duration)
    transcription = await send_audio_to_stt(audio_buffer)
    return transcription

# --- Continuous STT Loop with Dual Keyword Detection ---
client_mode = "passive"  # "passive" or "active"
last_active_time = None

async def continuous_stt_loop():
    global client_mode, last_active_time
    _LOGGER.info("Starting continuous STT loop (initially in PASSIVE mode).")
    last_active_time = None  # Not set until active mode

    while True:
        transcription = await perform_stt(duration=3)
        transcription_normalized = normalize_text(transcription)
        current_time = time.monotonic()
        _LOGGER.debug("Current time: %.2f", current_time)

        if client_mode == "passive":
            if WAKE_KEYWORD in transcription_normalized:
                client_mode = "active"
                last_active_time = current_time
                _LOGGER.info("Wake keyword detected ('%s'). Switching to ACTIVE mode.", WAKE_KEYWORD)
            else:
                _LOGGER.info("Passive transcription (ignored): %s", transcription.strip())
        elif client_mode == "active":
            if transcription_normalized and transcription_normalized != "[blank_audio]":
                last_active_time = current_time
                _LOGGER.debug("Updated last_active_time to: %.2f", last_active_time)
            if SLEEP_KEYWORD in transcription_normalized:
                client_mode = "passive"
                _LOGGER.info("Sleep keyword detected ('%s'). Switching to PASSIVE mode.", SLEEP_KEYWORD)
            else:
                _LOGGER.info("Active transcription: %s", transcription.strip())
            if last_active_time is not None:
                elapsed = current_time - last_active_time
                _LOGGER.debug("Elapsed time since last active: %.2f seconds", elapsed)
                if elapsed > ACTIVE_TIMEOUT:
                    client_mode = "passive"
                    _LOGGER.info("Inactivity timeout (%.0f sec) reached (elapsed %.2f sec). Reverting to PASSIVE mode.", ACTIVE_TIMEOUT, elapsed)
        await asyncio.sleep(0.1)

# --- Event Processing Functions ---
async def process_state_changed_event(event_data):
    _LOGGER.info("Processing state_changed event: %s", event_data)
    event = event_data.get("event", {})
    if event.get("event_type") != "state_changed":
        return

    new_state = event.get("data", {}).get("new_state", {})
    attributes = new_state.get("attributes", {})
    command = attributes.get("command")
    if command == "update_config":
        new_tts_engine = attributes.get("tts_engine")
        if not new_tts_engine:
            _LOGGER.error("No new TTS engine specified in the update_config command.")
            return
        _LOGGER.info("Processing update_config command with new TTS engine: %s", new_tts_engine)
        try:
            # Load existing config
            with open(CONFIG_FILE, "r") as f:
                config_data = json.load(f)
            _LOGGER.debug("Current config before update: %s", json.dumps(config_data))
            # Update the TTS engine value
            config_data["tts_engine"] = new_tts_engine
            # Write the updated config back to the file
            with open(CONFIG_FILE, "w") as f:
                json.dump(config_data, f, indent=4)
            _LOGGER.info("Configuration file updated successfully with new TTS engine: %s", new_tts_engine)
            # Update the globals so the change is reflected immediately
            global config, TTS_ENGINE
            config = config_data
            TTS_ENGINE = new_tts_engine
        except Exception as e:
            _LOGGER.error("Failed to update configuration file: %s", e)
    else:
        _LOGGER.debug("Ignoring state_changed event with command: %s", command)

async def process_call_service_event(event_data):
    _LOGGER.info("Processing call_service event: %s", event_data)
    service_data = event_data.get("event", {}).get("data", {})
    domain = service_data.get("domain")
    service = service_data.get("service")
    if domain != "media_player":
        _LOGGER.info("Ignoring call_service event from domain: %s", domain)
        return
    async with aiohttp.ClientSession() as session:
        if service in ["play_media", "media_play"]:
            media_content_id = service_data.get("service_data", {}).get("media_content_id")
            resolved_url = await resolve_tts_url(session, media_content_id)
            await play_media(resolved_url)
        elif service == "media_stop":
            await stop_media()
        elif service == "volume_set":
            volume_level = service_data.get("service_data", {}).get("volume_level", 0.5)
            await set_volume(volume_level)
        elif service == "media_pause":
            await pause_media()
        elif service == "media_resume":
            await resume_media()
        else:
            _LOGGER.info("Unhandled service: %s", service)

async def process_event(event_data):
    if event_data.get("type") != "event":
        return
    event_type = event_data.get("event", {}).get("event_type")
    if event_type == "state_changed":
        await process_state_changed_event(event_data)
    elif event_type == "call_service":
        await process_call_service_event(event_data)
    else:
        _LOGGER.debug("Received unsupported event type: %s", event_type)

async def listen_for_events():
    async with aiohttp.ClientSession() as session:
        await register_media_player(session)
        async with session.ws_connect(WS_API_URL) as ws:
            await ws.send_json({"type": "auth", "access_token": TOKEN})
            await ws.receive_json()  # Auth response
            await ws.send_json({"id": 1, "type": "subscribe_events", "event_type": "state_changed"})
            await ws.send_json({"id": 2, "type": "subscribe_events", "event_type": "call_service"})
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    await process_event(data)
                else:
                    _LOGGER.debug("Received non-text message: %s", msg)

# --- Main ---
async def main():
    global client_mode, last_active_time
    client_mode = "passive"
    last_active_time = None
    _LOGGER.info("🚀 Soundhive Client v%s - Combined TTS, Media Playback, STT & Dynamic Wake/Sleep (WAV)", VERSION)

    async with aiohttp.ClientSession() as session:
        await register_media_player(session)
    # Run both the continuous STT loop and the event listener concurrently
    await asyncio.gather(
        continuous_stt_loop(),
        listen_for_events()
    )

if __name__ == "__main__":
    asyncio.run(main())
