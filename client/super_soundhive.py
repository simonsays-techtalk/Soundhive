#!/usr/bin/env python3
import asyncio
import aiohttp
import json
import logging
import os
import sys
import time
import vlc
import sounddevice as sd
import soundfile as sf
import string
import numpy as np
import io
import queue
import threading
import signal
import concurrent.futures
import difflib
import re
import base64
from urllib.parse import urlparse, parse_qs, quote_plus  # Added quote_plus for URL encoding

# Added imports for decryption.
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from cryptography.fernet import Fernet

# --- Configuration and Global Constants ---
CONFIG_FILE = "soundhive_config.json"
VERSION = "2.5.20 Media playback, TTS, and threaded streaming STT with enhanced inter-thread communication"
MEDIA_PLAYER_ENTITY = "media_player.soundhive_media_player"
COOLDOWN_PERIOD = 2  # seconds cooldown after TTS finishes
STT_QUEUE_MAXSIZE = 50  # Maximum size for the STT priority queue

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger("SoundhiveClient")

# --- Global variables ---
stt_priority_queue = None  # Global STT queue, will be assigned in main()
tts_finished_time = 0        # Timestamp when TTS playback finished
last_tts_message = ""        # Initialize last TTS message (used for filtering)
last_llm_command = ""        # Last command processed by the LLM

# --- Load Config and Decrypt ---
def load_config():
    try:
        with open(CONFIG_FILE, "r") as f:
            encrypted_config = json.load(f)
    except Exception as e:
        _LOGGER.error("Error loading configuration file: %s", e)
        sys.exit(1)
    try:
        salt_b64 = encrypted_config["salt"]
        encrypted_data = encrypted_config["data"]
        salt = base64.b64decode(salt_b64)
    except KeyError as e:
        _LOGGER.error("Invalid config file format: missing key %s", e)
        sys.exit(1)
    master_password = os.getenv("MASTER_PASS")
    if not master_password:
        try:
            master_password = input("Enter master password to decrypt configuration: ").strip()
        except EOFError:
            _LOGGER.error("No interactive input available and MASTER_PASS not set.")
            sys.exit(1)
    password_bytes = master_password.encode()
    kdf = PBKDF2HMAC(
         algorithm=hashes.SHA256(),
         length=32,
         salt=salt,
         iterations=100000,
         backend=default_backend()
    )
    key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
    fernet = Fernet(key)
    try:
        decrypted_json = fernet.decrypt(encrypted_data.encode())
        config_data = json.loads(decrypted_json.decode())
        _LOGGER.info("Configuration decrypted and loaded successfully.")
        return config_data
    except Exception as e:
        _LOGGER.error("Failed to decrypt configuration: %s", e)
        sys.exit(1)

config = load_config()
HA_BASE_URL = config.get("ha_url")
TOKEN = config.get("auth_token")
TTS_ENGINE = config.get("tts_engine", "tts.google_translate_en_com")
STT_URI = config.get("stt_uri")
STT_FORMAT = config.get("stt_format", "wav")
VOLUME_SETTING = config.get("volume", 0.2)
RMS_THRESHOLD = float(config.get("rms_threshold", "0.001"))

# Keyword and timeout settings:
WAKE_KEYWORD = config.get("wake_keyword", "hey assistant")
SLEEP_KEYWORD = config.get("sleep_keyword", "goodbye")
ALARM_KEYWORD = config.get("alarm_keyword", "alarm now")
CLEAR_ALARM_KEYWORD = config.get("clear_alarm_keyword", "clear alarm")
ACTIVE_TIMEOUT = config.get("active_timeout", 5)
LLM_URI = config.get("llm_uri")  # New config value for the LLM endpoint

WS_API_URL = f"{HA_BASE_URL}/api/websocket"
REGISTER_ENTITY_API = f"{HA_BASE_URL}/api/states/{MEDIA_PLAYER_ENTITY}"
TTS_API_URL = f"{HA_BASE_URL}/api/tts_get_url"

# --- Helper Functions ---
def normalize_text(text):
    translator = str.maketrans("", "", string.punctuation)
    return text.lower().translate(translator).strip()

def keyword_in_text(text, keyword):
    pattern = r'\b' + re.escape(keyword.lower().strip()) + r'\b'
    return re.search(pattern, text) is not None

def trigger_alarm():
    _LOGGER.error("🚨 ALARM triggered! Notifying emergency services...")
    # TODO: Implement emergency services notification logic here.

async def clear_stt_queue(queue_obj):
    while not queue_obj.empty():
        try:
            queue_obj.get_nowait()
        except asyncio.QueueEmpty:
            break

def normalize_tts_text(text):
    return re.sub(r'\*+', '', text)

# VLC event callback.
def on_media_end(event):
    global tts_finished_time
    tts_finished_time = time.monotonic()
    _LOGGER.info("Media playback finished (VLC event); setting cooldown timestamp.")

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
        volume = VOLUME_SETTING
    headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
    payload = {
        "state": state,
        "attributes": {
            "friendly_name": "Soundhive Media Player",
            "supported_features": 52735,
            "media_content_id": media_url if media_url is not None else "",
            "media_content_type": "music" if media_url else None,
            "volume_level": volume,
            "media_library": MEDIA_LIBRARY
        }
    }
    _LOGGER.info("📡 Sending state update: %s", payload)
    async with aiohttp.ClientSession() as session:
        async with session.post(REGISTER_ENTITY_API, headers=headers, json=payload) as resp:
            if resp.status in [200, 201]:
                _LOGGER.info("✅ Media player state updated: %s", state)
            else:
                _LOGGER.error("❌ Failed to update media player state. Status: %s", resp.status)
    await asyncio.sleep(1)

async def resolve_tts_url(session, media_content_id):
    global config, TTS_ENGINE
    config = load_config()
    TTS_ENGINE = config.get("tts_engine", "tts.google_translate_en_com")
    if media_content_id is None:
        _LOGGER.error("❌ Received a NoneType media_content_id. Cannot resolve TTS URL.")
        return None
    if media_content_id.startswith("media-source://tts/"):
        try:
            query = media_content_id.split("?", 1)[1] if "?" in media_content_id else ""
            params = parse_qs(query)
            message = params.get("message", [""])[0]
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
    return media_content_id

async def process_llm_command(command_text):
    if not LLM_URI:
        _LOGGER.error("LLM endpoint not defined in configuration.")
        return None
    try:
        async with aiohttp.ClientSession() as session:
            payload = {"model": "llama3.1:8b", "prompt": command_text}
            _LOGGER.info("Sending command to LLM: %s", command_text)
            async with session.post(LLM_URI, json=payload) as resp:
                if resp.status != 200:
                    _LOGGER.error("LLM call failed with status: %s", resp.status)
                    return None
                full_response = ""
                async for line in resp.content:
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line.decode('utf-8').strip())
                    except Exception as e:
                        _LOGGER.error("Error decoding streamed line: %s", e)
                        continue
                    token = chunk.get("response", "")
                    full_response += token
                    if chunk.get("done", False):
                        break
                _LOGGER.info("LLM full response: %s", full_response)
                return full_response
    except Exception as e:
        _LOGGER.error("Error calling LLM: %s", e)
        return None

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
    event_manager = current_player.event_manager()
    event_manager.event_attach(vlc.EventType.MediaPlayerEndReached, on_media_end)
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
            current_player.pause()
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

def stream_audio_chunks(chunk_duration=4, samplerate=16000, channels=1):
    audio_queue = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        if status:
            _LOGGER.warning("Audio callback status: %s", status)
        audio_queue.put(indata.copy())

    with sd.InputStream(samplerate=samplerate, channels=channels, callback=audio_callback):
        while True:
            frames = []
            start_time = time.monotonic()
            while time.monotonic() - start_time < chunk_duration:
                frame = audio_queue.get()
                frames.append(frame)
            chunk_data = np.concatenate(frames, axis=0)
            rms = np.sqrt(np.mean(chunk_data**2))
            if rms < RMS_THRESHOLD:
                _LOGGER.debug("Detected silence (RMS=%.6f); skipping this chunk.", rms)
                continue
            buffer = io.BytesIO()
            sf.write(buffer, chunk_data, samplerate, format='WAV')
            buffer.seek(0)
            yield buffer

async def async_stream_audio_chunks(chunk_duration=4, samplerate=16000, channels=1):
    loop = asyncio.get_event_loop()
    gen = stream_audio_chunks(chunk_duration, samplerate, channels)
    while True:
        chunk = await loop.run_in_executor(None, next, gen)
        yield chunk

async def stream_transcriptions(stt_uri, chunk_duration=4, samplerate=16000, channels=1):
    async for audio_buffer in async_stream_audio_chunks(chunk_duration, samplerate, channels):
        async with aiohttp.ClientSession() as session:
            form = aiohttp.FormData()
            form.add_field("file",
                           audio_buffer.read(),
                           filename="audio_chunk.wav",
                           content_type="audio/wav")
            try:
                async with session.post(stt_uri, data=form) as resp:
                    if resp.status != 200:
                        _LOGGER.error("Error sending chunk to STT server: %s", resp.status)
                        transcription = ""
                    else:
                        transcription = await resp.text()
                    yield transcription
            except Exception as e:
                _LOGGER.error("Exception sending audio chunk: %s", e)
                yield ""

def stt_thread_func(main_loop, stt_priority_queue):
    async def stt_processor():
        async for transcription in stream_transcriptions(STT_URI, chunk_duration=4):
            try:
                result = json.loads(transcription)
                transcription_text = result.get("text", "")
            except Exception as e:
                _LOGGER.error("Error parsing STT transcription: %s", e)
                transcription_text = transcription

            _LOGGER.info("Raw transcription: %s", transcription_text)
            norm_text = normalize_text(transcription_text)
            _LOGGER.info("Normalized transcription: %s", norm_text)

            if current_player is not None and current_player.get_state() == vlc.State.Playing:
                continue

            if not norm_text or norm_text in ["blankaudio", "silence", "inaudible"]:
                _LOGGER.debug("Skipping blank transcription: '%s'", norm_text)
                continue

            if (keyword_in_text(norm_text, WAKE_KEYWORD) or 
                keyword_in_text(norm_text, SLEEP_KEYWORD) or 
                keyword_in_text(norm_text, ALARM_KEYWORD) or 
                keyword_in_text(norm_text, CLEAR_ALARM_KEYWORD)):
                priority = 0
            else:
                priority = 1

            current_queue_size = stt_priority_queue.qsize()
            if current_queue_size >= STT_QUEUE_MAXSIZE and priority > 0:
                _LOGGER.warning("STT queue full (size=%s); dropping non-urgent message: %s", current_queue_size, norm_text)
                continue

            try:
                asyncio.run_coroutine_threadsafe(
                    stt_priority_queue.put((priority, transcription_text)),
                    main_loop
                )
            except Exception as ex:
                _LOGGER.error("Error pushing transcription to queue: %s", ex)
    asyncio.run(stt_processor())

async def continuous_stt_loop(stt_priority_queue):
    global client_mode, last_active_time, last_tts_message, last_llm_command
    _LOGGER.info("Starting continuous STT loop (initially in PASSIVE mode).")
    client_mode = "passive"
    last_active_time = None
    while True:
        if current_player is not None and current_player.get_state() == vlc.State.Playing:
            _LOGGER.debug("Media is playing; skipping STT processing.")
            await asyncio.sleep(0.5)
            continue

        if time.monotonic() - tts_finished_time < COOLDOWN_PERIOD:
            remaining = COOLDOWN_PERIOD - (time.monotonic() - tts_finished_time)
            _LOGGER.debug("Within cooldown period (%.2f sec remaining); skipping STT processing.", remaining)
            await asyncio.sleep(0.5)
            continue

        try:
            priority, transcription = await asyncio.wait_for(stt_priority_queue.get(), timeout=0.5)
            transcription_normalized = normalize_text(transcription).strip()
            current_time = time.monotonic()
            _LOGGER.debug("Received transcription: %r", transcription_normalized)

            if transcription_normalized == normalize_text(last_tts_message).strip():
                _LOGGER.debug("Transcription matches last TTS message; ignoring.")
                continue

            if client_mode == "passive":
                if keyword_in_text(transcription_normalized, WAKE_KEYWORD):
                    client_mode = "active"
                    last_active_time = current_time
                    _LOGGER.info("Wake keyword detected: %r. Switching from PASSIVE to ACTIVE mode.", WAKE_KEYWORD)
                else:
                    _LOGGER.info("Passive transcription (ignored): %s", transcription.strip())
            elif client_mode == "active":
                if transcription_normalized and transcription_normalized not in ["blankaudio", "silence", "inaudible"]:
                    last_active_time = current_time
                if keyword_in_text(transcription_normalized, SLEEP_KEYWORD):
                    client_mode = "passive"
                    _LOGGER.info("Sleep keyword detected: %r. Switching from ACTIVE to PASSIVE mode.", SLEEP_KEYWORD)
                else:
                    _LOGGER.info("Active transcription: %s", transcription.strip())
                    if transcription_normalized != last_llm_command:
                        last_llm_command = transcription_normalized
                        response = await process_llm_command(transcription_normalized)
                        if response:
                            response = normalize_tts_text(response)
                            encoded_message = quote_plus(response)
                            tts_media_content = f"media-source://tts/?message={encoded_message}"
                            async with aiohttp.ClientSession() as session:
                                resolved_url = await resolve_tts_url(session, tts_media_content)
                            if resolved_url:
                                await play_media(resolved_url)
                if last_active_time is not None:
                    elapsed = current_time - last_active_time
                    _LOGGER.debug("Elapsed time since last active input: %.2f sec", elapsed)
                    if elapsed > ACTIVE_TIMEOUT:
                        client_mode = "passive"
                        _LOGGER.info("Inactivity timeout (%.0f sec) reached (elapsed %.2f sec). Switching to PASSIVE mode.", ACTIVE_TIMEOUT, elapsed)
        except asyncio.TimeoutError:
            if client_mode == "active" and last_active_time is not None:
                current_time = time.monotonic()
                elapsed = current_time - last_active_time
                _LOGGER.debug("No new transcription. Elapsed time since last active input: %.2f sec", elapsed)
                if elapsed > ACTIVE_TIMEOUT:
                    client_mode = "passive"
                    _LOGGER.info("Inactivity timeout (%.0f sec) reached during idle check (elapsed %.2f sec). Switching to PASSIVE mode.", ACTIVE_TIMEOUT, elapsed)
            await asyncio.sleep(0.1)

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
            _LOGGER.error("No new TTS engine specified in update_config command.")
            return
        _LOGGER.info("Processing update_config command with new TTS engine: %s", new_tts_engine)
        try:
            with open(CONFIG_FILE, "r") as f:
                config_data = json.load(f)
            _LOGGER.debug("Current config before update: %s", json.dumps(config_data))
            config_data["tts_engine"] = new_tts_engine
            with open(CONFIG_FILE, "w") as f:
                json.dump(config_data, f, indent=4)
            _LOGGER.info("Configuration file updated successfully with new TTS engine: %s", new_tts_engine)
            global config, TTS_ENGINE
            config = config_data
            TTS_ENGINE = new_tts_engine
        except Exception as e:
            _LOGGER.error("Failed to update configuration file: %s", e)
    else:
        _LOGGER.debug("Ignoring state_changed event with command: %s", command)
    entity_id = new_state.get("entity_id", "")
    if entity_id == MEDIA_PLAYER_ENTITY and new_state.get("state") in ["idle", "stopped"]:
        await clear_stt_queue(stt_priority_queue)
        _LOGGER.info("Media playback finished; STT processing resumed.")

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
            if media_content_id and ("media-source://tts/" in media_content_id or "tts_proxy" in media_content_id or
                                      (resolved_url and "tts_proxy" in resolved_url)):
                global last_tts_message
                query = media_content_id.split("?", 1)[1] if "?" in media_content_id else ""
                params = parse_qs(query)
                last_tts_message = params.get("message", [""])[0]
                _LOGGER.info("TTS playback detected; STT processing will resume when media finishes.")
                await clear_stt_queue(stt_priority_queue)
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
            await ws.receive_json()
            await ws.send_json({"id": 1, "type": "subscribe_events", "event_type": "state_changed"})
            await ws.send_json({"id": 2, "type": "subscribe_events", "event_type": "call_service"})
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    await process_event(data)
                else:
                    _LOGGER.debug("Received non-text message: %s", msg)

async def log_client_mode():
    while True:
        _LOGGER.info("Current client mode: %s", client_mode)
        await asyncio.sleep(5)

shutdown_event = asyncio.Event()

def shutdown_handler():
    _LOGGER.info("Shutdown signal received. Cancelling tasks...")
    shutdown_event.set()

async def main():
    global client_mode, last_active_time, stt_priority_queue
    client_mode = "passive"
    last_active_time = None
    _LOGGER.info("🚀 Soundhive Client v%s - Combined TTS, Media Playback, STT, and LLM Integration", VERSION)
    async with aiohttp.ClientSession() as session:
        await register_media_player(session)
    stt_priority_queue = asyncio.PriorityQueue(maxsize=STT_QUEUE_MAXSIZE)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    main_loop = asyncio.get_running_loop()
    stt_thread = threading.Thread(target=stt_thread_func, args=(main_loop, stt_priority_queue), daemon=True)
    stt_thread.start()
    for signame in ('SIGINT', 'SIGTERM'):
        main_loop.add_signal_handler(getattr(signal, signame), shutdown_handler)
    tasks = [
        asyncio.create_task(continuous_stt_loop(stt_priority_queue)),
        asyncio.create_task(listen_for_events()),
        asyncio.create_task(log_client_mode())
    ]
    await shutdown_event.wait()
    _LOGGER.info("Cancelling tasks...")
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    executor.shutdown(wait=True)
    _LOGGER.info("Shutdown complete.")

if __name__ == "__main__":
    asyncio.run(main())
