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
from urllib.parse import urlparse, parse_qs, quote_plus  # For URL encoding

# --- Configuration and Global Constants ---
CONFIG_FILE = "soundhive_config.json"
VERSION = "2.5.70 Media playback, TTS, and threaded streaming STT with enhanced inter-thread communication"
MEDIA_PLAYER_ENTITY = "media_player.soundhive_media_player"
COOLDOWN_PERIOD = 2           # seconds cooldown after TTS finishes
STT_QUEUE_MAXSIZE = 50        # Maximum size for the STT priority queue

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger("SoundhiveClient")

# --- Global Variables ---
stt_priority_queue = None     # Will be set in main()
tts_finished_time = 0           # Timestamp when TTS playback finished
last_tts_message = ""           # Last TTS message (for filtering)
last_llm_command = ""           # Last LLM command processed
client_mode = "passive"         # Global client mode

# New globals for pending learn buffering.
pending_learn = None            # Buffer (string) to accumulate learn command content
pending_learn_timestamp = None  # Timestamp of the last pending learn update
pending_learn_items = []        # List of learned facts pending commit

# --- Load Config (Plain JSON) ---
def load_config():
    try:
        with open(CONFIG_FILE, "r") as f:
            config_data = json.load(f)
        _LOGGER.info("Configuration loaded successfully.")
        return config_data
    except Exception as e:
        _LOGGER.error("Error loading configuration file: %s", e)
        sys.exit(1)

config = load_config()
# Ensure a unique ID is present for the client.
if "unique_id" not in config:
    friendly_name = config.get("name", "soundhive_media_player").strip().lower().replace(" ", "_")
    config["unique_id"] = friendly_name
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)
unique_id = config["unique_id"]

HA_BASE_URL = config.get("ha_url")
TOKEN = config.get("auth_token")
TTS_ENGINE = config.get("tts_engine", "tts.google_translate_en_com")
STT_URI = config.get("stt_uri")
STT_FORMAT = config.get("stt_format", "wav")
VOLUME_SETTING = config.get("volume", 0.2)
RMS_THRESHOLD = float(config.get("rms_threshold", "0.001"))

# Keyword and Special Command settings
WAKE_KEYWORD = config.get("wake_keyword", "hey assistant")
SLEEP_KEYWORD = config.get("sleep_keyword", "goodbye")
ALARM_KEYWORD = config.get("alarm_keyword", "alarm now")
CLEAR_ALARM_KEYWORD = config.get("clear_alarm_keyword", "clear alarm")
LEARN_COMMAND = config.get("learn_command", "learn this:")
FORGET_COMMAND = config.get("forget_command", "forget this:")
COMMIT_CODE = config.get("commit_code", "1234")
ACTIVE_TIMEOUT = config.get("active_timeout", 5)
LLM_URI = config.get("llm_uri")  # LLM endpoint

WS_API_URL = f"{HA_BASE_URL}/api/websocket"
REGISTER_ENTITY_API = f"{HA_BASE_URL}/api/states/media_player.{unique_id}"
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
    # TODO: Implement notification logic here.

async def clear_stt_queue(queue_obj):
    while not queue_obj.empty():
        try:
            queue_obj.get_nowait()
        except asyncio.QueueEmpty:
            break

def normalize_tts_text(text):
    return re.sub(r'\*+', '', text)

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

REGISTER_ENTITY_API = f"{HA_BASE_URL}/api/states/media_player.{unique_id}"

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

def on_media_end(event):
    global tts_finished_time
    tts_finished_time = time.monotonic()
    _LOGGER.info("Media playback finished (VLC event); setting cooldown timestamp.")

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
            form.add_field("file", audio_buffer.read(), filename="audio_chunk.wav", content_type="audio/wav")
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

# --- stt_thread_func Definition ---
def stt_thread_func(main_loop, stt_priority_queue):
    async def stt_processor():
        global client_mode, last_active_time, last_tts_message, last_llm_command
        global pending_learn, pending_learn_timestamp, pending_learn_items
        while True:
            try:
                async for transcription in stream_transcriptions(STT_URI, chunk_duration=4):
                    current_time = time.monotonic()
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
                    if not norm_text or norm_text in ["blankaudio", "silence", "indistinct chatter", "inaudible"]:
                        _LOGGER.debug("Skipping blank transcription: '%s'", norm_text)
                        continue
                    if pending_learn is not None:
                        if not (norm_text.startswith(normalize_text(LEARN_COMMAND).replace(":", "")) or
                                norm_text.startswith(normalize_text(FORGET_COMMAND).replace(":", "")) or
                                norm_text.startswith("commit changes:") or
                                keyword_in_text(norm_text, SLEEP_KEYWORD) or
                                keyword_in_text(norm_text, WAKE_KEYWORD)):
                            if current_time - pending_learn_timestamp < LEARN_BUFFER_TIMEOUT:
                                pending_learn += " " + transcription_text.strip()
                                pending_learn_timestamp = current_time
                                _LOGGER.info("Appended utterance to pending learn command: %s", pending_learn)
                                continue
                            else:
                                if pending_learn.strip():
                                    pending_learn_items.append(pending_learn.strip())
                                    _LOGGER.info("Buffered learn command stored locally: %s", pending_learn.strip())
                                pending_learn = None
                                pending_learn_timestamp = None
                    if client_mode == "passive":
                        if keyword_in_text(norm_text, WAKE_KEYWORD):
                            client_mode = "active"
                            last_active_time = current_time
                            _LOGGER.info("Wake keyword detected: %r. Switching from PASSIVE to ACTIVE mode.", WAKE_KEYWORD)
                        else:
                            _LOGGER.info("Passive transcription (ignored): %s", transcription.strip())
                        continue
                    if client_mode == "active" and norm_text == WAKE_KEYWORD:
                        _LOGGER.info("Wake keyword received in active mode; ignoring.")
                        continue
                    if norm_text.startswith("retrieve pending"):
                        if pending_learn_items:
                            pending_text = " ; ".join(pending_learn_items)
                            _LOGGER.info("Retrieving pending learn items: %s", pending_text)
                            tts_message = f"Pending entries: {pending_text}"
                        else:
                            _LOGGER.info("No pending learn items.")
                            tts_message = "There are no pending entries."
                        encoded_message = quote_plus(tts_message)
                        tts_media_content = f"media-source://tts/?message={encoded_message}"
                        async with aiohttp.ClientSession() as session:
                            resolved_url = await resolve_tts_url(session, tts_media_content)
                        if resolved_url:
                            await play_media(resolved_url)
                        continue
                    learn_prefix = normalize_text(LEARN_COMMAND).replace(":", "")
                    if norm_text.startswith(learn_prefix):
                        content = norm_text[len(learn_prefix):].strip(" :")
                        if not content:
                            pending_learn = ""
                            pending_learn_timestamp = current_time
                            _LOGGER.info("Learn command detected with no content; buffering input.")
                            continue
                        else:
                            if pending_learn is not None:
                                content = (pending_learn + " " + content).strip()
                                pending_learn = None
                                pending_learn_timestamp = None
                            pending_learn_items.append(content)
                            _LOGGER.info("Learn command stored pending commit: %s", content)
                            continue
                    forget_prefix = normalize_text(FORGET_COMMAND).replace(":", "")
                    if norm_text.startswith(forget_prefix):
                        forget_content = norm_text[len(forget_prefix):].strip(" :")
                        if forget_content:
                            if forget_content in pending_learn_items:
                                pending_learn_items.remove(forget_content)
                                _LOGGER.info("Pending learn entry '%s' removed.", forget_content)
                            await handle_forget_command(forget_content)
                        else:
                            _LOGGER.error("Forget command received but no fact content was found.")
                        continue
                    if norm_text.startswith("commit changes:"):
                        commit_parts = transcription.split(":", 1)
                        if len(commit_parts) == 2 and commit_parts[1].strip() == COMMIT_CODE:
                            for item in pending_learn_items:
                                await handle_learn_command(item)
                            pending_learn_items.clear()
                            _LOGGER.info("All pending learn items committed to the DB.")
                        else:
                            _LOGGER.error("Commit code verification failed.")
                        continue
                    if keyword_in_text(norm_text, SLEEP_KEYWORD):
                        client_mode = "passive"
                        _LOGGER.info("Sleep keyword detected: %r. Switching from ACTIVE to PASSIVE mode.", SLEEP_KEYWORD)
                        continue
                    _LOGGER.info("Active transcription: %s", transcription.strip())
                    if norm_text != last_llm_command:
                        last_llm_command = norm_text
                        response = await process_llm_command(norm_text)
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
    asyncio.run(stt_processor())

# --- Command Handlers ---
async def handle_learn_command(content):
    if not content.strip():
        _LOGGER.error("Empty fact provided; skipping learn command.")
        return
    payload = {"fact": content}
    chromadb_url = config.get("chromadb_url")
    _LOGGER.info("Sending learn command with content: %s", content)
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(f"{chromadb_url}/append_log", json=payload, headers={"Authorization": f"Bearer {TOKEN}"}) as resp:
                if resp.status == 200:
                    _LOGGER.info("Learn command processed successfully.")
                else:
                    _LOGGER.error("Learn command failed with status: %s", resp.status)
        except Exception as e:
            _LOGGER.error("Exception during learn command: %s", e)

async def handle_forget_command(content):
    payload = {"action": "forget", "content": content}
    chromadb_url = config.get("chromadb_url")
    _LOGGER.info("Sending forget command for content: %s", content)
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(f"{chromadb_url}/forget", json=payload, headers={"Authorization": f"Bearer {TOKEN}"}) as resp:
                if resp.status == 200:
                    _LOGGER.info("Forget command processed successfully.")
                else:
                    _LOGGER.error("Forget command failed with status: %s", resp.status)
        except Exception as e:
            _LOGGER.error("Exception during forget command: %s", e)

async def handle_commit_command():
    payload = {"action": "commit"}
    chromadb_url = config.get("chromadb_url")
    _LOGGER.info("Sending commit command to DB.")
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(f"{chromadb_url}/commit", json=payload, headers={"Authorization": f"Bearer {TOKEN}"}) as resp:
                if resp.status == 200:
                    _LOGGER.info("Commit command processed successfully.")
                else:
                    _LOGGER.error("Commit command failed with status: %s", resp.status)
        except Exception as e:
            _LOGGER.error("Exception during commit command: %s", e)

async def scheduled_commit(target_hour=2, target_minute=0):
    while True:
        now = time.localtime()
        target = time.mktime((now.tm_year, now.tm_mon, now.tm_mday, target_hour, target_minute, 0,
                                now.tm_wday, now.tm_yday, now.tm_isdst))
        if target < time.mktime(now):
            target += 86400
        wait_seconds = target - time.mktime(now)
        _LOGGER.info("Waiting %.0f seconds for scheduled commit.", wait_seconds)
        await asyncio.sleep(wait_seconds)
        await handle_commit_command()

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
    if domain == "homeassistant" and service == "restart":
        _LOGGER.info("Detected Home Assistant restart; waiting for HA to be ready before re-registering.")
        await asyncio.sleep(10)
        async with aiohttp.ClientSession() as session:
            await register_media_player(session)
        return
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
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                await register_media_player(session)
                _LOGGER.info("Attempting to connect to HA WebSocket...")
                async with session.ws_connect(WS_API_URL) as ws:
                    initial_msg = await ws.receive_json()
                    if initial_msg.get("type") == "auth_required":
                        await ws.send_json({"type": "auth", "access_token": TOKEN})
                        auth_response = await ws.receive_json()
                        if auth_response.get("type") != "auth_ok":
                            _LOGGER.error("WebSocket authentication failed: %s", auth_response)
                            return
                    else:
                        _LOGGER.error("Unexpected initial message: %s", initial_msg)
                        return
                    await ws.send_json({"id": 1, "type": "subscribe_events", "event_type": "state_changed"})
                    await ws.send_json({"id": 2, "type": "subscribe_events", "event_type": "call_service"})
                    _LOGGER.info("WebSocket connection established and subscribed to events.")
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            await process_event(data)
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            _LOGGER.error("WebSocket connection closed or error occurred. Reconnecting...")
                            break
        except Exception as e:
            _LOGGER.error("Error in WebSocket connection: %s. Reconnecting...", e)
        await asyncio.sleep(5)

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
        asyncio.create_task(log_client_mode()),
        asyncio.create_task(scheduled_commit(target_hour=2, target_minute=0))
    ]
    await shutdown_event.wait()
    _LOGGER.info("Cancelling tasks...")
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    executor.shutdown(wait=True)
    _LOGGER.info("Shutdown complete.")

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
        except asyncio.TimeoutError:
            if client_mode == "active" and last_active_time is not None:
                current_time = time.monotonic()
                elapsed = current_time - last_active_time
                _LOGGER.debug("No new transcription. Elapsed time since last active input: %.2f sec", elapsed)
                if elapsed > ACTIVE_TIMEOUT:
                    client_mode = "passive"
                    _LOGGER.info("Inactivity timeout (%.0f sec) reached during idle check (elapsed %.2f sec). Switching to PASSIVE mode.", ACTIVE_TIMEOUT, elapsed)
            await asyncio.sleep(0.1)
            continue
        transcription_normalized = normalize_text(transcription).strip()
        current_time = time.monotonic()
        _LOGGER.debug("Received transcription: %r", transcription_normalized)
        if transcription_normalized == normalize_text(last_tts_message).strip():
            _LOGGER.debug("Transcription matches last TTS message; ignoring.")
            continue
        global pending_learn, pending_learn_timestamp
        if pending_learn is not None:
            if not (transcription_normalized.startswith(normalize_text(LEARN_COMMAND).replace(":", "")) or
                    transcription_normalized.startswith(normalize_text(FORGET_COMMAND).replace(":", "")) or
                    transcription_normalized.startswith("commit changes:") or
                    keyword_in_text(transcription_normalized, SLEEP_KEYWORD) or
                    keyword_in_text(transcription_normalized, WAKE_KEYWORD)):
                if current_time - pending_learn_timestamp < LEARN_BUFFER_TIMEOUT:
                    pending_learn += " " + transcription.strip()
                    pending_learn_timestamp = current_time
                    _LOGGER.info("Appended utterance to pending learn command: %s", pending_learn)
                    continue
                else:
                    if pending_learn.strip():
                        pending_learn_items.append(pending_learn.strip())
                        _LOGGER.info("Buffered learn command stored locally: %s", pending_learn.strip())
                    pending_learn = None
                    pending_learn_timestamp = None
        if client_mode == "passive":
            if keyword_in_text(transcription_normalized, WAKE_KEYWORD):
                client_mode = "active"
                last_active_time = current_time
                _LOGGER.info("Wake keyword detected: %r. Switching from PASSIVE to ACTIVE mode.", WAKE_KEYWORD)
            else:
                _LOGGER.info("Passive transcription (ignored): %s", transcription.strip())
            continue
        if client_mode == "active" and transcription_normalized == WAKE_KEYWORD:
            _LOGGER.info("Wake keyword received in active mode; ignoring.")
            continue
        if transcription_normalized.startswith("retrieve pending"):
            if pending_learn_items:
                pending_text = " ; ".join(pending_learn_items)
                _LOGGER.info("Retrieving pending learn items: %s", pending_text)
                tts_message = f"Pending entries: {pending_text}"
            else:
                _LOGGER.info("No pending learn items.")
                tts_message = "There are no pending entries."
            encoded_message = quote_plus(tts_message)
            tts_media_content = f"media-source://tts/?message={encoded_message}"
            async with aiohttp.ClientSession() as session:
                resolved_url = await resolve_tts_url(session, tts_media_content)
            if resolved_url:
                await play_media(resolved_url)
            continue
        learn_prefix = normalize_text(LEARN_COMMAND).replace(":", "")
        if transcription_normalized.startswith(learn_prefix):
            content = transcription_normalized[len(learn_prefix):].strip(" :")
            if not content:
                pending_learn = ""
                pending_learn_timestamp = current_time
                _LOGGER.info("Learn command detected with no content; buffering input.")
                continue
            else:
                if pending_learn is not None:
                    content = (pending_learn + " " + content).strip()
                    pending_learn = None
                    pending_learn_timestamp = None
                pending_learn_items.append(content)
                _LOGGER.info("Learn command stored pending commit: %s", content)
                continue
        forget_prefix = normalize_text(FORGET_COMMAND).replace(":", "")
        if transcription_normalized.startswith(forget_prefix):
            forget_content = transcription_normalized[len(forget_prefix):].strip(" :")
            if forget_content:
                if forget_content in pending_learn_items:
                    pending_learn_items.remove(forget_content)
                    _LOGGER.info("Pending learn entry '%s' removed.", forget_content)
                await handle_forget_command(forget_content)
            else:
                _LOGGER.error("Forget command received but no fact content was found.")
            continue
        if transcription_normalized.startswith("commit changes:"):
            commit_parts = transcription.split(":", 1)
            if len(commit_parts) == 2 and commit_parts[1].strip() == COMMIT_CODE:
                for item in pending_learn_items:
                    await handle_learn_command(item)
                pending_learn_items.clear()
                _LOGGER.info("All pending learn items committed to the DB.")
            else:
                _LOGGER.error("Commit code verification failed.")
            continue
        if keyword_in_text(transcription_normalized, SLEEP_KEYWORD):
            client_mode = "passive"
            _LOGGER.info("Sleep keyword detected: %r. Switching from ACTIVE to PASSIVE mode.", SLEEP_KEYWORD)
            continue
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
    asyncio.run(stt_processor())

if __name__ == "__main__":
    asyncio.run(main())
