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
import requests
import threading
import signal
import concurrent.futures
import re
import base64
import uuid
from urllib.parse import parse_qs, quote_plus

# --- Configuration and Global Constants ---
CONFIG_FILE = "soundhive_config.json"
VERSION = "4.0.0.1 Media playback, TTS, and threaded streaming STT with enhanced inter-thread communication"
MEDIA_PLAYER_ENTITY = "media_player.soundhive_media_player"
COOLDOWN_PERIOD = 2           # seconds cooldown after TTS finishes
STT_QUEUE_MAXSIZE = 50        # Maximum size for the STT priority queue
LEARN_BUFFER_TIMEOUT = 10     # seconds for buffering learn commands

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger("SoundhiveClient")

# --- Global Variables (set via config) ---
config = None
HA_BASE_URL = None
TOKEN = None
TTS_ENGINE = None
STT_URI = None
STT_FORMAT = None
VOLUME_SETTING = None
RMS_THRESHOLD = None
WAKE_KEYWORD = None
SLEEP_KEYWORD = None
ALARM_KEYWORD = None
CLEAR_ALARM_KEYWORD = None
LEARN_COMMAND = None
FORGET_COMMAND = None
COMMIT_CODE = None
ACTIVE_TIMEOUT = None
LLM_URI = None
WS_API_URL = None
TTS_API_URL = None
CHROMADB_URL = None

# --- Other Global Variables ---
stt_priority_queue = None     # Will be set in main()
tts_finished_time = 0           # Timestamp when TTS playback finished
last_tts_message = ""           # Last TTS message (for filtering)
last_llm_command = ""           # Last LLM command processed
client_mode = "passive"         # Global client mode
last_active_time = None
last_media_url = None
pending_learn = None            # Buffer for learn command content
pending_learn_timestamp = None  # Timestamp for pending learn update
pending_learn_items = []        # List of learned facts pending commit
STOP_KEYWORD = None             # Keyword to stop media playback

# === Configuration Loading ===
def load_config():
    try:
        with open(CONFIG_FILE, "r") as f:
            cfg = json.load(f)
        _LOGGER.info("Configuration loaded successfully.")
        return cfg
    except Exception as e:
        _LOGGER.error("Error loading configuration file: %s", e)
        sys.exit(1)

def reload_config():
    global config, HA_BASE_URL, TOKEN, TTS_ENGINE, STT_URI, STT_FORMAT, VOLUME_SETTING, RMS_THRESHOLD
    global WAKE_KEYWORD, SLEEP_KEYWORD, ALARM_KEYWORD, CLEAR_ALARM_KEYWORD
    global LEARN_COMMAND, FORGET_COMMAND, COMMIT_CODE, ACTIVE_TIMEOUT, LLM_URI, WS_API_URL, TTS_API_URL, CHROMADB_URL
    global STOP_KEYWORD

    config = load_config()
    HA_BASE_URL = config.get("ha_url")
    TOKEN = config.get("auth_token")
    TTS_ENGINE = config.get("tts_engine", "tts.google_translate_en_com")
    STT_URI = config.get("stt_uri") or "http://localhost:10900/inference"
    STT_FORMAT = config.get("stt_format", "wav")
    VOLUME_SETTING = config.get("volume", 0.2)
    RMS_THRESHOLD = float(config.get("rms_threshold", "0.008"))
    WAKE_KEYWORD = config.get("wake_keyword", "hey assistant")
    SLEEP_KEYWORD = config.get("sleep_keyword", "goodbye")
    ALARM_KEYWORD = config.get("alarm_keyword", "alarm now")
    CLEAR_ALARM_KEYWORD = config.get("clear_alarm_keyword", "clear alarm")
    LEARN_COMMAND = config.get("learn_command", "learn this:")
    FORGET_COMMAND = config.get("forget_command", "forget this:")
    COMMIT_CODE = config.get("commit_code", "1234")
    ACTIVE_TIMEOUT = config.get("active_timeout", 5)
    LLM_URI = config.get("llm_uri") or "http://localhost:11434/api/generate"
    CHROMADB_URL = config.get("chromadb_url")
    WS_API_URL = f"{HA_BASE_URL}/api/websocket"
    TTS_API_URL = f"{HA_BASE_URL}/api/tts_get_url"
    STOP_KEYWORD = config.get("stop_keyword", "assistant stop")

    _LOGGER.info(
        "Configuration reloaded: TTS_ENGINE=%s, TOKEN=%s, VOLUME_SETTING=%.2f, RMS_THRESHOLD=%.3f, WAKE_KEYWORD=%s, "
        "SLEEP_KEYWORD=%s, HA_BASE_URL=%s, STT_URI=%s, LLM_URI=%s, CHROMADB_URL=%s, STOP_KEYWORD=%s",
        TTS_ENGINE, TOKEN, VOLUME_SETTING, RMS_THRESHOLD, WAKE_KEYWORD,
        SLEEP_KEYWORD, HA_BASE_URL, STT_URI, LLM_URI, CHROMADB_URL, STOP_KEYWORD
    )

# Initial load and ensure unique_id is present
reload_config()
if "unique_id" not in config:
    config["unique_id"] = str(uuid.uuid4())
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)
time.sleep(2)

# Report configuration attributes to Home Assistant
def report_config_to_ha(config):
    # Use the entity_id provided by Home Assistant if available, else fallback to legacy
    entity_id = config.get("entity_id") or f"media_player.{config.get('unique_id', 'soundhive_unknown')}"
    ha_url = config.get("ha_url", "http://localhost:8123")
    token = config.get("auth_token")

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    state_data = {
        "state": "idle",
        "attributes": {
            "command": "update_config",
            **{k: v for k, v in config.items() if k not in ["auth_token", "ha_url", "unique_id", "name"]},
            "unique_id": config.get("unique_id"),
            "entity_id": entity_id  # Include entity_id in case HA wants to persist it
        }
    }

    try:
        url = f"{ha_url}/api/states/{entity_id}"
        response = requests.post(url, headers=headers, data=json.dumps(state_data))
        response.raise_for_status()
        _LOGGER.info("✅ Reported config to HA successfully as %s", entity_id)
    except Exception as e:
        _LOGGER.error(f"❌ Failed to report config to HA for {entity_id}: {e}")

# Only report if entity_id or unique_id is known
if config.get("entity_id") or config.get("unique_id"):
    report_config_to_ha(config)
else:
    _LOGGER.warning("⚠️ Skipping report_config_to_ha(): no entity_id or unique_id defined.")


# === Helper Functions ===
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

# === VLC Media Playback Setup ===
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

# === REST API Functions ===
async def update_media_state(session, state, media_url=None, volume=None):
    _LOGGER.info("State update skipped (managed by HA integration): %s", state)
    await asyncio.sleep(1)

async def resolve_tts_url(session, media_content_id):
    global TTS_ENGINE
    reload_config()  # Refresh config if changed
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

# === Media Playback Functions ===
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

# === Audio Streaming Functions ===
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

# === STT Processor and Thread Function ===
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
                # Check for stop keyword
                if keyword_in_text(norm_text, STOP_KEYWORD):
                    _LOGGER.info("🛑 Stop keyword '%s' detected — stopping media and switching to passive mode.", STOP_KEYWORD)
                    await stop_media()
                    client_mode = "passive"
                    continue
                if current_player is not None and current_player.get_state() == vlc.State.Playing:
                    continue
                if not norm_text or norm_text in ["blankaudio", "silence", "indistinct chatter", "inaudible"]:
                    _LOGGER.debug("Skipping blank transcription: '%s'", norm_text)
                    continue

                # Handle buffering for learn commands
                if pending_learn is not None:
                    if not (norm_text.startswith(normalize_text(LEARN_COMMAND).replace(":", "")) or
                            norm_text.startswith(normalize_text(FORGET_COMMAND).replace(":", "")) or
                            norm_text.startswith("commit changes:") or
                            keyword_in_text(norm_text, SLEEP_KEYWORD) or
                            keyword_in_text(norm_text, WAKE_KEYWORD)):
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

                # Switch client modes based on wake keyword
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

                # Process commands: retrieve pending, learn, forget, commit, sleep
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
                        last_media_url = resolved_url
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
    return

def stt_thread_func(main_loop, stt_priority_queue):
    asyncio.run(stt_processor())

# === Updated LLM Command Processor with RAG Logic ===
async def process_llm_command(text):
    if not LLM_URI:
        _LOGGER.error("LLM endpoint not defined in configuration.")
        return None
    context = await query_db_for_context(text)
    combined_prompt = f"{context}\nUser query: {text}"
    try:
        async with aiohttp.ClientSession() as session:
            payload = {"model": "llama3.1:8b", "prompt": combined_prompt}
            _LOGGER.info("Sending command to LLM: %s", text)
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

# --- New RAG Helper Function ---
async def query_db_for_context(query_text):
    """Query the Chromadb server for context to augment LLM prompts.
       This function now combines results from the DB with any pending learn items."""
    global CHROMADB_URL
    if not CHROMADB_URL:
        _LOGGER.error("chromadb_url is not set in configuration.")
        return ""
    payload = {"fact": query_text}
    db_context = ""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(f"{CHROMADB_URL}/retrieve", json=payload,
                                    headers={"Authorization": f"Bearer {TOKEN}"}) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    db_context = " ".join(result.get("facts", []))
                    _LOGGER.info("DB context retrieved successfully.")
                else:
                    _LOGGER.error("DB query failed with status: %s", resp.status)
        except Exception as e:
            _LOGGER.error("Exception during DB query: %s", e)
    pending_context = " ".join(pending_learn_items) if pending_learn_items else ""
    combined = db_context + " " + pending_context
    return combined.strip()

# === Command Handlers ===
async def handle_learn_command(content):
    if not content.strip():
        _LOGGER.error("Empty fact provided; skipping learn command.")
        return
    payload = {"fact": content}
    chromadb_url = CHROMADB_URL
    _LOGGER.info("Sending learn command with content: %s", content)
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(f"{chromadb_url}/append_log", json=payload,
                                      headers={"Authorization": f"Bearer {TOKEN}"}) as resp:
                if resp.status == 200:
                    _LOGGER.info("Learn command processed successfully.")
                else:
                    _LOGGER.error("Learn command failed with status: %s", resp.status)
        except Exception as e:
            _LOGGER.error("Exception during learn command: %s", e)

async def handle_forget_command(content):
    payload = {"action": "forget", "content": content}
    chromadb_url = CHROMADB_URL
    _LOGGER.info("Sending forget command for content: %s", content)
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(f"{chromadb_url}/forget", json=payload,
                                      headers={"Authorization": f"Bearer {TOKEN}"}) as resp:
                if resp.status == 200:
                    _LOGGER.info("Forget command processed successfully.")
                else:
                    _LOGGER.error("Forget command failed with status: %s", resp.status)
        except Exception as e:
            _LOGGER.error("Exception during forget command: %s", e)

async def handle_commit_command():
    payload = {"action": "commit"}
    chromadb_url = CHROMADB_URL
    _LOGGER.info("Sending commit command to DB.")
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(f"{chromadb_url}/commit", json=payload,
                                      headers={"Authorization": f"Bearer {TOKEN}"}) as resp:
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

# === Event Handling ===
async def process_state_changed_event(event_data):
    _LOGGER.info("Processing state_changed event: %s", event_data)
    event = event_data.get("event", {})
    if event.get("event_type") != "state_changed":
        return
    new_state = event.get("data", {}).get("new_state", {})
    attributes = new_state.get("attributes", {})
    command = attributes.get("command")
    if command and "update_config" in command:
        args = attributes
        try:
            with open(CONFIG_FILE, "r") as f:
                config_data = json.load(f)
            valid_keys = [
                "tts_engine",
                "active_timeout",
                "rms_threshold",
                "auth_token",
                "wake_keyword",
                "stt_uri",
                "volume",
                "llm_uri",
                "alarm_keyword",
                "clear_alarm_keyword",
                "entity_id",
                "sleep_keyword"
            ]
            updated_fields = []
            for key in valid_keys:
                if key in args:
                    config_data[key] = args[key]
                    updated_fields.append(key)
            if not updated_fields:
                _LOGGER.warning("No valid configuration fields found in update_config command.")
                return
            with open(CONFIG_FILE, "w") as f:
                json.dump(config_data, f, indent=4)
            _LOGGER.info("Configuration file updated successfully with new fields: %s", updated_fields)
            _LOGGER.info("🔄 Reloading config after update_config")
            reload_config()
        except Exception as e:
            _LOGGER.error("Failed to update configuration file: %s", e)
        return
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
        _LOGGER.info("Detected Home Assistant restart; waiting for HA to be ready.")
        await asyncio.sleep(10)
        return
    if domain != "media_player":
        _LOGGER.info("Ignoring call_service event from domain: %s", domain)
        return
    service_payload = service_data.get("service_data", {})
    media_content_id = service_payload.get("media_content_id")
    async with aiohttp.ClientSession() as session:
        if service in ["play_media", "media_play"]:
            if not media_content_id:
                _LOGGER.info("No media_content_id provided. Assuming resume request.")
                if current_player and current_player.get_state() == vlc.State.Paused:
                    await resume_media()
                else:
                    _LOGGER.warning("Resume requested, but player not in paused state.")
                return
            if not isinstance(media_content_id, str) or not media_content_id.strip():
                _LOGGER.warning("Invalid media_content_id received: %r", media_content_id)
                return
            is_tts = "media-source://tts/" in media_content_id or "tts_proxy" in media_content_id
            resolved_url = await resolve_tts_url(session, media_content_id) if is_tts else media_content_id
            if is_tts:
                global last_tts_message
                query = media_content_id.split("?", 1)[1] if "?" in media_content_id else ""
                params = parse_qs(query)
                last_tts_message = params.get("message", [""])[0]
                _LOGGER.info("TTS playback detected; STT processing will resume when media finishes.")
                await clear_stt_queue(stt_priority_queue)
            if resolved_url:
                await play_media(resolved_url)
            else:
                _LOGGER.warning("⚠️ Skipping playback due to unresolved media URL.")
        elif service == "media_pause":
            await pause_media()
        elif service == "media_resume":
            await resume_media()
        elif service == "media_stop":
            await stop_media()
        elif service == "volume_set":
            volume_level = service_payload.get("volume_level", 0.5)
            await set_volume(volume_level)
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
            _LOGGER.info("Attempting to connect to HA WebSocket...")
            async with aiohttp.ClientSession() as session:
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

# === Continuous STT Loop ===
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
                    _LOGGER.info("Inactivity timeout (%.0f sec) reached (elapsed %.2f sec). Switching to PASSIVE mode.", ACTIVE_TIMEOUT, elapsed)
            await asyncio.sleep(0.1)
            continue
        transcription_normalized = normalize_text(transcription).strip()
        current_time = time.monotonic()
        _LOGGER.debug("Received transcription: %r", transcription_normalized)
        # Check for stop keyword in continuous loop
        if keyword_in_text(transcription_normalized, STOP_KEYWORD):
            _LOGGER.info("🛑 Stop keyword '%s' detected — stopping media and switching to passive mode.", STOP_KEYWORD)
            await stop_media()
            client_mode = "passive"
            continue
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
    return


# === STT Thread Function ===
def stt_thread_func(main_loop, stt_priority_queue):
    asyncio.run(stt_processor())

# === Main Execution and Shutdown Handling ===
async def listen_for_events():
    while True:
        try:
            _LOGGER.info("Attempting to connect to HA WebSocket...")
            async with aiohttp.ClientSession() as session:
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

if __name__ == "__main__":
    asyncio.run(main())

