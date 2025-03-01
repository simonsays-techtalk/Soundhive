import asyncio
import aiohttp
import json
import logging
import os
import shutil
import signal
import subprocess
from datetime import datetime

# Soundhive Client Version
VERSION = "1.0.04"

HA_BASE_URL = "http://192.168.188.62:8123"
MEDIA_PLAYER_ENTITY = "media_player.soundhive_media_player"
WS_API_URL = f"{HA_BASE_URL}/api/websocket"
REGISTER_ENTITY_API = f"{HA_BASE_URL}/api/states/{MEDIA_PLAYER_ENTITY}"
TOKEN_CACHE_FILE = ".soundhive_token_cache"
TTS_API_URL = f"{HA_BASE_URL}/api/tts_get_url"

FFPLAY_COMMAND = shutil.which("ffplay")
if not FFPLAY_COMMAND:
    logging.error("❌ 'ffplay' not found. Install it with 'sudo apt install ffmpeg'")

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger("SoundhiveClient")

current_process = None
media_paused = False

# Define supported features with media browsing flag.
# (SUPPORT_BROWSE_MEDIA = 16384; OR it with your other supported flags.)
SUPPORTED_FEATURES = 52735 | 16384

def get_volume_control_name():
    try:
        # Query controls on card 1 (USB headset)
        output = subprocess.check_output("amixer -c 1 scontrols", shell=True).decode()
        for line in output.split("\n"):
            if "Headset" in line:
                # Return "Headset" if it's present.
                return "Headset"
    except Exception as e:
        _LOGGER.error("🔇 Failed to detect volume control on card 1: %s", str(e))
    return "Headset"

VOLUME_CONTROL = get_volume_control_name()
_LOGGER.info("🔊 Detected volume control: %s", VOLUME_CONTROL)

# Dummy media library for browsing; replace with your own media items as needed.
MEDIA_LIBRARY = [
    {"title": "Song 1", "media_url": "http://example.com/song1.mp3"},
    {"title": "Song 2", "media_url": "http://example.com/song2.mp3"}
]

async def retrieve_token():
    if os.path.exists(TOKEN_CACHE_FILE):
        with open(TOKEN_CACHE_FILE, "r") as f:
            cached_token = f.read().strip()
            if cached_token:
                _LOGGER.info("✅ Retrieved token from cache.")
                return cached_token
    return None

async def register_media_player(session, token):
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "state": "idle",
        "attributes": {
            "friendly_name": "Soundhive Media Player",
            "supported_features": SUPPORTED_FEATURES,
            "media_content_id": None,
            "media_content_type": None,
            "volume_level": 0.5,
            "media_library": MEDIA_LIBRARY
        }
    }
    async with session.post(REGISTER_ENTITY_API, headers=headers, json=payload) as resp:
        if resp.status in [200, 201]:
            _LOGGER.info("✅ Media player entity registered successfully.")
        else:
            _LOGGER.error("❌ Failed to register media player entity. Status: %s", resp.status)
    await update_media_state(session, token, "idle")

async def update_media_state(session, token, state, media_url=None, volume=None):
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "state": state,
        "attributes": {
            "friendly_name": "Soundhive Media Player",
            "supported_features": SUPPORTED_FEATURES,
            "media_content_id": media_url if media_url is not None else "",
            "media_content_type": "music" if media_url else None,
            "volume_level": volume if volume is not None else 0.5,
            "media_library": MEDIA_LIBRARY
        }
    }
    async with session.post(REGISTER_ENTITY_API, headers=headers, json=payload) as resp:
        if resp.status in [200, 201]:
            _LOGGER.info("✅ Media player state updated: %s", state)
        else:
            _LOGGER.error("❌ Failed to update media player state. Status: %s", resp.status)
    await asyncio.sleep(1)

async def resolve_tts_url(session, token, media_content_id):
    if media_content_id.startswith("media-source://tts/"):
        try:
            message_param = media_content_id.split("message=")[1].split("&")[0]
            message = message_param.replace("+", " ")
            async with session.post(
                TTS_API_URL,
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json={"message": message, "platform": "tts.piper_2"}
            ) as resp:
                if resp.status == 200:
                    response_json = await resp.json()
                    return response_json.get("url")
                else:
                    _LOGGER.error("❌ Failed to retrieve TTS URL. Status: %s", resp.status)
        except Exception as e:
            _LOGGER.error("❌ Error resolving TTS URL: %s", str(e))
    return media_content_id

async def play_media(media_url, token):
    global current_process, media_paused
    _LOGGER.info("▶️ Playing media: %s", media_url)
    # Launch ffplay; note that ffplay might not support pause/resume natively.
    current_process = subprocess.Popen([FFPLAY_COMMAND, "-nodisp", "-autoexit", "-volume", "80", media_url])
    media_paused = False
    async with aiohttp.ClientSession() as session:
        await update_media_state(session, token, "playing", media_url=media_url)

async def stop_media(token):
    global current_process, media_paused
    if current_process:
        _LOGGER.info("⏹ Stopping media playback")
        current_process.terminate()
        current_process = None
        media_paused = False
    async with aiohttp.ClientSession() as session:
        await update_media_state(session, token, "idle")

async def pause_media(token):
    global current_process, media_paused
    if current_process:
        _LOGGER.info("⏸ Pausing media playback")
        try:
            os.kill(current_process.pid, signal.SIGSTOP)
            media_paused = True
            async with aiohttp.ClientSession() as session:
                await update_media_state(session, token, "paused")
        except Exception as e:
            _LOGGER.error("❌ Error pausing media: %s", str(e))

async def resume_media(token):
    global current_process, media_paused
    if current_process and media_paused:
        _LOGGER.info("▶️ Resuming media playback")
        try:
            os.kill(current_process.pid, signal.SIGCONT)
            media_paused = False
            async with aiohttp.ClientSession() as session:
                await update_media_state(session, token, "playing")
        except Exception as e:
            _LOGGER.error("❌ Error resuming media: %s", str(e))

async def set_volume(level, token):
    _LOGGER.info("🔊 Setting volume to: %s", level)
    # Use subprocess.run to call amixer on card 1 with the "Headset" control.
    cmd = ["amixer", "-c", "1", "set", VOLUME_CONTROL, f"{int(level * 100)}%"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        _LOGGER.error("Failed to set volume: %s", result.stderr.strip())
    async with aiohttp.ClientSession() as session:
        state = "playing" if not media_paused else "paused"
        await update_media_state(session, token, state, volume=level)

async def process_event(event_data, token):
    service_data = event_data.get("event", {}).get("data", {})
    domain = service_data.get("domain")
    service = service_data.get("service")
    _LOGGER.info("Received service call: domain=%s, service=%s", domain, service)
    # Only process events for the media_player domain.
    if domain != "media_player":
        _LOGGER.info("Ignoring service call from domain: %s", domain)
        return
    async with aiohttp.ClientSession() as session:
        if service == "play_media":
            media_content_id = service_data.get("service_data", {}).get("media_content_id")
            resolved_url = await resolve_tts_url(session, token, media_content_id)
            await play_media(resolved_url, token)
        elif service == "media_play":
            if current_process and media_paused:
                await resume_media(token)
            else:
                media_content_id = service_data.get("service_data", {}).get("media_content_id")
                resolved_url = await resolve_tts_url(session, token, media_content_id)
                await play_media(resolved_url, token)
        elif service == "media_stop":
            await stop_media(token)
        elif service == "volume_set":
            volume_level = service_data.get("service_data", {}).get("volume_level", 0.5)
            await set_volume(volume_level, token)
        elif service == "media_pause":
            await pause_media(token)
        elif service == "browse_media":
            _LOGGER.info("Browse media service requested, returning media library")
            # In this simple implementation, we update the state so that the media_library attribute is exposed.
            await update_media_state(session, token, "idle")
        else:
            _LOGGER.info("Unhandled service: %s", service)

async def listen_for_media_commands(token):
    async with aiohttp.ClientSession() as session:
        await register_media_player(session, token)
        async with session.ws_connect(WS_API_URL) as ws:
            await ws.send_json({"type": "auth", "access_token": token})
            await ws.receive_json()
            await ws.send_json({"id": 1, "type": "subscribe_events", "event_type": "call_service"})
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    await process_event(data, token)
                else:
                    _LOGGER.debug("Received non-text message: %s", msg)

async def main():
    _LOGGER.info("🚀 Soundhive Client v%s - TTS & Media", VERSION)
    token = await retrieve_token()
    if token:
        await listen_for_media_commands(token)
    else:
        _LOGGER.error("❌ No valid token found. Exiting.")

if __name__ == "__main__":
    asyncio.run(main())

