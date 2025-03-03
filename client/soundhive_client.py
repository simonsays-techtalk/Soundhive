#!/usr/bin/env python3
import asyncio
import aiohttp
import json
import logging
import os
import sys
import vlc

CONFIG_FILE = "soundhive_config.json"
VERSION = "1.1.02"
MEDIA_PLAYER_ENTITY = "media_player.soundhive_media_player"

# Set up logging
logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger("SoundhiveClient")

def load_config():
    try:
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        _LOGGER.error("Error loading configuration file: %s", e)
        sys.exit(1)

config = load_config()
HA_BASE_URL = config.get("ha_url")
TOKEN = config.get("auth_token")
TTS_ENGINE = config.get("tts_engine", "tts.google_translate_en_com")  # Default value

WS_API_URL = f"{HA_BASE_URL}/api/websocket"
REGISTER_ENTITY_API = f"{HA_BASE_URL}/api/states/{MEDIA_PLAYER_ENTITY}"
TTS_API_URL = f"{HA_BASE_URL}/api/tts_get_url"

# Create a global VLC instance forcing ALSA output (mic2hat should be set as default via ALSA config)
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

async def register_media_player(session):
    headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
    payload = {
        "state": "idle",
        "attributes": {
            "friendly_name": "Soundhive Media Player",
            "supported_features": 52735,
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
    await update_media_state(session, "idle")

async def update_media_state(session, state, media_url=None, volume=None):
    headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
    payload = {
        "state": state,
        "attributes": {
            "friendly_name": "Soundhive Media Player",
            "supported_features": 52735,
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

async def resolve_tts_url(session, media_content_id):
    # Reload configuration on every call
    config = load_config()
    tts_engine = config.get("tts_engine", "tts.google_translate_en_com")
    if media_content_id.startswith("media-source://tts/"):
        try:
            message_param = media_content_id.split("message=")[1].split("&")[0]
            message = message_param.replace("+", " ")
            async with session.post(
                TTS_API_URL,
                headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
                json={"message": message, "platform": tts_engine}
            ) as resp:
                if resp.status == 200:
                    response_json = await resp.json()
                    return response_json.get("url")
                else:
                    _LOGGER.error("❌ Failed to retrieve TTS URL. Status: %s", resp.status)
        except Exception as e:
            _LOGGER.error("❌ Error resolving TTS URL: %s", str(e))
    return media_content_id

async def play_media(media_url):
    global current_player, media_paused
    _LOGGER.info("▶️ Playing media: %s", media_url)
    if current_player:
        current_player.stop()
    current_player = vlc_instance.media_player_new()
    media = vlc_instance.media_new(media_url)
    current_player.set_media(media)
    current_player.audio_set_volume(80)
    current_player.play()
    media_paused = False
    async with aiohttp.ClientSession() as session:
        await update_media_state(session, "playing", media_url=media_url)

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
    if current_player and not media_paused:
        _LOGGER.info("⏸ Pausing media playback")
        current_player.pause()
        media_paused = True
        async with aiohttp.ClientSession() as session:
            await update_media_state(session, "paused")

async def resume_media():
    global current_player, media_paused
    if current_player and media_paused:
        _LOGGER.info("▶️ Resuming media playback")
        current_player.pause()  # VLC's pause toggles playback.
        media_paused = False
        async with aiohttp.ClientSession() as session:
            await update_media_state(session, "playing")

async def set_volume(level):
    global current_player
    _LOGGER.info("🔊 Setting volume to: %s", level)
    if current_player:
        current_player.audio_set_volume(int(level * 100))
    async with aiohttp.ClientSession() as session:
        state = "playing" if not media_paused else "paused"
        await update_media_state(session, state, volume=level)

async def process_event(event_data):
    service_data = event_data.get("event", {}).get("data", {})
    domain = service_data.get("domain")
    service = service_data.get("service")
    _LOGGER.info("Received service call: domain=%s, service=%s", domain, service)
    if domain != "media_player":
        _LOGGER.info("Ignoring service call from domain: %s", domain)
        return
    async with aiohttp.ClientSession() as session:
        if service == "play_media":
            media_content_id = service_data.get("service_data", {}).get("media_content_id")
            resolved_url = await resolve_tts_url(session, media_content_id)
            await play_media(resolved_url)
        elif service == "media_play":
            if current_player and media_paused:
                await resume_media()
            else:
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
        else:
            _LOGGER.info("Unhandled service: %s", service)

async def listen_for_media_commands():
    async with aiohttp.ClientSession() as session:
        await register_media_player(session)
        async with session.ws_connect(WS_API_URL) as ws:
            await ws.send_json({"type": "auth", "access_token": TOKEN})
            await ws.receive_json()  # Auth response
            await ws.send_json({"id": 1, "type": "subscribe_events", "event_type": "call_service"})
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    await process_event(data)
                else:
                    _LOGGER.debug("Received non-text message: %s", msg)

async def main():
    _LOGGER.info("🚀 Soundhive Client v%s - TTS & Media (using VLC with ALSA)", VERSION)
    await listen_for_media_commands()

if __name__ == "__main__":
    asyncio.run(main())
