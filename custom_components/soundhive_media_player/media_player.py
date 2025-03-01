# soundhive_media_player/media_player.py
# Soundhive Custom Component for Home Assistant: Version 2.1.0 (REST API Integration with TTS & Streaming Fixes)
# - Direct integration with Soundhive Client via REST API
# - Supports TTS and media streaming
# - Play/Pause/Stop/Resume/Volume control
# - Handles multi-instance support with unique IDs
# - Ensures full compatibility with HA play_media service for TTS & streaming

import logging
from homeassistant.components.media_player import MediaPlayerEntity
from homeassistant.components.media_player.const import (
    SUPPORT_PLAY, SUPPORT_PAUSE, SUPPORT_STOP, SUPPORT_VOLUME_SET, SUPPORT_VOLUME_STEP,
    SUPPORT_TURN_ON, SUPPORT_TURN_OFF, SUPPORT_PLAY_MEDIA
)
from homeassistant.const import STATE_IDLE, STATE_PLAYING, STATE_PAUSED
import aiohttp

_LOGGER = logging.getLogger("custom_components.soundhive_media_player")

SUPPORT_SOUNHIVE = (
    SUPPORT_PLAY
    | SUPPORT_PAUSE
    | SUPPORT_STOP
    | SUPPORT_VOLUME_SET
    | SUPPORT_VOLUME_STEP
    | SUPPORT_TURN_ON
    | SUPPORT_TURN_OFF
    | SUPPORT_PLAY_MEDIA  # Added to support HA play_media service
)

HA_BASE_URL = "http://homeassistanttest.local:8123"
HA_TOKEN = ""  # Replace with actual token or retrieve securely
HEADERS = {
    "Authorization": f"Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJkN2I3M2Q0NGIxZmE0ZWY1YmYxZTczMWM1YThjNGU0NSIsImlhdCI6MTc0MDU2OTEzMSwiZXhwIjoyMDU1OTI5MTMxfQ.Er-E-RT92XtVvC3cuRrhRZgtWvNw8AjKt98cVV-K0b4",
    "Content-Type": "application/json"
}

async def async_setup_platform(hass, config, async_add_entities, discovery_info=None):
    """Set up the Soundhive media player platform."""
    name = config.get("name", "Soundhive Media Player")
    unique_id = config.get("unique_id", "Soundhive_mediaplayer")
    async_add_entities([SoundhiveMediaPlayer(name, unique_id)], True)

class SoundhiveMediaPlayer(MediaPlayerEntity):
    """Representation of the Soundhive Media Player."""

    def __init__(self, name, unique_id):
        self._name = name
        self._unique_id = unique_id
        self._state = STATE_IDLE
        self._volume_level = 0.5
        self._media_title = None
        self._attr_supported_features = SUPPORT_SOUNHIVE

    @property
    def name(self):
        return self._name

    @property
    def state(self):
        return self._state

    @property
    def volume_level(self):
        return self._volume_level

    @property
    def media_title(self):
        return self._media_title

    async def async_set_volume_level(self, volume):
        self._volume_level = volume
        await self._send_command("volume", {"volume_level": int(volume * 100)})

    async def async_media_play(self):
        await self._send_command("resume")

    async def async_media_pause(self):
        await self._send_command("pause")

    async def async_media_stop(self):
        await self._send_command("stop")

    async def async_play_media(self, media_type, media_id, **kwargs):
        """Handle play media command with media ID (TTS or stream)."""
        _LOGGER.debug(f"📡 async_play_media called with media_type: {media_type}, media_id: {media_id}")
        if media_type in ["music", "audio/mp3", "audio/wav"]:
            await self._send_command("play", {"filepath": media_id})
        elif media_type == "tts":
            # Ensure TTS uses the correct URL format
            await self._send_command("tts", {"tts_url": media_id})
        elif media_type == "url":
            # For streaming services
            await self._send_command("stream", {"stream_url": media_id})
        else:
            _LOGGER.warning(f"⚠️ Unsupported media_type: {media_type}")

    async def async_update(self):
        """Fetch the latest state from the client."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{HA_BASE_URL}/api/states/media_player.{self._unique_id}",
                                       headers=HEADERS) as response:
                    if response.status == 200:
                        state_info = await response.json()
                        self._state = state_info.get("state", STATE_IDLE)
                        attributes = state_info.get("attributes", {})
                        self._media_title = attributes.get("now_playing")
                        self._volume_level = attributes.get("volume", 50) / 100
        except Exception as e:
            _LOGGER.error(f"❌ Failed to update state: {e}")

    async def _send_command(self, command, args=None):
        """Send a control command to the Soundhive client."""
        payload = {"attributes": {"command": command}}
        if args:
            payload["attributes"].update(args)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{HA_BASE_URL}/api/states/media_player.{self._unique_id}",
                                        headers=HEADERS, json=payload) as response:
                    if response.status == 200:
                        _LOGGER.debug(f"📡 Command sent successfully: {command} with args {args}")
                    else:
                        _LOGGER.error(f"❌ Failed to send command {command}, status code: {response.status}")
        except Exception as e:
            _LOGGER.error(f"❌ Exception when sending command {command}: {e}")

# ✅ Version 2.1.0:
# - Added SUPPORT_PLAY_MEDIA for full compatibility with HA TTS & media streaming
# - Enhanced logging for play_media calls
# - Switched to aiohttp for async HTTP requests
# - Improved media type handling for broader HA compatibility
