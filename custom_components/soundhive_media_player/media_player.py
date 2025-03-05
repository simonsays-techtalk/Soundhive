# Soundhive Custom Component for Home Assistant: Version 1.1.03 (REST API Integration with TTS & Streaming Fixes)
import logging
from homeassistant.components.media_player import MediaPlayerEntity, MediaPlayerEntityFeature
from homeassistant.const import STATE_IDLE, STATE_PLAYING, STATE_PAUSED, CONF_TOKEN
import aiohttp

_LOGGER = logging.getLogger("custom_components.soundhive_media_player")

SUPPORT_SOUNHIVE = (
    MediaPlayerEntityFeature.PLAY |
    MediaPlayerEntityFeature.PAUSE |
    MediaPlayerEntityFeature.STOP |
    MediaPlayerEntityFeature.VOLUME_SET |
    MediaPlayerEntityFeature.VOLUME_STEP |
    MediaPlayerEntityFeature.TURN_ON |
    MediaPlayerEntityFeature.TURN_OFF |
    MediaPlayerEntityFeature.PLAY_MEDIA
)

async def async_setup_entry(hass, entry, async_add_entities):
    config = entry.data
    name = config.get("name", "Soundhive Media Player")
    unique_id = config.get("unique_id", "soundhive_mediaplayer")
    ha_url = config.get("ha_url", "http://localhost:8123")
    tts_engine = config.get("tts_engine", "tts.google_translate_en_com")
    token = config.get(CONF_TOKEN)
    async_add_entities([SoundhiveMediaPlayer(name, unique_id, ha_url, tts_engine, token)])

class SoundhiveMediaPlayer(MediaPlayerEntity):
    def __init__(self, name, unique_id, ha_url, tts_engine, token):
        self._name = name
        self._unique_id = unique_id
        self._ha_url = ha_url  # URL from the config entry
        self._tts_engine = tts_engine
        self._token = token    # Token from the config entry
        self._state = STATE_IDLE
        self._volume_level = 0.2
        self._media_title = None
        self._attr_supported_features = SUPPORT_SOUNHIVE

    def _headers(self):
        """Build request headers with the stored token."""
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json"
        }

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
            # Build a TTS URL that incorporates the selected engine from the config
            tts_url = f"{self._ha_url}/api/tts_proxy/{self._tts_engine}?text={media_id}"
            await self._send_command("tts", {"tts_url": tts_url})
        elif media_type == "url":
            await self._send_command("stream", {"stream_url": media_id})
        else:
            _LOGGER.warning(f"⚠️ Unsupported media_type: {media_type}")

    async def async_update(self):
        """Fetch the latest state from the client."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self._ha_url}/api/states/media_player.{self._unique_id}",
                                         headers=self._headers()) as response:
                    if response.status == 200:
                        state_info = await response.json()
                        self._state = state_info.get("state", STATE_IDLE)
                        attributes = state_info.get("attributes", {})
                        self._media_title = attributes.get("now_playing")
                        self._volume_level = attributes.get("volume", 20) / 100
        except Exception as e:
            _LOGGER.error(f"❌ Failed to update state: {e}")

    async def _send_command(self, command, args=None):
        """Send a control command to the Soundhive client."""
        payload = {"attributes": {"command": command}}
        if args:
            payload["attributes"].update(args)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self._ha_url}/api/states/media_player.{self._unique_id}",
                                        headers=self._headers(), json=payload) as response:
                    if response.status == 200:
                        _LOGGER.debug(f"📡 Command sent successfully: {command} with args {args}")
                    else:
                        _LOGGER.error(f"❌ Failed to send command {command}, status code: {response.status}")
        except Exception as e:
            _LOGGER.error(f"❌ Exception when sending command {command}: {e}")

