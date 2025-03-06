# VERSION = "1.1.05" 
import logging
import aiohttp
import voluptuous as vol
import homeassistant.helpers.config_validation as cv
from homeassistant.components.media_player import MediaPlayerEntity, MediaPlayerEntityFeature
from homeassistant.const import STATE_IDLE, CONF_TOKEN
from homeassistant.helpers.entity_platform import async_get_current_platform

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
    unique_id = config.get("unique_id", "soundhive_media_player")
    ha_url = config.get("ha_url", "http://localhost:8123")
    tts_engine = config.get("tts_engine", "tts.google_translate_en_com")
    token = config.get(CONF_TOKEN)
    entity = SoundhiveMediaPlayer(name, unique_id, ha_url, tts_engine, token)
    async_add_entities([entity])
    # Store the entity reference keyed by entry.entry_id + "_entity"
    hass.data.setdefault("soundhive_media_player", {})[entry.entry_id + "_entity"] = entity

    # Register the custom service "update_config" on this platform.
    platform = async_get_current_platform()
    platform.async_register_entity_service(
         "update_config",
         {vol.Required("tts_engine"): cv.string},
         "async_update_config",
    )

class SoundhiveMediaPlayer(MediaPlayerEntity):
    def __init__(self, name, unique_id, ha_url, tts_engine, token):
        self._name = name
        self._unique_id = unique_id
        self._ha_url = ha_url
        self._tts_engine = tts_engine
        self._token = token
        self._state = STATE_IDLE
        self._volume_level = 0.2
        self._media_title = None
        self._attr_supported_features = SUPPORT_SOUNHIVE

    def _headers(self):
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json"
        }

    @property
    def unique_id(self):
        return self._unique_id

    @property
    def name(self):
        return self._name

    @property
    def state(self):
        return self._state

    @property
    def volume_level(self):
        return self._volume_level

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
        _LOGGER.debug("async_play_media called with media_type: %s, media_id: %s", media_type, media_id)
        if media_type in ["music", "audio/mp3", "audio/wav"]:
            await self._send_command("play", {"filepath": media_id})
        elif media_type == "tts":
            tts_url = f"{self._ha_url}/api/tts_proxy/{self._tts_engine}?text={media_id}"
            await self._send_command("tts", {"tts_url": tts_url})
        elif media_type == "url":
            await self._send_command("stream", {"stream_url": media_id})
        else:
            _LOGGER.warning("Unsupported media_type: %s", media_type)

    async def async_update(self):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self._ha_url}/api/states/media_player.{self._unique_id}",
                    headers=self._headers()
                ) as response:
                    if response.status == 200:
                        state_info = await response.json()
                        self._state = state_info.get("state", STATE_IDLE)
                        attributes = state_info.get("attributes", {})
                        self._media_title = attributes.get("now_playing")
                        self._volume_level = attributes.get("volume", 20) / 100
        except Exception as e:
            _LOGGER.error("Failed to update state: %s", e)

    async def _send_command(self, command, args=None):
        # Build a full payload with required attributes.
        base_attributes = {
            "friendly_name": self._name,
            "supported_features": 52735,
            "media_content_id": "",
            "media_content_type": None,
            "volume_level": self._volume_level,
            "media_library": [
                {"title": "Song 1", "media_url": "http://example.com/song1.mp3"},
                {"title": "Song 2", "media_url": "http://example.com/song2.mp3"}
            ]
        }
        # Add the custom command and any extra arguments.
        base_attributes["command"] = command
        if args:
            base_attributes.update(args)
        payload = {"state": self._state, "attributes": base_attributes}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._ha_url}/api/states/media_player.{self._unique_id}",
                    headers=self._headers(),
                    json=payload
                ) as response:
                    if response.status == 200:
                        _LOGGER.debug("Command sent successfully: %s with args %s", command, args)
                    else:
                        _LOGGER.error("Failed to send command %s, status code: %s", command, response.status)
        except Exception as e:
            _LOGGER.error("Exception when sending command %s: %s", command, e)

    async def async_update_config(self, tts_engine):
        _LOGGER.info("Updating client config with new TTS engine: %s", tts_engine)
        self._tts_engine = tts_engine
        await self._send_command("update_config", {"tts_engine": tts_engine})
