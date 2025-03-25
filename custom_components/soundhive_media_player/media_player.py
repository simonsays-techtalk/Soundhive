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
    rms_threshold = config.get("rms_threshold", 0.008)
    entity = SoundhiveMediaPlayer(name, unique_id, ha_url, tts_engine, token, rms_threshold)
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
    def __init__(self, name, unique_id, ha_url, tts_engine, token, rms_threshold):
        self._name = name
        self._unique_id = unique_id
        self._ha_url = ha_url
        self._tts_engine = tts_engine
        self._token = token
        self._rms_threshold = rms_threshold
        self._state = STATE_IDLE
        self._volume_level = 0.2
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

    async def async_update_config(self, new_tts_engine):
        """Update the configuration with a new TTS engine."""
        self._tts_engine = new_tts_engine
        _LOGGER.info("Updated TTS engine to: %s", new_tts_engine)
