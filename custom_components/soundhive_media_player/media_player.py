import logging
from homeassistant.components.media_player import (
    MediaPlayerEntity,
    MediaPlayerEntityFeature,
)
from homeassistant.components.media_player.const import MediaPlayerState
from homeassistant.config_entries import ConfigEntry

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

async def async_setup_entry(hass, config_entry, async_add_entities):
    entity = hass.data[DOMAIN]["entity"]
    async_add_entities([entity], update_before_add=True)

class SoundhiveMediaPlayer(MediaPlayerEntity):
    def __init__(self, config_entry: ConfigEntry):
        self._entry = config_entry
        self._state = MediaPlayerState.IDLE
        self._volume = 0.4
        self._attr_name = "Soundhive"
        self._attr_unique_id = "soundhive_default"
        self._load_config()

    def _load_config(self):
        data = self._entry.data
        options = self._entry.options

        def opt(key, default=None):
            return options.get(key, data.get(key, default))

        self._attr_name = data.get("name", "Soundhive")
        self._attr_unique_id = data.get("unique_id", "soundhive_default")
        self._volume = opt("volume", 0.4)
        self._tts_engine = opt("tts_engine")
        self._rms_threshold = opt("rms_threshold")
        self._wake_keyword = opt("wake_keyword")
        self._sleep_keyword = opt("sleep_keyword")
        self._stop_keyword = opt("stop_keyword")
        self._llm_uri = opt("llm_uri")
        self._llm_model = opt("llm_model")
        self._chromadb_url = opt("chromadb_url")
        self._stt_uri = opt("stt_uri")
        self._active_timeout = opt("active_timeout")
        self._alarm_keyword = opt("alarm_keyword")
        self._clear_alarm_keyword = opt("clear_alarm_keyword")
        self._alsa_device = opt("alsa_device")
        self._alsa_devices = opt("alsa_devices", [])
        self._client_ip = data.get("client_ip")

    def config_entry_updated(self, new_entry: ConfigEntry):
        self._entry = new_entry
        self._load_config()
        self.async_write_ha_state()

    @property
    def state(self):
        return self._state

    @property
    def volume_level(self):
        return self._volume

    @property
    def supported_features(self):
        return (
            MediaPlayerEntityFeature.VOLUME_SET
            | MediaPlayerEntityFeature.PLAY
            | MediaPlayerEntityFeature.PAUSE
            | MediaPlayerEntityFeature.STOP
        )

    @property
    def extra_state_attributes(self):
        return {
            "tts_engine": self._tts_engine,
            "rms_threshold": self._rms_threshold,
            "volume": self._volume,
            "wake_keyword": self._wake_keyword,
            "sleep_keyword": self._sleep_keyword,
            "stop_keyword": self._stop_keyword,
            "llm_uri": self._llm_uri,
            "llm_model": self._llm_model,
            "chromadb_url": self._chromadb_url,
            "stt_uri": self._stt_uri,
            "active_timeout": self._active_timeout,
            "alarm_keyword": self._alarm_keyword,
            "clear_alarm_keyword": self._clear_alarm_keyword,
            "alsa_device": self._alsa_device,
            "alsa_devices": self._alsa_devices,
            "client_ip": self._client_ip,
            "friendly_name": self._attr_name,
            "media_state": self._state,
            "entity_id": self.entity_id,
            "command": "update_config"
        }

    def set_state(self, state):
        self._state = state
        self.async_write_ha_state()

    async def async_set_volume_level(self, volume):
        self._volume = volume
        self.async_write_ha_state()

    async def async_media_play(self):
        self.set_state(MediaPlayerState.PLAYING)

    async def async_media_pause(self):
        self.set_state(MediaPlayerState.PAUSED)

    async def async_media_stop(self):
        self.set_state(MediaPlayerState.IDLE)

