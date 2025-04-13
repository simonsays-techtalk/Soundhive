import logging
import socket
import aiohttp
import voluptuous as vol
import homeassistant.helpers.config_validation as cv

from homeassistant.components.media_player import (
    MediaPlayerEntity,
    MediaPlayerEntityFeature,
)
from homeassistant.const import (
    STATE_IDLE,
    STATE_PLAYING,
    STATE_PAUSED,
    CONF_TOKEN,
)
from homeassistant.helpers.entity_platform import async_get_current_platform

_LOGGER = logging.getLogger("custom_components.soundhive_media_player")

SUPPORT_SOUNHIVE = (
    MediaPlayerEntityFeature.PLAY |
    MediaPlayerEntityFeature.PAUSE |
    MediaPlayerEntityFeature.STOP |
    MediaPlayerEntityFeature.VOLUME_SET |
    MediaPlayerEntityFeature.VOLUME_STEP |
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
    stt_uri = config.get("stt_uri")
    llm_uri = config.get("llm_uri")
    wake_keyword = config.get("wake_keyword")
    sleep_keyword = config.get("sleep_keyword")
    active_timeout = config.get("active_timeout", 30)
    volume = config.get("volume", 0.4)
    alarm_keyword = config.get("alarm_keyword", "alarm now")
    clear_alarm_keyword = config.get("clear_alarm_keyword", "clear alarm")
    stop_keyword = config.get("stop_keyword", "stop")
    # For audio devices, accept the dynamic list pushed by the client.
    selected_alsa_device = config.get("alsa_device", None)
    alsa_devices = config.get("alsa_devices", [])
    # New fields:
    chromadb_url = config.get("chromadb_url")
    llm_model = config.get("llm_model")

    entity = SoundhiveMediaPlayer(
        hass,
        name=name,
        unique_id=unique_id,
        ha_url=ha_url,
        tts_engine=tts_engine,
        token=token,
        rms_threshold=rms_threshold,
        stt_uri=stt_uri,
        llm_uri=llm_uri,
        wake_keyword=wake_keyword,
        sleep_keyword=sleep_keyword,
        active_timeout=active_timeout,
        volume=volume,
        alarm_keyword=alarm_keyword,
        clear_alarm_keyword=clear_alarm_keyword,
        stop_keyword=stop_keyword,
        selected_alsa_device=selected_alsa_device,
        alsa_devices=alsa_devices,
        chromadb_url=chromadb_url,
        llm_model=llm_model
    )

    async_add_entities([entity])
    hass.data.setdefault("soundhive_media_player", {})[entry.entry_id + "_entity"] = entity

    platform = async_get_current_platform()
    platform.async_register_entity_service(
        "update_config",
        {
            vol.Required("tts_engine"): cv.string,
            vol.Required("rms_threshold"): vol.Coerce(float),
            vol.Required("stt_uri"): cv.string,
            vol.Required("llm_uri"): cv.string,
            vol.Required("llm_model"): cv.string,
            vol.Required("chromadb_url"): cv.string,
            vol.Required("wake_keyword"): cv.string,
            vol.Required("sleep_keyword"): cv.string,
            vol.Required("active_timeout"): vol.Coerce(int),
            vol.Required("volume"): vol.Coerce(float),
            vol.Required("alarm_keyword"): cv.string,
            vol.Required("clear_alarm_keyword"): cv.string,
            vol.Required("stop_keyword"): cv.string,
            vol.Required("alsa_device"): cv.string,
        },
        "async_update_config",
    )

class SoundhiveMediaPlayer(MediaPlayerEntity):
    def __init__(self, hass, name, unique_id, ha_url, tts_engine, token, rms_threshold,
                 stt_uri=None, llm_uri=None, wake_keyword=None, sleep_keyword=None,
                 active_timeout=30, volume=0.4, alarm_keyword="alarm now", clear_alarm_keyword="clear alarm", stop_keyword="stop",
                 selected_alsa_device=None, alsa_devices=None, chromadb_url=None, llm_model=None):
        super().__init__()
        self.hass = hass
        self._attr_name = name
        self._attr_unique_id = unique_id
        self._ha_url = ha_url
        self._tts_engine = tts_engine
        self._token = token
        self._rms_threshold = rms_threshold
        self._stt_uri = stt_uri
        self._llm_uri = llm_uri
        self._wake_keyword = wake_keyword
        self._sleep_keyword = sleep_keyword
        self._active_timeout = active_timeout
        self._volume_level = volume
        self._alarm_keyword = alarm_keyword
        self._clear_alarm_keyword = clear_alarm_keyword
        self._stop_keyword = stop_keyword
        self._state = STATE_IDLE
        self._media_loaded = False
        self._attr_supported_features = SUPPORT_SOUNHIVE

        # Use the dynamic list of alsa devices from the client.
        self._alsa_devices = alsa_devices if alsa_devices else []
        # Use the selected alsa device if provided; otherwise, default to the first in the list, or fallback to "default".
        if selected_alsa_device:
            self._selected_alsa_device = selected_alsa_device
        elif self._alsa_devices:
            self._selected_alsa_device = self._alsa_devices[0]
        else:
            self._selected_alsa_device = "default"

        self._chromadb_url = chromadb_url if chromadb_url is not None else "http://madison.autohome.local:8001"
        self._llm_model = llm_model if llm_model is not None else "llama3.1:8b"

    @property
    def unique_id(self):
        return self._attr_unique_id

    @property
    def state(self):
        return self._state

    @property
    def volume_level(self):
        return self._volume_level

    @property
    def extra_state_attributes(self):
        return {
            "tts_engine": self._tts_engine,
            "rms_threshold": self._rms_threshold,
            "stt_uri": self._stt_uri,
            "llm_uri": self._llm_uri,
            "llm_model": self._llm_model,
            "chromadb_url": self._chromadb_url,
            "wake_keyword": self._wake_keyword,
            "sleep_keyword": self._sleep_keyword,
            "active_timeout": self._active_timeout,
            "volume": self._volume_level,
            "alarm_keyword": self._alarm_keyword,
            "clear_alarm_keyword": self._clear_alarm_keyword,
            "stop_keyword": self._stop_keyword,
            "client_ip": self._get_local_ip(),
            "alsa_devices": self._alsa_devices,
            "alsa_device": self._selected_alsa_device,
        }

    def _get_local_ip(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    async def async_added_to_hass(self):
        await super().async_added_to_hass()
        state = self.hass.states.get(self.entity_id)
        if state:
            attrs = state.attributes
            self._tts_engine = attrs.get("tts_engine", self._tts_engine)
            self._rms_threshold = attrs.get("rms_threshold", self._rms_threshold)
            self._stt_uri = attrs.get("stt_uri", self._stt_uri)
            self._llm_uri = attrs.get("llm_uri", self._llm_uri)
            self._llm_model = attrs.get("llm_model", self._llm_model)
            self._chromadb_url = attrs.get("chromadb_url", self._chromadb_url)
            self._wake_keyword = attrs.get("wake_keyword", self._wake_keyword)
            self._sleep_keyword = attrs.get("sleep_keyword", self._sleep_keyword)
            self._active_timeout = attrs.get("active_timeout", self._active_timeout)
            self._volume_level = attrs.get("volume", self._volume_level)
            self._alarm_keyword = attrs.get("alarm_keyword", self._alarm_keyword)
            self._clear_alarm_keyword = attrs.get("clear_alarm_keyword", self._clear_alarm_keyword)
            self._stop_keyword = attrs.get("stop_keyword", self._stop_keyword)
            self._selected_alsa_device = attrs.get("alsa_device", self._selected_alsa_device)
            self._alsa_devices = attrs.get("alsa_devices", self._alsa_devices)
        await self.async_update_ha_state(force_refresh=True)

    async def async_media_play(self):
        if self._state == STATE_PAUSED:
            _LOGGER.info("Resuming playback")
            self._state = STATE_PLAYING
            self.hass.states.async_set(
                self.entity_id,
                self._state,
                {"command": "resume_media"}
            )
        else:
            _LOGGER.warning("Play called, but media is not paused")

    async def async_media_pause(self):
        if self._state == STATE_PLAYING:
            _LOGGER.info("Pausing playback")
            self._state = STATE_PAUSED
            self.hass.states.async_set(
                self.entity_id,
                self._state,
                {"command": "pause_media"}
            )
        else:
            _LOGGER.warning("Pause called, but media is not playing")

    async def async_media_stop(self):
        if self._state != STATE_IDLE:
            _LOGGER.info("Stopping playback")
            self._state = STATE_IDLE
            self._media_loaded = False
        else:
            _LOGGER.debug("Stop called, but already idle")

    async def async_set_volume_level(self, volume):
        _LOGGER.info("Volume set to %.2f", volume)
        self._volume_level = volume

    async def async_play_media(self, media_type, media_id, **kwargs):
        _LOGGER.info("Play media request: type=%s, id=%s", media_type, media_id)
        self._media_loaded = True
        self._state = STATE_PLAYING

    async def async_update_config(
        self,
        new_tts_engine,
        new_rms_threshold,
        new_stt_uri,
        new_llm_uri,
        new_wake_keyword,
        new_sleep_keyword,
        new_active_timeout,
        new_volume,
        new_alarm_keyword,
        new_clear_alarm_keyword,
        new_stop_keyword,
        new_alsa_device,
        new_chromadb_url,
        new_llm_model
    ):
        self._tts_engine = new_tts_engine
        self._rms_threshold = new_rms_threshold
        self._stt_uri = new_stt_uri
        self._llm_uri = new_llm_uri
        self._wake_keyword = new_wake_keyword
        self._sleep_keyword = new_sleep_keyword
        self._active_timeout = new_active_timeout
        self._volume_level = new_volume
        self._alarm_keyword = new_alarm_keyword
        self._clear_alarm_keyword = new_clear_alarm_keyword
        self._stop_keyword = new_stop_keyword
        self._selected_alsa_device = new_alsa_device
        self._chromadb_url = new_chromadb_url
        self._llm_model = new_llm_model

        _LOGGER.info(
            "Updated config: TTS=%s, RMS=%.4f, STT URI=%s, LLM URI=%s, LLM Model=%s, Chromadb URL=%s, "
            "Wake Keyword=%s, Sleep Keyword=%s, Timeout=%s, Volume=%.1f, Alarm Keyword=%s, "
            "Clear Alarm Keyword=%s, Stop Keyword=%s, ALSA Device=%s",
            new_tts_engine,
            new_rms_threshold,
            new_stt_uri,
            new_llm_uri,
            new_llm_model,
            new_chromadb_url,
            new_wake_keyword,
            new_sleep_keyword,
            new_active_timeout,
            new_volume,
            new_alarm_keyword,
            new_clear_alarm_keyword,
            new_stop_keyword,
            new_alsa_device
        )
        # Force an immediate HA state update after reconfiguration.
        await self.async_update_ha_state(force_refresh=True)

