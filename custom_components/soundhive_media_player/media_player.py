import logging
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

    entity = SoundhiveMediaPlayer(hass, name, unique_id, ha_url, tts_engine, token, rms_threshold)
    async_add_entities([entity])

    hass.data.setdefault("soundhive_media_player", {})[entry.entry_id + "_entity"] = entity

    platform = async_get_current_platform()
    platform.async_register_entity_service(
        "update_config",
        {vol.Required("tts_engine"): cv.string},
        "async_update_config",
    )

class SoundhiveMediaPlayer(MediaPlayerEntity):
    def __init__(self, hass, name, unique_id, ha_url, tts_engine, token, rms_threshold,
                 stt_uri=None, llm_uri=None, wake_keyword=None, sleep_keyword=None,
                 active_timeout=30, volume=0.4, alarm_keyword="alarm now", clear_alarm_keyword="clear alarm", stop_keyword="stop"):

        self.hass = hass
        self._name = name
        self._unique_id = unique_id
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
        new_stop_keyword
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

        _LOGGER.info(
            "Updated config: TTS=%s, RMS=%.4f, STT URI=%s, LLM URI=%s, Wake Keyword=%s, Sleep Keyword=%s, Timeout=%s, Volume=%.1f, Alarm Keyword=%s, Clear Alarm Keyword=%s, Stop Keyword=%s",
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
            new_stop_keyword
        )

