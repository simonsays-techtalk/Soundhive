# soundhive_media_player.py
# Soundhive Custom Component: Version 1.2.31
# Changelog:
# - Fixed TTS playback by converting media-source URLs to full URLs
# - Updated pause/resume handling for accurate UI state sync
# - Improved MQTT topic management and client communication
# - Enhanced logging for TTS and streaming issues

import logging
import voluptuous as vol
import asyncio
import json
from homeassistant.components.media_player import (
    MediaPlayerEntity,
    PLATFORM_SCHEMA,
    MediaType,
    MediaPlayerEntityFeature,
)
from homeassistant.const import CONF_NAME, STATE_IDLE, STATE_PLAYING, STATE_PAUSED
import homeassistant.helpers.config_validation as cv
from homeassistant.core import callback
from homeassistant.components.mqtt import async_publish, async_subscribe

_LOGGER = logging.getLogger(__name__)

DEFAULT_NAME = "Soundhive Media Player"
DEFAULT_UNIQUE_ID = "soundhive_mediaplayer_default"
HA_BASE_URL = "http://homeassistanttest.local:8123"

PLATFORM_SCHEMA = PLATFORM_SCHEMA.extend({
    vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
    vol.Optional("unique_id", default=DEFAULT_UNIQUE_ID): cv.string,
})

SUPPORT_FLAGS = (
    MediaPlayerEntityFeature.PLAY
    | MediaPlayerEntityFeature.PAUSE
    | MediaPlayerEntityFeature.STOP
    | MediaPlayerEntityFeature.VOLUME_SET
    | MediaPlayerEntityFeature.PLAY_MEDIA
)

class SoundhiveMediaPlayer(MediaPlayerEntity):
    def __init__(self, name, unique_id):
        self._name = name
        self._unique_id = unique_id
        self._state = STATE_IDLE
        self._volume = 0.5
        self._media_content_id = None
        self._media_title = None

    @property
    def name(self):
        return self._name

    @property
    def unique_id(self):
        return self._unique_id

    @property
    def state(self):
        return self._state

    @property
    def supported_features(self):
        return SUPPORT_FLAGS

    @property
    def volume_level(self):
        return self._volume

    @property
    def media_content_id(self):
        return self._media_content_id

    @property
    def media_title(self):
        return self._media_title

    async def async_set_volume_level(self, volume):
        self._volume = volume
        _LOGGER.debug("🔊 Volume set to: %s", volume)
        await self._publish_state("volume_changed")
        await self.async_update_ha_state()

    async def async_media_play(self):
        if self._media_content_id:
            self._state = STATE_PLAYING
            _LOGGER.debug("▶️ Playback started.")
            await self._trigger_mqtt_command("resume")
            await self._publish_state("playing")
            await self.async_update_ha_state()

    async def async_media_pause(self):
        if self._state == STATE_PLAYING:
            self._state = STATE_PAUSED
            _LOGGER.debug("⏸️ Playback paused.")
            await self._trigger_mqtt_command("pause")
            await self._publish_state("paused")
            await self.async_update_ha_state()

    async def async_media_stop(self):
        if self._state in [STATE_PLAYING, STATE_PAUSED]:
            self._state = STATE_IDLE
            _LOGGER.debug("⏹️ Playback stopped.")
            await self._trigger_mqtt_command("stop")
            await self._publish_state("idle")
            await self.async_update_ha_state()

    async def async_play_media(self, media_type: str, media_id: str, **kwargs):
        _LOGGER.debug("🎬 async_play_media triggered | Type: %s | ID: %s", media_type, media_id)
        if media_id.startswith("media-source://"):
            media_id = media_id.replace("media-source://tts", f"{HA_BASE_URL}/media/local/tts_proxy")
            _LOGGER.debug("🔄 Transformed media ID for TTS: %s", media_id)

        if media_type in [MediaType.MUSIC, MediaType.URL, MediaType.SOUND]:
            self._media_content_id = media_id
            self._media_title = media_id.split("/")[-1]
            _LOGGER.debug("🎵 Now playing (async): %s", self._media_title)
            self._state = STATE_PLAYING
            await self._trigger_mqtt_playback(media_id)
            await self._publish_state("playing")
            await self.async_update_ha_state()
        else:
            _LOGGER.error("❌ Unsupported media type: %s", media_type)

    async def _trigger_mqtt_playback(self, media_url):
        payload = json.dumps({
            "command": "play",
            "args": {"filepath": media_url, "media_type": "audio/mp3"}
        })
        topic = f"{self._unique_id}/command"
        _LOGGER.debug("📡 Sending MQTT play command: %s to topic: %s", payload, topic)
        await async_publish(self.hass, topic, payload, qos=0, retain=False)

    async def _trigger_mqtt_command(self, action):
        payload = json.dumps({"command": action})
        topic = f"{self._unique_id}/command"
        _LOGGER.debug("📡 Sending MQTT action command: %s to topic: %s", payload, topic)
        await async_publish(self.hass, topic, payload, qos=0, retain=False)

    async def _publish_state(self, state):
        payload = json.dumps({"state": state, "media_title": self._media_title})
        topic = f"{self._unique_id}/state"
        _LOGGER.debug("📡 Publishing MQTT state update: %s to topic: %s", payload, topic)
        await async_publish(self.hass, topic, payload, qos=0, retain=False)

    async def async_added_to_hass(self):
        _LOGGER.info("✅ Soundhive Media Player '%s' added with unique_id '%s'",
                     self._name, self._unique_id)
        await self.async_update_ha_state()

    @callback
    def handle_mqtt_message(self, msg, payload, qos):
        _LOGGER.debug("📨 MQTT message received on topic %s: %s", msg.topic, payload)
        try:
            data = json.loads(payload)
            new_state = data.get("state", STATE_IDLE)
            if new_state != self._state:
                self._state = new_state
                _LOGGER.debug("🔄 State updated to: %s", self._state)
                asyncio.create_task(self.async_update_ha_state())
        except json.JSONDecodeError as e:
            _LOGGER.error("❌ Failed to decode MQTT message: %s", e)

    async def async_will_remove_from_hass(self):
        _LOGGER.info("❌ Soundhive Media Player '%s' removed.", self._name)

async def async_setup_platform(hass, config, async_add_entities, discovery_info=None):
    name = config.get(CONF_NAME)
    unique_id = config.get("unique_id")
    entity = SoundhiveMediaPlayer(name, unique_id)
    async_add_entities([entity], True)

    async def message_received(msg, payload, qos):
        _LOGGER.debug("📩 State message received: %s", payload)
        entity.handle_mqtt_message(msg, payload, qos)

    topic = f"{unique_id}/state"
    await async_subscribe(
        hass, topic, message_received
    )
    _LOGGER.info("🎬 Soundhive Media Player v1.2.31 setup complete and subscribed to MQTT state updates on topic: %s.", topic)
