# soundhive_media_player/media_player.py
# Soundhive: Custom Home Assistant MQTT Media Player with TTS Support
# Version: 0.5.0 (Fixed TTS playback - Convert media-source URLs to HTTP)
#
# Changelog:
# v0.5.0 - Implemented media-source URL resolution to accessible HTTP paths for proper TTS playback (Feb 20, 2025)

import logging
import asyncio
import json
import paho.mqtt.client as mqtt
from homeassistant.components.media_player import MediaPlayerEntity
from homeassistant.components.media_player.const import (
    MediaPlayerEntityFeature
)
from homeassistant.const import STATE_IDLE, STATE_PLAYING, STATE_PAUSED

_LOGGER = logging.getLogger(__name__)

# MQTT Configuration (Ensure these match your broker settings)
MQTT_BROKER = "192.168.188.62"
MQTT_PORT = 1883
MQTT_USER = "test"
MQTT_PASSWORD = "test"
MQTT_TOPIC_COMMAND = "selfhosted_mediaplayer/command"

SUPPORT_SOUNDHIVE = (
    MediaPlayerEntityFeature.PLAY |
    MediaPlayerEntityFeature.PAUSE |
    MediaPlayerEntityFeature.STOP |
    MediaPlayerEntityFeature.VOLUME_SET |
    MediaPlayerEntityFeature.TURN_ON |
    MediaPlayerEntityFeature.TURN_OFF |
    MediaPlayerEntityFeature.PLAY_MEDIA
)

async def async_setup_platform(hass, config, async_add_entities, discovery_info=None):
    """Set up the Soundhive Media Player entity."""
    _LOGGER.info("✅ Setting up Soundhive Media Player platform with MQTT support.")
    async_add_entities([SoundhiveMediaPlayer(hass)])

class SoundhiveMediaPlayer(MediaPlayerEntity):
    """Representation of the Soundhive Media Player with MQTT integration."""

    def __init__(self, hass):
        """Initialize the media player and MQTT client."""
        self._hass = hass
        self._state = STATE_IDLE
        self._volume = 0.5
        self._name = "Soundhive Media Player"
        self._media_content_id = None
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
        self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        self.mqtt_client.loop_start()
        _LOGGER.info("📡 MQTT client connected and loop started.")

    @property
    def name(self):
        return self._name

    @property
    def state(self):
        return self._state

    @property
    def supported_features(self):
        return SUPPORT_SOUNDHIVE

    async def async_turn_on(self):
        _LOGGER.info("🔌 Turning on Soundhive Media Player.")
        self._state = STATE_IDLE
        self.async_write_ha_state()

    async def async_turn_off(self):
        _LOGGER.info("🔌 Turning off Soundhive Media Player.")
        self._state = STATE_IDLE
        self.async_write_ha_state()

    async def async_play_media(self, media_type, media_id, **kwargs):
        """Handle playing media with support for TTS streaming via MQTT."""
        _LOGGER.info(f"🎵 Received media request: {media_id} (type: {media_type})")
        self._media_content_id = media_id
        self._state = STATE_PLAYING
        self.async_write_ha_state()

        # Resolve media-source:// URLs to HTTP URLs
        if media_id.startswith("media-source://"):
            try:
                media_id = await self._hass.helpers.media_source.async_resolve_media(media_id)
                media_id = media_id.url
                _LOGGER.info(f"🔗 Resolved media-source URL to: {media_id}")
            except Exception as e:
                _LOGGER.error(f"❌ Failed to resolve media-source URL: {e}")
                self._state = STATE_IDLE
                self.async_write_ha_state()
                return

        # Send MQTT command to the Soundhive MQTT client
        mqtt_payload = {
            "command": "play",
            "args": {
                "filepath": media_id
            }
        }
        try:
            self.mqtt_client.publish(MQTT_TOPIC_COMMAND, json.dumps(mqtt_payload))
            _LOGGER.info(f"📡 Published MQTT command for playback: {mqtt_payload}")
        except Exception as e:
            _LOGGER.error(f"❌ Failed to publish MQTT command: {e}")
            self._state = STATE_IDLE
            self.async_write_ha_state()

    async def async_media_pause(self):
        _LOGGER.info("⏸ Pausing media.")
        self._state = STATE_PAUSED
        self.async_write_ha_state()

    async def async_media_stop(self):
        _LOGGER.info("⏹ Stopping media.")
        self._state = STATE_IDLE
        self.async_write_ha_state()

    async def async_media_play(self):
        _LOGGER.info("▶️ Resuming media.")
        self._state = STATE_PLAYING
        self.async_write_ha_state()

    async def async_set_volume_level(self, volume):
        _LOGGER.info(f"🔊 Setting volume to {volume * 100}%.")
        self._volume = volume
        self.async_write_ha_state()

# ✅ Version 0.5.0: Fixed TTS playback by resolving media-source URLs to HTTP URLs before MQTT publishing.
# Next Steps:
# - Test TTS playback with resolved URLs on Soundhive MQTT client v4.0.0.
# - Refine state feedback based on client responses.
# - Confirm stable playback loop and TTS audio output.
