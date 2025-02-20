# soundhive_media_player/media_player.py
# Soundhive: Custom Home Assistant MQTT Media Player with TTS Support
# Version: 0.3.0 (Updated for HA 2025.10 compatibility)
#
# Changelog:
# v0.3.0 - Replaced deprecated SUPPORT_* constants with MediaPlayerEntityFeature.* for HA Core 2025.10+ compatibility (Feb 20, 2025)

import logging
from homeassistant.components.media_player import MediaPlayerEntity
from homeassistant.components.media_player.const import (
    MediaPlayerEntityFeature
)
from homeassistant.const import STATE_IDLE, STATE_PLAYING, STATE_PAUSED

_LOGGER = logging.getLogger(__name__)

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
    _LOGGER.info("✅ Setting up Soundhive Media Player platform.")
    async_add_entities([SoundhiveMediaPlayer(hass)])

class SoundhiveMediaPlayer(MediaPlayerEntity):
    """Representation of the Soundhive Media Player."""

    def __init__(self, hass):
        """Initialize the media player."""
        self._hass = hass
        self._state = STATE_IDLE
        self._volume = 0.5
        self._name = "Soundhive Media Player"
        self._media_content_id = None

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
        """Handle playing media with support for TTS streaming."""
        _LOGGER.info(f"🎵 Received media request: {media_id} (type: {media_type})")
        self._media_content_id = media_id
        self._state = STATE_PLAYING
        self.async_write_ha_state()
        # TODO: Send MQTT command to Soundhive client for playback

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

# ✅ Version 0.3.0: Updated deprecated constants to MediaPlayerEntityFeature.* for HA Core 2025.10 compatibility.
# Next Steps:
# - Implement MQTT communication for playback actions.
# - Test TTS playback through Home Assistant's tts.speak service.
# - Confirm auto-discovery behavior in Home Assistant UI.
