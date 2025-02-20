# soundhive_media_player/media_player.py
from homeassistant.components.media_player import MediaPlayerEntity
from homeassistant.components.media_player.const import (
    SUPPORT_PLAY,
    SUPPORT_PAUSE,
    SUPPORT_STOP,
    SUPPORT_VOLUME_SET,
    SUPPORT_TURN_ON,
    SUPPORT_TURN_OFF,
    SUPPORT_PLAY_MEDIA
)
from homeassistant.const import STATE_IDLE, STATE_PLAYING, STATE_PAUSED

_LOGGER = logging.getLogger(__name__)

SUPPORT_SOUNDHIVE = (
    SUPPORT_PLAY | SUPPORT_PAUSE | SUPPORT_STOP |
    SUPPORT_VOLUME_SET | SUPPORT_TURN_ON | SUPPORT_TURN_OFF |
    SUPPORT_PLAY_MEDIA
)

async def async_setup_platform(hass, config, async_add_entities, discovery_info=None):
    """Set up the Soundhive Media Player entity."""
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
        self._state = STATE_IDLE
        self.async_write_ha_state()

    async def async_turn_off(self):
        self._state = STATE_IDLE
        self.async_write_ha_state()

    async def async_play_media(self, media_type, media_id, **kwargs):
        """Handle playing media with support for TTS streaming."""
        _LOGGER.info(f"🎵 Received media request: {media_id}")
        self._media_content_id = media_id
        self._state = STATE_PLAYING
        self.async_write_ha_state()
        # TODO: Send MQTT command to Soundhive client for playback

    async def async_media_pause(self):
        self._state = STATE_PAUSED
        self.async_write_ha_state()

    async def async_media_stop(self):
        self._state = STATE_IDLE
        self.async_write_ha_state()

    async def async_media_play(self):
        self._state = STATE_PLAYING
        self.async_write_ha_state()

    async def async_set_volume_level(self, volume):
        self._volume = volume
        self.async_write_ha_state()
