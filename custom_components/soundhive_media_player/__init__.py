# VERSION = "2.5.40"
import logging
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

async def async_setup(hass: HomeAssistant, config: dict):
    _LOGGER.info("Setting up Soundhive Media Player component.")
    return True

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry):
    try:
        _LOGGER.info("Setting up Soundhive Media Player entry: %s", entry.data)
        # Store the config data (if needed)
        hass.data.setdefault(DOMAIN, {})[entry.entry_id] = entry.data

        # Register the update listener so that option changes trigger our callback.
        entry.async_on_unload(entry.add_update_listener(update_listener))
        
        # Forward the config entry to the media_player platform so that the entity is created.
        await hass.config_entries.async_forward_entry_setups(entry, ["media_player"])
        return True
    except Exception as e:
        _LOGGER.exception("Error setting up Soundhive Media Player entry: %s", e)
        return False

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry):
    _LOGGER.info("Unloading Soundhive Media Player entry: %s", entry.entry_id)
    hass.data[DOMAIN].pop(entry.entry_id, None)
    return True

async def update_listener(hass: HomeAssistant, entry: ConfigEntry):
    _LOGGER.info("Soundhive config entry updated: %s", entry.data)
    new_tts_engine = entry.options.get("tts_engine", entry.data.get("tts_engine"))
    _LOGGER.info("New TTS engine in config entry: %s", new_tts_engine)
    
    # Retrieve the entity reference stored by the media_player platform.
    entity = hass.data.get(DOMAIN, {}).get(entry.entry_id + "_entity")
    if entity is not None:
         await entity.async_update_config(new_tts_engine)
         _LOGGER.info("Requested client config update with new TTS engine: %s", new_tts_engine)
    else:
         _LOGGER.error("No entity found for update_listener for entry_id: %s", entry.entry_id)
