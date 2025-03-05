# VERSION = "1.1.02"

import logging
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

async def async_setup(hass: HomeAssistant, config: dict):
    """Set up the Soundhive Media Player component from YAML configuration (if any)."""
    _LOGGER.info("✅ Soundhive Media Player component is being set up.")
    return True

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry):
    """Set up Soundhive Media Player from a config entry (UI flow)."""
    _LOGGER.info("🚀 Setting up Soundhive Media Player entry: %s", entry.data)
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = entry.data

    # Forward the config entry to the media_player platform so that the entity is created.
    await hass.config_entries.async_forward_entry_setups(entry, ["media_player"])

    return True

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry):
    """Unload a config entry."""
    _LOGGER.info("🛑 Unloading Soundhive Media Player entry: %s", entry.entry_id)
    hass.data[DOMAIN].pop(entry.entry_id)
    return True

