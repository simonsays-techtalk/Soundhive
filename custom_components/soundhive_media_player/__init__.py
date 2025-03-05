# soundhive/__init__.py
# Soundhive: Component Initialization with Config Flow Support
# VERSION = "1.1.02" (Updated to Use const.py for DOMAIN)

import logging
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from .const import DOMAIN  # Updated import

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

    return True

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry):
    """Unload a config entry."""
    _LOGGER.info("🛑 Unloading Soundhive Media Player entry: %s", entry.entry_id)

    hass.data[DOMAIN].pop(entry.entry_id)
    return True

# ✅ Version "1.1.02":
# - Created const.py with DOMAIN definition.
# - Updated __init__.py to import DOMAIN from const.py for consistency.
# - This change ensures alignment with Home Assistant's best practices and resolves potential handler issues.

