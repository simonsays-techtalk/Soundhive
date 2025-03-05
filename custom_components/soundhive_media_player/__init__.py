# VERSION = "1.1.03" (Updated to Use const.py for DOMAIN)
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
    _LOGGER.info("🚀 Setting up Soundhive Media Player entry: %s", entry.data)
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = entry.data

    # Forward the entry to the media_player platform using the new API
    await hass.config_entries.async_forward_entry_setups(entry, ["media_player"])

    return True

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry):
    """Unload a config entry."""
    _LOGGER.info("🛑 Unloading Soundhive Media Player entry: %s", entry.entry_id)
    hass.data[DOMAIN].pop(entry.entry_id)
    return True

import logging
import json
import os
import subprocess

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry):
    _LOGGER.info("🚀 Setting up Soundhive Media Player entry: %s", entry.data)
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = entry.data

    # Register the update listener
    entry.async_on_unload(entry.add_update_listener(update_listener))

    # Forward the entry to the media_player platform (using the new API)
    await hass.config_entries.async_forward_entry_setups(entry, ["media_player"])

    return True

async def update_listener(hass: HomeAssistant, entry: ConfigEntry):
    """Called when the options are updated in the UI."""
    _LOGGER.info("Soundhive config entry updated: %s", entry.data)
    new_tts_engine = entry.data.get("tts_engine")
    # Define the absolute path to the client's config file.
    client_config_path = os.path.expanduser("~/Soundhive/client/soundhive_config.json")
    try:
        with open(client_config_path, "r") as f:
            config_data = json.load(f)
    except Exception as e:
        _LOGGER.error("Failed to load client config from %s: %s", client_config_path, e)
        return

    # Update the TTS engine in the client config.
    config_data["tts_engine"] = new_tts_engine

    try:
        with open(client_config_path, "w") as f:
            json.dump(config_data, f, indent=4)
        _LOGGER.info("Updated client config with new TTS engine: %s", new_tts_engine)
    except Exception as e:
        _LOGGER.error("Failed to write updated client config: %s", e)
        return

    # Restart the client service so that it picks up the new config.
    try:
        subprocess.run(["systemctl", "restart", "soundhive.service"], check=True)
        _LOGGER.info("Client service restarted successfully.")
    except Exception as e:
        _LOGGER.error("Failed to restart client service: %s", e)

