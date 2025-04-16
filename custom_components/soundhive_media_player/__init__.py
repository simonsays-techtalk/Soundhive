from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers.typing import ConfigType

from .const import DOMAIN

PLATFORMS = ["media_player"]

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up Soundhive Media Player integration (YAML not supported)."""
    return True

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up a config entry."""
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = entry

    # Forward config entry to platform(s)
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    # ✅ Correct update listener with 2 arguments
    async def update_listener(hass: HomeAssistant, updated_entry: ConfigEntry):
        await hass.config_entries.async_reload(updated_entry.entry_id)

    entry.async_on_unload(entry.add_update_listener(update_listener))

    return True

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unload_ok:
        hass.data[DOMAIN].pop(entry.entry_id, None)
    return unload_ok

async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload a config entry."""
    await async_unload_entry(hass, entry)
    await async_setup_entry(hass, entry)

