from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers.typing import ConfigType

from .const import DOMAIN
from .media_player import SoundhiveMediaPlayer

PLATFORMS = ["media_player"]

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    return True  # No YAML support

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    from .media_player import SoundhiveMediaPlayer  # avoid circular import if needed

    # Store config entry
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = entry

    # Create and store entity reference for updates
    entity = SoundhiveMediaPlayer(entry)
    hass.data[DOMAIN]["entity"] = entity

    # Register update listener
    async def update_listener(hass, updated_entry):
        entity.config_entry_updated(updated_entry)

    entry.async_on_unload(entry.add_update_listener(update_listener))

    # Forward to media_player platform (do not rely on its return value)
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    return True  # ✅ Always return boolean

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unload_ok:
        hass.data[DOMAIN].pop(entry.entry_id, None)
    return unload_ok

async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle reload of the integration entry."""
    await async_unload_entry(hass, entry)
    await async_setup_entry(hass, entry)

