
import logging
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import async_get
from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

async def async_setup(hass: HomeAssistant, config: dict):
    _LOGGER.info("Setting up Soundhive Media Player component.")
    return True

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry):
    try:
        _LOGGER.info("Setting up Soundhive Media Player entry: %s", entry.data)
        hass.data.setdefault(DOMAIN, {})[entry.entry_id] = entry.data

        device_registry = async_get(hass)
        device_registry.async_get_or_create(
            config_entry_id=entry.entry_id,
            identifiers={(DOMAIN, entry.data["unique_id"])},
            manufacturer="Soundhive",
            name=entry.data.get("name"),
            model="Soundhive Service",
        )

        entry.async_on_unload(entry.add_update_listener(update_listener))
        await hass.config_entries.async_forward_entry_setups(entry, ["media_player"])
        return True
    except Exception as e:
        _LOGGER.exception("Error setting up Soundhive Media Player entry: %s", e)
        return False

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry):
    _LOGGER.info("Unloading Soundhive Media Player entry: %s", entry.entry_id)
    hass.data[DOMAIN].pop(entry.entry_id, None)
    return True

async def update_listener(hass, entry):
    _LOGGER.info("Soundhive config entry updated: %s", entry.data)
    new_tts_engine = entry.options.get("tts_engine") or entry.data.get("tts_engine") or "tts.google_translate_en_com"
    new_rms_threshold = entry.options.get("rms_threshold") or entry.data.get("rms_threshold") or 0.008

    entity = hass.data.get(DOMAIN, {}).get(entry.entry_id + "_entity")
    if entity is not None:
        await entity.async_update_config(new_tts_engine, new_rms_threshold)
        _LOGGER.info("Updated client config with TTS: %s and RMS: %.4f", new_tts_engine, new_rms_threshold)

        hass.states.async_set(
            entity.entity_id,
            entity.state,
            {
                "command": "update_config",
                "tts_engine": new_tts_engine,
                "rms_threshold": new_rms_threshold,
                "auth_token": entity._token,
                "ha_url": entity._ha_url,
            },
        )
    else:
        _LOGGER.error("No entity found for update_listener for entry_id: %s", entry.entry_id)
