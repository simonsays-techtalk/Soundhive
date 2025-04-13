import logging
import asyncio
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import async_get
from homeassistant.helpers import entity_registry as er
from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

async def async_setup(hass: HomeAssistant, config: dict):
    _LOGGER.info("Setting up Soundhive Media Player component.")
    return True

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry):
    _LOGGER.info("Setting up Soundhive Media Player entry: %s", entry.data)
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = entry.data

    # Determine the entity IDs
    managed_entity_id = f"media_player.{entry.data['unique_id']}"
    name_slug = entry.data.get("name", "").lower().replace(" ", "_")
    potential_ghost_id = f"media_player.{name_slug}"

    # Clean up any ghost (unmanaged) entity with no unique_id
    entity_registry = er.async_get(hass)
    ghost_entity = entity_registry.async_get(potential_ghost_id)
    if ghost_entity and not ghost_entity.unique_id:
        _LOGGER.warning("🧼 Removing ghost entity: %s (no unique_id)", potential_ghost_id)
        try:
            entity_registry.async_remove(potential_ghost_id)
        except Exception as e:
            _LOGGER.warning("Could not remove registry entity: %s", e)
        if hass.states.get(potential_ghost_id):
            hass.states.async_remove(potential_ghost_id)

    # Prevent ghost state from returning after reboot
    async def remove_ghost_state(event):
        if event.data.get("entity_id") == potential_ghost_id:
            _LOGGER.info("🧽 Removing ghost entity state from event: %s", potential_ghost_id)
            hass.states.async_remove(potential_ghost_id)

    hass.bus.async_listen("state_changed", remove_ghost_state)

    # Wait up to 10s for the managed entity to appear in HA state registry
    state = None
    for _ in range(20):
        await asyncio.sleep(0.5)
        state = hass.states.get(managed_entity_id)
        if state and "tts_engine" in state.attributes:
            break

    if state:
        _LOGGER.info("Fetched managed entity state: %s", state.attributes)
        state_attrs = state.attributes

        # Sync IP
        new_ip = state_attrs.get("client_ip")
        if new_ip and new_ip != entry.data.get("client_ip"):
            _LOGGER.info("Updating config_entry.data with new client IP: %s", new_ip)
            hass.config_entries.async_update_entry(
                entry,
                data={**entry.data, "client_ip": new_ip}
            )

        # Sync options
        updated_options = {
            key: value for key, value in state_attrs.items()
            if key in [
                "tts_engine", "rms_threshold", "stt_uri", "llm_uri",
                "wake_keyword", "sleep_keyword", "active_timeout", "volume",
                "alarm_keyword", "clear_alarm_keyword", "stop_keyword", "alsa_device", "alsa_devices"
            ]
        }
        if updated_options and updated_options != entry.options:
            _LOGGER.info("Updating config_entry.options with values from client state")
            hass.config_entries.async_update_entry(
                entry,
                options=updated_options
            )

    # Register the device in HA device registry
    device_registry = async_get(hass)
    device_registry.async_get_or_create(
        config_entry_id=entry.entry_id,
        identifiers={(DOMAIN, entry.data["unique_id"])},
        manufacturer="Soundhive",
        name=entry.data.get("name", entry.data["unique_id"]),
        model="Soundhive Service",
        sw_version="Soundhive Client",
        configuration_url=entry.data.get("ha_url")
    )

    entry.async_on_unload(entry.add_update_listener(update_listener))
    await hass.config_entries.async_forward_entry_setups(entry, ["media_player"])
    _LOGGER.info("✅ Soundhive media_player setup complete for %s", managed_entity_id)
    return True

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry):
    _LOGGER.info("Unloading Soundhive Media Player entry: %s", entry.entry_id)
    hass.data[DOMAIN].pop(entry.entry_id, None)
    return True

async def update_listener(hass, entry):
    _LOGGER.info("Soundhive config entry updated: %s", entry.data)

    def get_opt(key, default):
        return entry.options.get(key) or entry.data.get(key, default)

    entity = hass.data.get("soundhive_media_player", {}).get(entry.entry_id + "_entity")
    if entity:
        await entity.async_update_config(
            get_opt("tts_engine", "tts.google_translate_en_com"),
            get_opt("rms_threshold", 0.008),
            get_opt("stt_uri", "http://localhost:10900/inference"),
            get_opt("llm_uri", "http://localhost:11434/api/generate"),
            get_opt("wake_keyword", "hey assistant"),
            get_opt("sleep_keyword", "goodbye"),
            get_opt("active_timeout", 30),
            get_opt("volume", 0.4),
            get_opt("alarm_keyword", "alarm now"),
            get_opt("clear_alarm_keyword", "clear alarm"),
            get_opt("stop_keyword", "stop"),
            get_opt("alsa_device", "default"),
            get_opt("chromadb_url", "http://madison.autohome.local:8001"),
            get_opt("llm_model", "llama3.1:8b")
        )

        hass.states.async_set(
            entity.entity_id,
            entity.state,
            {
                "command": "update_config",
                "tts_engine": get_opt("tts_engine", ""),
                "rms_threshold": get_opt("rms_threshold", 0.008),
                "auth_token": entity._token,
                "ha_url": entity._ha_url,
                "stt_uri": get_opt("stt_uri", ""),
                "llm_uri": get_opt("llm_uri", ""),
                "wake_keyword": get_opt("wake_keyword", ""),
                "sleep_keyword": get_opt("sleep_keyword", ""),
                "active_timeout": get_opt("active_timeout", 30),
                "volume": get_opt("volume", 0.4),
                "client_ip": entry.data.get("client_ip", "127.0.0.1"),
                "alarm_keyword": get_opt("alarm_keyword", ""),
                "clear_alarm_keyword": get_opt("clear_alarm_keyword", ""),
                "stop_keyword": get_opt("stop_keyword", ""),
                "alsa_device": get_opt("alsa_device", "default"),
                "chromadb_url": get_opt("chromadb_url", "http://madison.autohome.local:8001"),
                "llm_model": get_opt("llm_model", "llama3.1:8b"),
                "entity_id": entity.entity_id,
            }
        )
    else:
        _LOGGER.warning("No entity found for update_listener on entry_id: %s", entry.entry_id)

