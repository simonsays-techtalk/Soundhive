import logging
import asyncio
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

        # Wait briefly for client state to be pushed
        await asyncio.sleep(1.5)

        # Check for state-pushed config from client
        entity_id = f"media_player.{entry.data['unique_id']}"
        state = hass.states.get(entity_id)
        _LOGGER.info("Fetched entity state: %s", state.attributes if state else "No state yet")

        if state and state.attributes.get("command") == "update_config":
            _LOGGER.info("Loading config from client state attributes")
            client_options = {
                key: state.attributes.get(key)
                for key in [
                    "tts_engine", "rms_threshold", "stt_uri", "llm_uri",
                    "wake_keyword", "sleep_keyword", "active_timeout", "volume",
                    "alarm_keyword", "clear_alarm_keyword", "client_ip", "stop_keyword"
                ]
                if state.attributes.get(key) is not None
            }

            if client_options:
                # Separate client_ip for data update
                new_ip = client_options.get("client_ip")
                if new_ip and new_ip != entry.data.get("client_ip"):
                    _LOGGER.info(f"Updating config_entry.data with new client IP: {new_ip}")
                    hass.config_entries.async_update_entry(
                        entry,
                        data={**entry.data, "client_ip": new_ip}
                    )

                # Exclude client_ip from options
                entry_options = client_options.copy()
                entry_options.pop("client_ip", None)
                if entry_options != entry.options:
                    hass.config_entries.async_update_entry(entry, options=entry_options)
                    _LOGGER.info("Client config options updated (excluding IP), waiting for update_listener to handle reload.")

        device_registry = async_get(hass)
        device_registry.async_get_or_create(
            config_entry_id=entry.entry_id,
            identifiers={(DOMAIN, entry.data["unique_id"])} ,
            manufacturer="Soundhive",
            name=entry.data.get("name", entry.data["unique_id"]),
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

    entity_id = f"media_player.{entry.data['unique_id']}"
    state = hass.states.get(entity_id)

    new_tts_engine = entry.options.get("tts_engine") or entry.data.get("tts_engine", "tts.google_translate_en_com")
    new_rms_threshold = entry.options.get("rms_threshold") or entry.data.get("rms_threshold", 0.008)
    new_stt_uri = entry.options.get("stt_uri") or entry.data.get("stt_uri", "http://localhost:10900/inference")
    new_llm_uri = entry.options.get("llm_uri") or entry.data.get("llm_uri", "http://localhost:11434/api/generate")
    new_wake_keyword = entry.options.get("wake_keyword") or entry.data.get("wake_keyword", "hey assistant")
    new_sleep_keyword = entry.options.get("sleep_keyword") or entry.data.get("sleep_keyword", "goodbye")
    new_active_timeout = entry.options.get("active_timeout") or entry.data.get("active_timeout", 30)
    new_volume = entry.options.get("volume") or entry.data.get("volume", 0.4)
    new_client_ip = entry.data.get("client_ip", "127.0.0.1")
    new_alarm_keyword = entry.options.get("alarm_keyword") or entry.data.get("alarm_keyword", "alarm now")
    new_clear_alarm_keyword = entry.options.get("clear_alarm_keyword") or entry.data.get("clear_alarm_keyword", "clear alarm")
    new_stop_keyword = entry.options.get("stop_keyword") or entry.data.get("stop_keyword", "stop")

    entity = hass.data.get(DOMAIN, {}).get(entry.entry_id + "_entity")
    if entity is not None:
        await entity.async_update_config(
            new_tts_engine,
            new_rms_threshold,
            new_stt_uri,
            new_llm_uri,
            new_wake_keyword,
            new_sleep_keyword,
            new_active_timeout,
            new_volume,
            new_alarm_keyword,
            new_clear_alarm_keyword,
            new_stop_keyword
        )

        _LOGGER.info(
            "Updated client config: TTS=%s, RMS=%.3f, STT_URI=%s, LLM_URI=%s, Timeout=%ss, Volume=%.1f, IP=%s, StopWord=%s",
            new_tts_engine, new_rms_threshold, new_stt_uri, new_llm_uri,
            new_active_timeout, new_volume, new_client_ip, new_stop_keyword
        )

        hass.states.async_set(
            entity.entity_id,
            entity.state,
            {
                "command": "update_config",
                "tts_engine": new_tts_engine,
                "rms_threshold": new_rms_threshold,
                "auth_token": entity._token,
                "ha_url": entity._ha_url,
                "stt_uri": new_stt_uri,
                "llm_uri": new_llm_uri,
                "wake_keyword": new_wake_keyword,
                "sleep_keyword": new_sleep_keyword,
                "active_timeout": new_active_timeout,
                "volume": new_volume,
                "client_ip": new_client_ip,
                "alarm_keyword": new_alarm_keyword,
                "clear_alarm_keyword": new_clear_alarm_keyword,
                "stop_keyword": new_stop_keyword,
                "entity_id": entity.entity_id,
            },
        )
    else:
        _LOGGER.error("No entity found for update_listener for entry_id: %s", entry.entry_id)

