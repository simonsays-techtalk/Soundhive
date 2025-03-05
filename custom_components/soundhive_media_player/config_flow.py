#VERSION = "1.1.04"
import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.const import CONF_NAME, CONF_TOKEN
from .const import DOMAIN
import aiohttp
import logging

_LOGGER = logging.getLogger(__name__)

def get_ha_url(hass):
    """Retrieve the Home Assistant URL from core configuration."""
    if hasattr(hass.config, "internal_url") and hass.config.internal_url:
        return hass.config.internal_url
    elif hasattr(hass.config, "external_url") and hass.config.external_url:
        return hass.config.external_url
    else:
        return "http://localhost:8123"

class ConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Soundhive Media Player."""

    VERSION = 1
    CONNECTION_CLASS = config_entries.CONN_CLASS_LOCAL_PUSH

    async def _get_tts_engines(self):
        """Retrieve a sorted list of TTS engine entity_ids from Home Assistant."""
        engines = {state.entity_id for state in self.hass.states.async_all() if state.entity_id.startswith("tts.")}
        if not engines:
            engines = {"tts.google_translate_en_com"}
        return sorted(engines)

    async def async_step_user(self, user_input=None):
        """Handle the initial step of the config flow."""
        errors = {}
        tts_engines = await self._get_tts_engines()
        schema = vol.Schema({
            vol.Required("ha_url", default="http://localhost:8123"): str,
            vol.Required(CONF_NAME, default="Soundhive_mediaplayer"): str,
            vol.Required(CONF_TOKEN): str,
            vol.Required("tts_engine", default=tts_engines[0]): vol.In(tts_engines),
        })

        if user_input is not None:
            _LOGGER.debug("Received input for validation: %s", user_input)
            valid = await self._validate_input(user_input)
            if valid:
                return self.async_create_entry(title=user_input[CONF_NAME], data=user_input)
            else:
                errors["base"] = "auth_failed"
                _LOGGER.error("Token validation failed for input: %s", user_input)

        return self.async_show_form(
            step_id="user",
            data_schema=schema,
            errors=errors,
        )

    async def _validate_input(self, data):
        """Validate the user input by connecting to the Home Assistant API using the provided URL."""
        ha_url = data.get("ha_url", "http://localhost:8123")
        url = f"{ha_url}/api/config"
        headers = {
            "Authorization": f"Bearer {data[CONF_TOKEN]}",
            "Content-Type": "application/json"
        }
        _LOGGER.debug("Validating token at endpoint: %s", url)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    _LOGGER.debug("Token validation response status: %s", response.status)
                    if response.status != 200:
                        _LOGGER.error("Invalid token. Response: %s", await response.text())
                    return response.status == 200
        except Exception as e:
            _LOGGER.error("Exception during token validation: %s", str(e))
            return False

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        return SoundhiveOptionsFlow(config_entry)

class SoundhiveOptionsFlow(config_entries.OptionsFlow):
    """Handle options flow for Soundhive Media Player."""

    def __init__(self, config_entry):
        # The base class already stores the config entry as self.config_entry.
        pass

    async def _get_tts_engines(self):
        """Retrieve a sorted list of TTS engine entity_ids from Home Assistant."""
        engines = {state.entity_id for state in self.hass.states.async_all() if state.entity_id.startswith("tts.")}
        if not engines:
            engines = {"tts.google_translate_en_com"}
        return sorted(engines)

    async def async_step_init(self, user_input=None):
        """Manage the Soundhive options."""
        errors = {}
        tts_engines = await self._get_tts_engines()
        schema = vol.Schema({
            vol.Required("ha_url", default=self.config_entry.data.get("ha_url", "http://localhost:8123")): str,
            vol.Required(CONF_TOKEN, default=self.config_entry.data.get(CONF_TOKEN, "")): str,
            vol.Required("tts_engine", default=self.config_entry.data.get("tts_engine", tts_engines[0])): vol.In(tts_engines),
        })
        if user_input is not None:
            valid = await self._validate_token(user_input[CONF_TOKEN], user_input.get("ha_url"))
            if valid:
                return self.async_create_entry(title="", data=user_input)
            else:
                errors["base"] = "auth_failed"
                _LOGGER.error("Token validation failed during options flow.")

        return self.async_show_form(
            step_id="init",
            data_schema=schema,
            errors=errors,
        )

    async def _validate_token(self, token, ha_url):
        """Validate the updated token for options flow."""
        ha_url = ha_url or "http://localhost:8123"
        url = f"{ha_url}/api/config"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        _LOGGER.debug("Validating updated token at endpoint: %s", url)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    _LOGGER.debug("Updated token validation response status: %s", response.status)
                    if response.status != 200:
                        _LOGGER.error("Invalid updated token. Response: %s", await response.text())
                    return response.status == 200
        except Exception as e:
            _LOGGER.error("Exception during updated token validation: %s", str(e))
            return False


