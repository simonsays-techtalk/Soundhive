
import logging
import uuid
import aiohttp

from homeassistant import config_entries
from homeassistant.const import CONF_NAME, CONF_TOKEN
from homeassistant.core import callback
from homeassistant.helpers.selector import selector
import voluptuous as vol

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

DEFAULT_TTS = "tts.google_translate_en_com"
DEFAULT_RMS = 0.007


class ConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    VERSION = 1
    CONNECTION_CLASS = config_entries.CONN_CLASS_LOCAL_PUSH

    async def async_step_user(self, user_input=None):
        errors = {}
        tts_engines = await self._get_tts_engines()

        if user_input is not None:
            if not user_input.get("unique_id"):
                user_input["unique_id"] = str(uuid.uuid4())

            user_input["ha_url"] = "http://localhost:8123"

            await self.async_set_unique_id(user_input["unique_id"])
            self._abort_if_unique_id_configured()

            if not await self._validate_input(user_input):
                errors["base"] = "auth_failed"
            else:
                return self.async_create_entry(
                    title=user_input.get(CONF_NAME, "Soundhive"),
                    data=user_input,
                )

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema({
                vol.Required(CONF_NAME): str,
                vol.Required(CONF_TOKEN): str,
                vol.Required("tts_engine", default=tts_engines[0]): selector({
                    "select": {
                        "options": tts_engines,
                        "mode": "dropdown"
                    }
                }),
                vol.Required("rms_threshold", default=DEFAULT_RMS): selector({
                    "number": {
                        "min": 0.001,
                        "max": 0.1,
                        "step": 0.001,
                        "mode": "slider"
                    }
                }),
                vol.Optional("unique_id"): str,
            }),
            errors=errors
        )

    async def _get_tts_engines(self):
        return sorted({
            state.entity_id for state in self.hass.states.async_all()
            if state.entity_id.startswith("tts.")
        } or [DEFAULT_TTS])

    async def _validate_input(self, data):
        url = f"{data['ha_url']}/api/config"
        headers = {
            "Authorization": f"Bearer {data[CONF_TOKEN]}",
            "Content-Type": "application/json"
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as resp:
                    return resp.status == 200
        except Exception as e:
            _LOGGER.error("Error validating HA connection: %s", e)
            return False

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        return SoundhiveOptionsFlow(config_entry)


class SoundhiveOptionsFlow(config_entries.OptionsFlow):
    def __init__(self, config_entry):
        self.config_entry = config_entry

    async def async_step_init(self, user_input=None):
        errors = {}
        tts_engines = await self._get_tts_engines()

        defaults = {
            CONF_TOKEN: self.config_entry.options.get(CONF_TOKEN, self.config_entry.data.get(CONF_TOKEN, "")),
            "tts_engine": self.config_entry.options.get("tts_engine", self.config_entry.data.get("tts_engine", DEFAULT_TTS)),
            "rms_threshold": self.config_entry.options.get("rms_threshold", self.config_entry.data.get("rms_threshold", DEFAULT_RMS)),
        }

        if user_input is not None:
            if not await self._validate_token(user_input[CONF_TOKEN], self.config_entry.data.get("ha_url", "http://localhost:8123")):
                errors["base"] = "auth_failed"
            else:
                return self.async_create_entry(title=self.config_entry.title, data=user_input)

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema({
                vol.Required(CONF_TOKEN, default=defaults[CONF_TOKEN]): str,
                vol.Required("tts_engine", default=defaults["tts_engine"]): selector({
                    "select": {
                        "options": tts_engines,
                        "mode": "dropdown"
                    }
                }),
                vol.Required("rms_threshold", default=defaults["rms_threshold"]): selector({
                    "number": {
                        "min": 0.001,
                        "max": 0.1,
                        "step": 0.001,
                        "mode": "slider"
                    }
                }),
            }),
            errors=errors
        )

    async def _get_tts_engines(self):
        return sorted({
            state.entity_id for state in self.hass.states.async_all()
            if state.entity_id.startswith("tts.")
        } or [DEFAULT_TTS])

    async def _validate_token(self, token, ha_url):
        url = f"{ha_url}/api/config"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    return response.status == 200
        except Exception as e:
            _LOGGER.error("Token validation failed: %s", e)
            return False
