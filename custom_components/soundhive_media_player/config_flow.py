import uuid
import voluptuous as vol
from homeassistant import config_entries
from homeassistant.const import CONF_NAME, CONF_TOKEN
from homeassistant.helpers.selector import selector
from homeassistant.core import callback

DOMAIN = "soundhive_media_player"
DEFAULT_TTS = "tts.google_translate_en_com"
DEFAULT_RMS = 0.008
DEFAULT_STT_URI = "http://localhost:10900/inference"
DEFAULT_LLM_URI = "http://localhost:11434/api/generate"
DEFAULT_WAKE = "hey assistant"
DEFAULT_SLEEP = "goodbye"

class ConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    VERSION = 1
    CONNECTION_CLASS = config_entries.CONN_CLASS_LOCAL_PUSH

    async def async_step_user(self, user_input=None):
        errors = {}

        if user_input is not None:
            if not user_input.get("unique_id"):
                user_input["unique_id"] = str(uuid.uuid4())

            user_input["ha_url"] = "http://localhost:8123"

            await self.async_set_unique_id(user_input["unique_id"])
            self._abort_if_unique_id_configured()

            if not await self._validate_token(user_input[CONF_TOKEN], user_input["ha_url"]):
                errors["base"] = "auth_failed"
            else:
                return self.async_create_entry(
                    title=user_input.get(CONF_NAME, "Soundhive"),
                    data=user_input,
                )

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema({
                vol.Required(CONF_NAME, default="Soundhive"): str,
                vol.Required(CONF_TOKEN): str,
                vol.Required("unique_id"): str,
            }),
            errors=errors,
        )

    async def _validate_token(self, token, ha_url):
        return True  # Replace with real token validation logic

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
            "stt_uri": self.config_entry.options.get("stt_uri", self.config_entry.data.get("stt_uri", DEFAULT_STT_URI)),
            "llm_uri": self.config_entry.options.get("llm_uri", self.config_entry.data.get("llm_uri", DEFAULT_LLM_URI)),
            "wake_keyword": self.config_entry.options.get("wake_keyword", self.config_entry.data.get("wake_keyword", DEFAULT_WAKE)),
            "sleep_keyword": self.config_entry.options.get("sleep_keyword", self.config_entry.data.get("sleep_keyword", DEFAULT_SLEEP)),
            "active_timeout": self.config_entry.options.get("active_timeout", self.config_entry.data.get("active_timeout", 30)),
            "volume": self.config_entry.options.get("volume", self.config_entry.data.get("volume", 0.4)),
            "client_ip": self.config_entry.data.get("client_ip", "127.0.0.1"),
        }

        if user_input is not None:
            user_input.pop("client_ip", None)
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
                vol.Required("stt_uri", default=defaults["stt_uri"]): str,
                vol.Required("llm_uri", default=defaults["llm_uri"]): str,
                vol.Required("wake_keyword", default=defaults["wake_keyword"]): str,
                vol.Required("sleep_keyword", default=defaults["sleep_keyword"]): str,
                vol.Required("active_timeout", default=defaults["active_timeout"]): selector({
                    "number": {
                        "min": 5,
                        "max": 60,
                        "step": 5,
                        "mode": "slider"
                    }
                }),
                vol.Required("volume", default=defaults["volume"]): selector({
                    "number": {
                        "min": 0.1,
                        "max": 1.0,
                        "step": 0.1,
                        "mode": "slider"
                    }
                }),
                vol.Optional("client_ip", default=defaults["client_ip"]): str,
            }),
            errors=errors,
        )

    async def _validate_token(self, token, ha_url):
        return True

    async def _get_tts_engines(self):
        return sorted({
            state.entity_id for state in self.hass.states.async_all()
            if state.entity_id.startswith("tts.")
        } or [DEFAULT_TTS])

