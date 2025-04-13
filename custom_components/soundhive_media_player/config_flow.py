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
        return True

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        return SoundhiveOptionsFlow(config_entry)


class SoundhiveOptionsFlow(config_entries.OptionsFlow):
    def __init__(self, config_entry):
        self._entry = config_entry

    async def async_step_init(self, user_input=None):
        errors = {}
        tts_engines = await self._get_tts_engines()

        current_data = self._entry.data
        current_options = self._entry.options

        entity_id = f"media_player.{current_data['unique_id']}"
        state = self.hass.states.get(entity_id)
        default_alsa_devices = [
            "seeed-2mic-voicecard: bcm2835-i2s-wm8960-hifi wm8960-hifi-0 (hw:2,0)",
            "capture",
            "array",
            "default"
        ]
        alsa_devices = state.attributes.get("alsa_devices", default_alsa_devices) if state else default_alsa_devices

        defaults = {
            CONF_TOKEN: current_options.get(CONF_TOKEN, current_data.get(CONF_TOKEN, "")),
            "tts_engine": current_options.get("tts_engine", current_data.get("tts_engine", DEFAULT_TTS)),
            "rms_threshold": current_options.get("rms_threshold", current_data.get("rms_threshold", DEFAULT_RMS)),
            "stt_uri": current_options.get("stt_uri", current_data.get("stt_uri", DEFAULT_STT_URI)),
            "llm_uri": current_options.get("llm_uri", current_data.get("llm_uri", DEFAULT_LLM_URI)),
            "llm_model": current_options.get("llm_model", current_data.get("llm_model", "llama3.1:8b")),
            "chromadb_url": current_options.get("chromadb_url", current_data.get("chromadb_url", "http://madison.autohome.local:8001")),
            "wake_keyword": current_options.get("wake_keyword", current_data.get("wake_keyword", DEFAULT_WAKE)),
            "sleep_keyword": current_options.get("sleep_keyword", current_data.get("sleep_keyword", DEFAULT_SLEEP)),
            "active_timeout": current_options.get("active_timeout", current_data.get("active_timeout", 30)),
            "volume": current_options.get("volume", current_data.get("volume", 0.4)),
            "alarm_keyword": current_options.get("alarm_keyword", current_data.get("alarm_keyword", "alarm now")),
            "clear_alarm_keyword": current_options.get("clear_alarm_keyword", current_data.get("clear_alarm_keyword", "clear alarm")),
            "stop_keyword": current_options.get("stop_keyword", current_data.get("stop_keyword", "stop")),
            "alsa_device": current_options.get("alsa_device", alsa_devices[0]),
            "client_ip": current_data.get("client_ip", "127.0.0.1")
        }

        if user_input is not None:
            if not await self._validate_token(user_input[CONF_TOKEN], current_data.get("ha_url", "http://localhost:8123")):
                errors["base"] = "auth_failed"
            else:
                return self.async_create_entry(title=self._entry.title, data=user_input)

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema({
                vol.Required(CONF_TOKEN, default=defaults[CONF_TOKEN]): str,
                vol.Required("tts_engine", default=defaults["tts_engine"]): selector({
                    "select": {"options": tts_engines, "mode": "dropdown"}
                }),
                vol.Required("rms_threshold", default=defaults["rms_threshold"]): selector({
                    "number": {"min": 0.001, "max": 0.1, "step": 0.001, "mode": "slider"}
                }),
                vol.Required("stt_uri", default=defaults["stt_uri"]): str,
                vol.Required("llm_uri", default=defaults["llm_uri"]): str,
                vol.Required("llm_model", default=defaults["llm_model"]): str,
                vol.Required("chromadb_url", default=defaults["chromadb_url"]): str,
                vol.Required("wake_keyword", default=defaults["wake_keyword"]): str,
                vol.Required("sleep_keyword", default=defaults["sleep_keyword"]): str,
                vol.Required("active_timeout", default=defaults["active_timeout"]): selector({
                    "number": {"min": 5, "max": 60, "step": 5, "mode": "slider"}
                }),
                vol.Required("volume", default=defaults["volume"]): selector({
                    "number": {"min": 0.1, "max": 1.0, "step": 0.1, "mode": "slider"}
                }),
                vol.Required("alarm_keyword", default=defaults["alarm_keyword"]): str,
                vol.Required("clear_alarm_keyword", default=defaults["clear_alarm_keyword"]): str,
                vol.Required("stop_keyword", default=defaults["stop_keyword"]): str,
                vol.Required("alsa_device", default=defaults["alsa_device"]): selector({
                    "select": {"options": alsa_devices, "mode": "dropdown"}
                }),
            }),
            description_placeholders={"client_ip": defaults["client_ip"]},
            errors=errors,
        )

    async def _validate_token(self, token, ha_url):
        return True

    async def _get_tts_engines(self):
        return sorted({
            state.entity_id for state in self.hass.states.async_all()
            if state.entity_id.startswith("tts.")
        } or [DEFAULT_TTS])

