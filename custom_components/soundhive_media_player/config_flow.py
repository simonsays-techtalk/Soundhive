import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.helpers.selector import selector

DOMAIN = "soundhive_media_player"

class SoundhiveConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    VERSION = 1

    async def async_step_user(self, user_input=None):
        if user_input is not None:
            return self.async_create_entry(title=user_input["name"], data=user_input)

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema({
                vol.Required("name"): str,
                vol.Required("auth_token"): str,
                vol.Required("unique_id"): str,
            }),
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        return SoundhiveOptionsFlow(config_entry)


class SoundhiveOptionsFlow(config_entries.OptionsFlow):
    def __init__(self, config_entry):
        self._entry = config_entry

    async def async_step_init(self, user_input=None):
        tts_engines = await self._get_tts_engines()

        current_data = self._entry.data
        current_options = self._entry.options

        entity_id = f"media_player.{current_data['unique_id']}"
        state = self.hass.states.get(entity_id)

        default_alsa_devices = [
            "seeed-2mic-voicecard: bcm2835-i2s-wm8960-hifi wm8960-hifi-0 (hw:2,0)",
            "capture", "array", "default"
        ]
        alsa_devices = state.attributes.get("alsa_devices", default_alsa_devices) if state else default_alsa_devices
        client_ip = current_data.get("client_ip", "127.0.0.1")

        defaults = {
            "tts_engine": current_options.get("tts_engine", current_data.get("tts_engine", "")),
            "rms_threshold": current_options.get("rms_threshold", current_data.get("rms_threshold", 0.008)),
            "stt_uri": current_options.get("stt_uri", current_data.get("stt_uri", "")),
            "llm_uri": current_options.get("llm_uri", current_data.get("llm_uri", "")),
            "llm_model": current_options.get("llm_model", current_data.get("llm_model", "")),
            "chromadb_url": current_options.get("chromadb_url", current_data.get("chromadb_url", "")),
            "wake_keyword": current_options.get("wake_keyword", current_data.get("wake_keyword", "")),
            "sleep_keyword": current_options.get("sleep_keyword", current_data.get("sleep_keyword", "")),
            "active_timeout": current_options.get("active_timeout", current_data.get("active_timeout", 30)),
            "volume": current_options.get("volume", current_data.get("volume", 0.4)),
            "alarm_keyword": current_options.get("alarm_keyword", current_data.get("alarm_keyword", "alarm now")),
            "clear_alarm_keyword": current_options.get("clear_alarm_keyword", current_data.get("clear_alarm_keyword", "clear alarm")),
            "stop_keyword": current_options.get("stop_keyword", current_data.get("stop_keyword", "stop")),
            "alsa_device": current_options.get("alsa_device", alsa_devices[0]),
        }

        if user_input is not None:
            return self.async_create_entry(title=self._entry.title, data=user_input)

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema({
                vol.Required("tts_engine", default=defaults["tts_engine"]): selector({
                    "select": {"options": tts_engines, "mode": "dropdown"}
                }),
                vol.Required("rms_threshold", default=defaults["rms_threshold"]): selector({
                    "number": {"min": 0.001, "max": 0.01, "step": 0.001, "mode": "slider"}
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
                    "number": {"min": 0.0, "max": 1.0, "step": 0.1, "mode": "slider"}
                }),
                vol.Required("alarm_keyword", default=defaults["alarm_keyword"]): str,
                vol.Required("clear_alarm_keyword", default=defaults["clear_alarm_keyword"]): str,
                vol.Required("stop_keyword", default=defaults["stop_keyword"]): str,
                vol.Required("alsa_device", default=defaults["alsa_device"]): selector({
                    "select": {"options": alsa_devices, "mode": "dropdown"}
                }),
            }),
            description_placeholders={"client_ip": client_ip}
        )

    async def _get_tts_engines(self):
        """Return available TTS engines as dropdown options."""
        tts_entities = [
            state.entity_id.split(".")[-1]
            for state in self.hass.states.async_all("tts")
        ]
        return sorted(tts_entities)

