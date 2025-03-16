# config_flow.py
# VERSION = "2.5.70"
import uuid
import voluptuous as vol
from homeassistant import config_entries
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
    VERSION = 1
    CONNECTION_CLASS = config_entries.CONN_CLASS_LOCAL_PUSH

    async def async_step_user(self, user_input=None):
        # Auto-create the integration entry with default global settings.
        ha_url = get_ha_url(self.hass)
        default_tts_engine = "tts.google_translate_en_com"
        data = {
            "ha_url": ha_url,
            "default_tts_engine": default_tts_engine,
        }
        # Initialize with an empty devices list.
        options = {"devices": []}
        return self.async_create_entry(title="Soundhive Integration", data=data, options=options)

    async def _validate_token(self, token, ha_url):
        """Validate the token using the Home Assistant API."""
        url = f"{ha_url}/api/config"
        headers = {
            "Authorization": f"Bearer {token}",
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
    def async_get_options_flow(config_entry):
        return SoundhiveOptionsFlow(config_entry)

class SoundhiveOptionsFlow(config_entries.OptionsFlow):
    """Options flow for adding or editing a device (a media player entry) in Soundhive Integration."""

    def __init__(self, config_entry):
        self.config_entry = config_entry

    async def _get_tts_engines(self):
        engines = {
            state.entity_id
            for state in self.hass.states.async_all()
            if state.entity_id.startswith("tts.")
        }
        if not engines:
            engines = {"tts.google_translate_en_com"}
        return sorted(engines)

    async def _validate_token(self, token, ha_url):
        """Validate the token using the Home Assistant API (duplicated for options flow)."""
        url = f"{ha_url}/api/config"
        headers = {
            "Authorization": f"Bearer {token}",
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

    async def async_step_init(self, user_input=None):
        devices = self.config_entry.options.get("devices", [])
        # If the options flow is started with a device context, load that device directly.
        device_id = self.context.get("device_id")
        if device_id:
            return await self.async_step_edit_device(device_id)
        if devices:
            # If only one device exists, edit it directly.
            if len(devices) == 1:
                return await self.async_step_edit_device(devices[0]["unique_id"])
            else:
                # If more than one device exists, show a selection form.
                return await self.async_step_select_device(user_input)
        else:
            # No devices exist; jump directly to add device.
            return await self.async_step_add_device(user_input)

    async def async_step_select_device(self, user_input=None):
        """Step to select which device to configure."""
        devices = self.config_entry.options.get("devices", [])
        device_options = {d["unique_id"]: d["name"] for d in devices}
        schema = vol.Schema({
            vol.Required("device", default=list(device_options.keys())[0]): vol.In(device_options)
        })
        if user_input is not None:
            return await self.async_step_edit_device(user_input["device"])
        return self.async_show_form(step_id="select_device", data_schema=schema)

    async def async_step_edit_device(self, device_id, user_input=None):
        """Step to edit an existing device."""
        devices = self.config_entry.options.get("devices", [])
        device = next((d for d in devices if d["unique_id"] == device_id), None)
        if device is None:
            return self.async_abort(reason="device_not_found")
        tts_engines = await self._get_tts_engines()
        schema = vol.Schema({
            vol.Required("name", default=device["name"]): str,
            vol.Required("token", default=device["token"]): str,
            vol.Required("tts_engine", default=device.get("tts_engine", self.config_entry.data.get("default_tts_engine", tts_engines[0]))):
                vol.In(tts_engines),
        })
        errors = {}
        if user_input is not None:
            ha_url = self.config_entry.data.get("ha_url", "http://localhost:8123")
            valid = await self._validate_token(user_input["token"], ha_url)
            if valid:
                # Do not update unique_id; keep it stable.
                device["name"] = user_input["name"]
                device["token"] = user_input["token"]
                device["tts_engine"] = user_input["tts_engine"]
                new_options = {**self.config_entry.options, "devices": devices}
                self.hass.config_entries.async_update_entry(self.config_entry, options=new_options)
                return self.async_create_entry(title="Device Updated", data=new_options)
            else:
                errors["base"] = "auth_failed"
                return self.async_show_form(step_id="edit_device", data_schema=schema, errors=errors)
        return self.async_show_form(step_id="edit_device", data_schema=schema)

    async def async_step_add_device(self, user_input=None):
        """Step to add a new device."""
        tts_engines = await self._get_tts_engines()
        schema = vol.Schema({
            vol.Required("name", default="Soundhive Media Player"): str,
            vol.Required("token"): str,
            vol.Required("tts_engine", default=self.config_entry.data.get("default_tts_engine", tts_engines[0])):
                vol.In(tts_engines),
        })
        errors = {}
        if user_input is not None:
            ha_url = self.config_entry.data.get("ha_url", "http://localhost:8123")
            valid = await self._validate_token(user_input["token"], ha_url)
            if valid:
                # Generate a stable unique id that doesn't change.
                unique_id = str(uuid.uuid4())
                devices = self.config_entry.options.get("devices", [])
                if any(d.get("unique_id") == unique_id for d in devices):
                    errors["base"] = "device_exists"
                    return self.async_show_form(step_id="add_device", data_schema=schema, errors=errors)
                devices.append({
                    "unique_id": unique_id,
                    "name": user_input["name"],
                    "token": user_input["token"],
                    "tts_engine": user_input["tts_engine"],
                })
                new_options = {**self.config_entry.options, "devices": devices}
                self.hass.config_entries.async_update_entry(self.config_entry, options=new_options)
                return self.async_create_entry(title="Device Added", data=new_options)
            else:
                errors["base"] = "auth_failed"
                return self.async_show_form(step_id="add_device", data_schema=schema, errors=errors)
        return self.async_show_form(step_id="add_device", data_schema=schema)
