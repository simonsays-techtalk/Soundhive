# soundhive/config_flow.py
# Soundhive: Configuration Flow with Fixed URL Resolution (Updated for homeassistanttest.local)
# Version: 0.4.4 (Set Fixed URL to homeassistanttest.local)

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.const import CONF_NAME, CONF_TOKEN
from .const import DOMAIN
import aiohttp
import logging

_LOGGER = logging.getLogger(__name__)

class ConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Soundhive Media Player."""

    VERSION = 1
    CONNECTION_CLASS = config_entries.CONN_CLASS_LOCAL_PUSH

    async def async_step_user(self, user_input=None):
        """Handle the initial step of the config flow."""
        errors = {}

        if user_input is not None:
            _LOGGER.debug("🔑 Received token for validation: %s", user_input.get(CONF_TOKEN, "None"))
            valid = await self._validate_input(user_input)
            if valid:
                return self.async_create_entry(title=user_input[CONF_NAME], data=user_input)
            else:
                errors["base"] = "auth_failed"
                _LOGGER.error("❌ Token validation failed for user input: %s", user_input)

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_NAME, default="Soundhive_mediaplayer"): str,
                    vol.Required(CONF_TOKEN): str,
                }
            ),
            errors=errors,
        )

    async def _validate_input(self, data):
        """Validate the user input allows connection to Home Assistant API."""
        # 🌐 Fixed URL set to homeassistanttest.local
        ha_url = "http://homeassistanttest.local:8123"
        url = f"{ha_url}/api/config"
        headers = {
            "Authorization": f"Bearer {data[CONF_TOKEN]}",
            "Content-Type": "application/json"
        }

        _LOGGER.debug("🌐 Validating token at endpoint: %s", url)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    _LOGGER.debug("📝 Token validation response status: %s", response.status)
                    if response.status != 200:
                        _LOGGER.error("❌ Invalid token. Response: %s", await response.text())
                    return response.status == 200
        except Exception as e:
            _LOGGER.error("🔥 Exception during token validation: %s", str(e))
            return False

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        return SoundhiveOptionsFlow(config_entry)


class SoundhiveOptionsFlow(config_entries.OptionsFlow):
    """Handle options flow for Soundhive Media Player."""

    def __init__(self, config_entry):
        """Initialize Soundhive options flow."""
        self.config_entry = config_entry

    async def async_step_init(self, user_input=None):
        """Manage the Soundhive options."""
        errors = {}

        if user_input is not None:
            valid = await self._validate_token(user_input[CONF_TOKEN])
            if valid:
                return self.async_create_entry(title="", data=user_input)
            else:
                errors["base"] = "auth_failed"
                _LOGGER.error("❌ Token validation failed during options flow.")

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_TOKEN, default=self.config_entry.data.get(CONF_TOKEN, "")): str,
                }
            ),
            errors=errors,
        )

    async def _validate_token(self, token):
        """Validate the updated token for options flow."""
        ha_url = "http://homeassistanttest.local:8123"
        url = f"{ha_url}/api/config"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        _LOGGER.debug("🌐 Validating updated token at endpoint: %s", url)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    _LOGGER.debug("📝 Updated token validation response status: %s", response.status)
                    if response.status != 200:
                        _LOGGER.error("❌ Invalid updated token. Response: %s", await response.text())
                    return response.status == 200
        except Exception as e:
            _LOGGER.error("🔥 Exception during updated token validation: %s", str(e))
            return False

# ✅ Version 0.4.4:
# - Updated token validation to always use homeassistanttest.local for consistency.
# - Removed dynamic URL resolution to match the current test environment.
# - Retained enhanced debug logging for troubleshooting.

