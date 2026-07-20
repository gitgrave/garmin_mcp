"""Unit tests for _GarminProxy: runtime exception translation."""

import pytest
from unittest.mock import Mock

from garminconnect import (
    GarminConnectAuthenticationError,
    GarminConnectConnectionError,
    GarminConnectTooManyRequestsError,
)

from garmin_mcp import _GarminProxy, GarminNotFoundError, GarminClientError


class TestGarminProxy:
    """Tests for _GarminProxy."""

    def _proxy(self, **methods):
        client = Mock()
        for name, behaviour in methods.items():
            if isinstance(behaviour, Exception):
                getattr(client, name).side_effect = behaviour
            else:
                getattr(client, name).return_value = behaviour
        return _GarminProxy(client)

    def test_successful_call_passes_through(self):
        proxy = self._proxy(get_full_name="Alice")
        assert proxy.get_full_name() == "Alice"

    def test_non_callable_attribute_passes_through(self):
        client = Mock()
        client.some_attr = 42
        proxy = _GarminProxy(client)
        assert proxy.some_attr == 42

    def test_auth_error_message_is_actionable(self):
        proxy = self._proxy(get_activities=GarminConnectAuthenticationError("expired"))
        exc = pytest.raises(GarminConnectAuthenticationError, proxy.get_activities)
        assert "Re-run 'garmin-mcp-auth'" in str(exc.value)

    def test_rate_limit_error_message_is_actionable(self):
        proxy = self._proxy(get_activities=GarminConnectTooManyRequestsError("429"))
        exc = pytest.raises(GarminConnectTooManyRequestsError, proxy.get_activities)
        assert "Wait a few minutes" in str(exc.value)

    def test_connection_error_message_is_actionable(self):
        proxy = self._proxy(get_steps_data=GarminConnectConnectionError("timeout"))
        exc = pytest.raises(GarminConnectConnectionError, proxy.get_steps_data)
        assert "unreachable" in str(exc.value)

    def test_404_maps_to_not_found_not_unreachable(self):
        # The library raises GarminConnectConnectionError for a missing resource
        # (e.g. deleting an already-deleted workout) with an "API Error 404" message.
        proxy = self._proxy(
            delete_workout=GarminConnectConnectionError(
                "API Error 404 - {'message': None, 'error': 'NotFoundException'}"
            )
        )
        exc = pytest.raises(GarminNotFoundError, proxy.delete_workout, 123)
        assert "not found" in str(exc.value).lower()
        assert "unreachable" not in str(exc.value)

    def test_other_4xx_surfaces_status_not_unreachable(self):
        proxy = self._proxy(
            get_workout_by_id=GarminConnectConnectionError("API Error 400 - bad request")
        )
        exc = pytest.raises(GarminClientError, proxy.get_workout_by_id, 1)
        assert "HTTP 400" in str(exc.value)
        assert "unreachable" not in str(exc.value)

    def test_5xx_still_treated_as_unreachable(self):
        proxy = self._proxy(
            get_steps_data=GarminConnectConnectionError("API Error 503 - unavailable")
        )
        exc = pytest.raises(GarminConnectConnectionError, proxy.get_steps_data)
        assert "unreachable" in str(exc.value)

    def test_unknown_exception_is_re_raised_unchanged(self):
        proxy = self._proxy(get_activities=ValueError("unexpected"))
        with pytest.raises(ValueError, match="unexpected"):
            proxy.get_activities()

    def test_args_and_kwargs_forwarded_to_client(self):
        client = Mock()
        client.get_activities.return_value = []
        proxy = _GarminProxy(client)
        proxy.get_activities(0, 10, activityType="running")
        client.get_activities.assert_called_once_with(0, 10, activityType="running")
