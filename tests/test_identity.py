import logging
import time

import pytest
from fastapi import HTTPException
from pytest import LogCaptureFixture, MonkeyPatch
from starlette.datastructures import Headers
from starlette.requests import Request

from studylife_ai.api import identity as identity_module
from studylife_ai.api.identity import PROXY_TOKEN_HEADER, _sign, resolve_identity
from studylife_ai.config import Settings

_SECRET = "test-shared-secret"
_SIGNING_SECRET_CONFIG = "v1:signing-secret-one,v2:signing-secret-two"


def _request(headers: dict[str, str]) -> Request:
    scope = {"type": "http", "headers": Headers(headers).raw}
    return Request(scope)


def _token(user_id: str, expiry: int, secret: str = _SECRET) -> str:
    """Legacy, un-keyed 3-part format."""
    payload = f"{user_id}.{expiry}"
    return f"{payload}.{_sign(payload, secret)}"


def _keyed_token(user_id: str, expiry: int, kid: str, secret: str) -> str:
    """New, key-id-tagged 4-part format (audit A5)."""
    payload = f"{user_id}.{expiry}"
    return f"{payload}.{kid}.{_sign(payload, secret)}"


def _patch_secret(
    monkeypatch: MonkeyPatch,
    secret: str | None,
    *,
    signing_secret: str | None = None,
) -> None:
    monkeypatch.setattr(
        identity_module,
        "get_settings",
        lambda: Settings(  # type: ignore[call-arg]
            studylife_shared_secret=secret, studylife_token_signing_secret=signing_secret
        ),
    )


def test_resolve_identity_returns_user_id_for_a_valid_token(monkeypatch: MonkeyPatch) -> None:
    _patch_secret(monkeypatch, _SECRET)
    token = _token("42", int(time.time()) + 60)

    identity = resolve_identity(_request({PROXY_TOKEN_HEADER: token}))

    assert identity.user_id == "42"


def test_resolve_identity_raises_401_when_header_is_missing(monkeypatch: MonkeyPatch) -> None:
    _patch_secret(monkeypatch, _SECRET)

    with pytest.raises(HTTPException) as exc_info:
        resolve_identity(_request({}))

    assert exc_info.value.status_code == 401


@pytest.mark.parametrize("token", ["not-three-parts", "a.b.c.d", ""])
def test_resolve_identity_raises_401_for_malformed_token(
    monkeypatch: MonkeyPatch, token: str
) -> None:
    _patch_secret(monkeypatch, _SECRET)

    with pytest.raises(HTTPException) as exc_info:
        resolve_identity(_request({PROXY_TOKEN_HEADER: token}))

    assert exc_info.value.status_code == 401


def test_resolve_identity_raises_401_for_a_bad_signature(monkeypatch: MonkeyPatch) -> None:
    _patch_secret(monkeypatch, _SECRET)
    token = _token("42", int(time.time()) + 60, secret="wrong-secret")

    with pytest.raises(HTTPException) as exc_info:
        resolve_identity(_request({PROXY_TOKEN_HEADER: token}))

    assert exc_info.value.status_code == 401


def test_resolve_identity_raises_401_for_an_expired_token(monkeypatch: MonkeyPatch) -> None:
    _patch_secret(monkeypatch, _SECRET)
    token = _token("42", int(time.time()) - 1)

    with pytest.raises(HTTPException) as exc_info:
        resolve_identity(_request({PROXY_TOKEN_HEADER: token}))

    assert exc_info.value.status_code == 401


def test_resolve_identity_raises_401_for_a_non_integer_expiry(monkeypatch: MonkeyPatch) -> None:
    _patch_secret(monkeypatch, _SECRET)
    payload = "42.not-a-number"
    token = f"{payload}.{_sign(payload, _SECRET)}"

    with pytest.raises(HTTPException) as exc_info:
        resolve_identity(_request({PROXY_TOKEN_HEADER: token}))

    assert exc_info.value.status_code == 401


def test_resolve_identity_raises_503_when_secret_is_not_configured(
    monkeypatch: MonkeyPatch,
) -> None:
    _patch_secret(monkeypatch, None)
    token = _token("42", int(time.time()) + 60)

    with pytest.raises(HTTPException) as exc_info:
        resolve_identity(_request({PROXY_TOKEN_HEADER: token}))

    assert exc_info.value.status_code == 503


# --- Audit A5: new key-id-tagged token format, with a legacy 3-part fallback ---


def test_resolve_identity_accepts_the_new_format_signed_with_the_first_kid(
    monkeypatch: MonkeyPatch,
) -> None:
    _patch_secret(monkeypatch, None, signing_secret=_SIGNING_SECRET_CONFIG)
    token = _keyed_token("42", int(time.time()) + 60, "v1", "signing-secret-one")

    identity = resolve_identity(_request({PROXY_TOKEN_HEADER: token}))

    assert identity.user_id == "42"


def test_resolve_identity_accepts_the_new_format_signed_with_an_older_kid(
    monkeypatch: MonkeyPatch,
) -> None:
    """Rotation (audit A5): every kid still listed in STUDYLIFE_TOKEN_SIGNING_SECRET verifies,
    not just the first (signing) one - so a token minted just before a rotation keeps working
    until it naturally expires."""
    _patch_secret(monkeypatch, None, signing_secret=_SIGNING_SECRET_CONFIG)
    token = _keyed_token("42", int(time.time()) + 60, "v2", "signing-secret-two")

    identity = resolve_identity(_request({PROXY_TOKEN_HEADER: token}))

    assert identity.user_id == "42"


def test_resolve_identity_raises_401_for_the_new_format_with_an_unknown_kid(
    monkeypatch: MonkeyPatch,
) -> None:
    _patch_secret(monkeypatch, None, signing_secret=_SIGNING_SECRET_CONFIG)
    token = _keyed_token("42", int(time.time()) + 60, "v99", "signing-secret-one")

    with pytest.raises(HTTPException) as exc_info:
        resolve_identity(_request({PROXY_TOKEN_HEADER: token}))

    assert exc_info.value.status_code == 401


def test_resolve_identity_raises_401_for_the_new_format_with_a_wrong_secret(
    monkeypatch: MonkeyPatch,
) -> None:
    _patch_secret(monkeypatch, None, signing_secret=_SIGNING_SECRET_CONFIG)
    token = _keyed_token("42", int(time.time()) + 60, "v1", "wrong-secret")

    with pytest.raises(HTTPException) as exc_info:
        resolve_identity(_request({PROXY_TOKEN_HEADER: token}))

    assert exc_info.value.status_code == 401


def test_resolve_identity_raises_401_for_the_new_format_when_signing_secret_is_unset(
    monkeypatch: MonkeyPatch,
) -> None:
    """STUDYLIFE_TOKEN_SIGNING_SECRET unset, only the legacy secret configured: a 4-part token
    has nothing to verify against - it must not silently fall back to legacy verification."""
    _patch_secret(monkeypatch, _SECRET, signing_secret=None)
    token = _keyed_token("42", int(time.time()) + 60, "v1", "signing-secret-one")

    with pytest.raises(HTTPException) as exc_info:
        resolve_identity(_request({PROXY_TOKEN_HEADER: token}))

    assert exc_info.value.status_code == 401


def test_resolve_identity_raises_503_for_a_malformed_signing_secret_configuration(
    monkeypatch: MonkeyPatch,
) -> None:
    _patch_secret(monkeypatch, None, signing_secret="not-a-valid-entry")
    token = _keyed_token("42", int(time.time()) + 60, "v1", "signing-secret-one")

    with pytest.raises(HTTPException) as exc_info:
        resolve_identity(_request({PROXY_TOKEN_HEADER: token}))

    assert exc_info.value.status_code == 503


def test_resolve_identity_accepts_the_legacy_format_when_both_secrets_are_configured(
    monkeypatch: MonkeyPatch,
) -> None:
    """Rollout compatibility (audit A5): a legacy 3-part token still works even after
    STUDYLIFE_TOKEN_SIGNING_SECRET is configured, as long as the legacy secret is too - so
    StudyLife's backend and studylife-ai can deploy the split independently, in either order."""
    _patch_secret(monkeypatch, _SECRET, signing_secret=_SIGNING_SECRET_CONFIG)
    token = _token("42", int(time.time()) + 60)

    identity = resolve_identity(_request({PROXY_TOKEN_HEADER: token}))

    assert identity.user_id == "42"


def test_resolve_identity_logs_a_deprecation_warning_once_for_the_legacy_fallback(
    monkeypatch: MonkeyPatch, caplog: LogCaptureFixture
) -> None:
    _patch_secret(monkeypatch, _SECRET, signing_secret=None)
    token = _token("42", int(time.time()) + 60)

    with caplog.at_level(logging.WARNING):
        resolve_identity(_request({PROXY_TOKEN_HEADER: token}))
        caplog.clear()
        resolve_identity(_request({PROXY_TOKEN_HEADER: token}))

    assert caplog.text.count("STUDYLIFE_TOKEN_SIGNING_SECRET is not set") == 0


def test_resolve_identity_logs_a_deprecation_warning_the_first_time_the_legacy_fallback_is_used(
    monkeypatch: MonkeyPatch, caplog: LogCaptureFixture
) -> None:
    _patch_secret(monkeypatch, _SECRET, signing_secret=None)
    token = _token("42", int(time.time()) + 60)

    with caplog.at_level(logging.WARNING):
        resolve_identity(_request({PROXY_TOKEN_HEADER: token}))

    assert "STUDYLIFE_TOKEN_SIGNING_SECRET is not set" in caplog.text


def test_resolve_identity_does_not_warn_when_the_new_signing_secret_is_configured(
    monkeypatch: MonkeyPatch, caplog: LogCaptureFixture
) -> None:
    _patch_secret(monkeypatch, _SECRET, signing_secret=_SIGNING_SECRET_CONFIG)
    token = _token("42", int(time.time()) + 60)

    with caplog.at_level(logging.WARNING):
        resolve_identity(_request({PROXY_TOKEN_HEADER: token}))

    assert "STUDYLIFE_TOKEN_SIGNING_SECRET is not set" not in caplog.text
