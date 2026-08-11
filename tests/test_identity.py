import time

import pytest
from fastapi import HTTPException
from pytest import MonkeyPatch
from starlette.datastructures import Headers
from starlette.requests import Request

from studylife_ai.api import identity as identity_module
from studylife_ai.api.identity import PROXY_TOKEN_HEADER, _sign, resolve_identity
from studylife_ai.config import Settings

_SECRET = "test-shared-secret"


def _request(headers: dict[str, str]) -> Request:
    scope = {"type": "http", "headers": Headers(headers).raw}
    return Request(scope)


def _token(user_id: str, expiry: int, secret: str = _SECRET) -> str:
    payload = f"{user_id}.{expiry}"
    return f"{payload}.{_sign(payload, secret)}"


def _patch_secret(monkeypatch: MonkeyPatch, secret: str | None) -> None:
    monkeypatch.setattr(
        identity_module,
        "get_settings",
        lambda: Settings(studylife_shared_secret=secret),  # type: ignore[arg-type]
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
