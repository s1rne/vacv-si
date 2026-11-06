"""Polite HTTP client with on-disk caching of raw responses.

Every network call goes through :func:`get_json`. Responses are written to
``data/raw/<source>/`` keyed by a hash of the URL, so a second run of the
pipeline is offline and byte-identical. Rate limits are per host and
deliberately conservative: the public ChEMBL and PubChem endpoints are shared
infrastructure.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import urllib.parse
from pathlib import Path
from typing import Any

import requests

from .paths import raw_dir

log = logging.getLogger(__name__)

USER_AGENT = "vacv-si-dataset/0.1 (academic hackathon; contact via repository)"

# Minimum seconds between requests to the same host.
RATE_LIMIT = {
    "www.ebi.ac.uk": 0.15,
    "pubchem.ncbi.nlm.nih.gov": 0.25,
    "eutils.ncbi.nlm.nih.gov": 0.40,
}
DEFAULT_RATE_LIMIT = 0.5

_last_call: dict[str, float] = {}
_session: requests.Session | None = None


def session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers["User-Agent"] = USER_AGENT
    return _session


def _throttle(host: str) -> None:
    delay = RATE_LIMIT.get(host, DEFAULT_RATE_LIMIT)
    previous = _last_call.get(host)
    if previous is not None:
        wait = delay - (time.monotonic() - previous)
        if wait > 0:
            time.sleep(wait)
    _last_call[host] = time.monotonic()


def cache_path(source: str, url: str, suffix: str = ".json") -> Path:
    digest = hashlib.sha1(url.encode()).hexdigest()[:16]
    stem = urllib.parse.urlparse(url).path.strip("/").replace("/", "_")[:60]
    return raw_dir(source) / f"{stem or 'query'}-{digest}{suffix}"


def get_text(
    url: str,
    source: str,
    *,
    params: dict[str, Any] | None = None,
    refresh: bool = False,
    retries: int = 4,
    timeout: int = 300,
    suffix: str = ".json",
) -> str:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    path = cache_path(source, url, suffix)
    if path.exists() and not refresh:
        return path.read_text()

    host = urllib.parse.urlparse(url).netloc
    last: Exception | None = None
    for attempt in range(retries):
        _throttle(host)
        try:
            response = session().get(url, timeout=timeout)
        except requests.RequestException as exc:  # pragma: no cover - network
            last = exc
            time.sleep(2**attempt)
            continue
        if response.status_code == 404:
            path.write_text("")
            return ""
        if response.status_code in (429, 500, 502, 503, 504):
            last = requests.HTTPError(f"{response.status_code} for {url}")
            time.sleep(2 ** (attempt + 1))
            continue
        response.raise_for_status()
        path.write_text(response.text)
        return response.text
    raise RuntimeError(f"giving up on {url}") from last


def get_json(url: str, source: str, **kwargs: Any) -> Any:
    text = get_text(url, source, **kwargs)
    if not text:
        return None
    return json.loads(text)
