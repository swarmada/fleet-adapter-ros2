# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""An object store speaking the Amazon object-storage API, for what an ObjectStore ArtifactStore holds.

Only what the adapter needs: put one object, get one object. No listing and no deletion, so a
credential scoped to exactly that is enough (the adapter writes its own maps and reads the
keep-out sets rendered for it). Requests are signed with AWS Signature Version 4 from the
standard library alone, so the adapter takes no new dependency onto a robot.

The credential is read from the adapter's environment (``SWARMADA_STORE_ACCESS_KEY``,
``SWARMADA_STORE_SECRET_KEY``), never from a flag, whose value any process listing would
show, and never written anywhere. No message this module raises carries it.
"""

from __future__ import annotations

import datetime
import hashlib
import hmac
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping

ACCESS_KEY_ENV = "SWARMADA_STORE_ACCESS_KEY"
SECRET_KEY_ENV = "SWARMADA_STORE_SECRET_KEY"

_UNSIGNED_SAFE = "-_.~/"


class StoreError(Exception):
    """The store could not answer, or answered no. Names the object, never the credential."""


def _hmac(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode(), hashlib.sha256).digest()


def sign(method: str, url: str, headers: Mapping[str, str], payload_sha256: str, *,
         access_key: str, secret_key: str, amz_date: str, region: str = "us-east-1",
         service: str = "s3") -> str:
    """The Authorization header value for a request (AWS Signature Version 4).

    ``headers`` are the headers to sign, ``host`` among them; ``amz_date`` is
    ``YYYYMMDDTHHMMSSZ`` and must also be sent as ``x-amz-date``."""
    parts = urllib.parse.urlsplit(url)
    path = urllib.parse.quote(parts.path or "/", safe=_UNSIGNED_SAFE)
    query = "&".join(
        f"{urllib.parse.quote(k, safe='-_.~')}={urllib.parse.quote(v, safe='-_.~')}"
        for k, v in sorted(urllib.parse.parse_qsl(parts.query, keep_blank_values=True)))
    canon = {k.lower().strip(): " ".join(str(v).split()) for k, v in headers.items()}
    signed = ";".join(sorted(canon))
    request = "\n".join([method, path, query,
                         "".join(f"{k}:{canon[k]}\n" for k in sorted(canon)),
                         signed, payload_sha256])
    scope = f"{amz_date[:8]}/{region}/{service}/aws4_request"
    to_sign = "\n".join(["AWS4-HMAC-SHA256", amz_date, scope,
                         hashlib.sha256(request.encode()).hexdigest()])
    key = _hmac(("AWS4" + secret_key).encode(), amz_date[:8])
    for part in (region, service, "aws4_request"):
        key = _hmac(key, part)
    signature = hmac.new(key, to_sign.encode(), hashlib.sha256).hexdigest()
    return (f"AWS4-HMAC-SHA256 Credential={access_key}/{scope}, SignedHeaders={signed}, "
            f"Signature={signature}")


def _utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


class ObjectStore:
    """One bucket at one endpoint (``http[s]://host[:port]``), addressed path-style."""

    def __init__(self, endpoint: str, bucket: str, access_key: str, secret_key: str, *,
                 region: str = "us-east-1", timeout: float = 15.0,
                 opener: Callable = urllib.request.urlopen,
                 now: Callable[[], datetime.datetime] = _utc_now) -> None:
        parts = urllib.parse.urlsplit(endpoint)
        if parts.scheme not in ("http", "https") or not parts.netloc or parts.path not in ("", "/"):
            raise ValueError("the store endpoint must be http[s]://host[:port] with no path")
        if not bucket or "/" in bucket:
            raise ValueError("the bucket must be one plain name")
        if not access_key or not secret_key:
            raise ValueError(f"the store credential is not set ({ACCESS_KEY_ENV}, "
                             f"{SECRET_KEY_ENV})")
        self.endpoint, self.bucket = f"{parts.scheme}://{parts.netloc}", bucket
        self._host = parts.netloc
        self._access, self._secret, self._region = access_key, secret_key, region
        self._timeout, self._open, self._now = timeout, opener, now

    @classmethod
    def from_env(cls, endpoint: str, bucket: str, environ: Mapping[str, str], **kw):
        return cls(endpoint, bucket, environ.get(ACCESS_KEY_ENV, ""),
                   environ.get(SECRET_KEY_ENV, ""), **kw)

    def describe(self) -> str:
        """The store as messages name it: the bucket. The endpoint is not named: on a robot
        run it is an address on a site's own network, and these messages are kept."""
        return f"bucket {self.bucket} at --store-endpoint"

    def _request(self, method: str, key: str, body: bytes = b""):
        if not key or key.startswith("/") or ".." in key.split("/"):
            raise StoreError(f"{key!r} is not an object name in {self.bucket}")
        url = f"{self.endpoint}/{self.bucket}/{urllib.parse.quote(key, safe=_UNSIGNED_SAFE)}"
        amz_date = self._now().strftime("%Y%m%dT%H%M%SZ")
        digest = hashlib.sha256(body).hexdigest()
        headers = {"host": self._host, "x-amz-content-sha256": digest, "x-amz-date": amz_date}
        auth = sign(method, url, headers, digest, access_key=self._access,
                    secret_key=self._secret, amz_date=amz_date, region=self._region)
        req = urllib.request.Request(url, data=body if method == "PUT" else None, method=method,
                                     headers={**headers, "Authorization": auth})
        return self._open(req, timeout=self._timeout)

    def put(self, key: str, data: bytes) -> None:
        """Store ``data`` at ``key``. The store's ETag of a single-part object is its md5,
        so the bytes that arrived are compared with the bytes sent: md5 at this hop too."""
        try:
            with self._request("PUT", key, data) as resp:
                etag = (resp.headers.get("ETag") or "").strip('"')
        except urllib.error.HTTPError as exc:
            raise StoreError(f"{self.bucket}/{key}: the store refused the upload "
                             f"(HTTP {exc.code})") from None
        except (urllib.error.URLError, OSError) as exc:
            raise StoreError(f"{self.bucket}/{key}: the store did not answer "
                             f"({getattr(exc, 'reason', exc)})") from None
        want = hashlib.md5(data).hexdigest()
        if etag != want:
            raise StoreError(f"{self.bucket}/{key}: the store holds md5 {etag or '(none)'}, "
                             f"not the {want} sent")

    def get(self, key: str) -> bytes | None:
        """The object at ``key``; None when there is none."""
        try:
            with self._request("GET", key) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise StoreError(f"{self.bucket}/{key}: the store refused the read "
                             f"(HTTP {exc.code})") from None
        except (urllib.error.URLError, OSError) as exc:
            raise StoreError(f"{self.bucket}/{key}: the store did not answer "
                             f"({getattr(exc, 'reason', exc)})") from None
