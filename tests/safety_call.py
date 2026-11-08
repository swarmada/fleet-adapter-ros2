# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""A SafetyStream call as a test's control plane returns it, shaped like gRPC's.

The adapter opens its estop path only once the control plane's response headers have come on the
SafetyStream: the control plane sends them when it serves the stream. gRPC's call object has
``initial_metadata()`` (blocks until the headers come, or the call ends), ``done()`` and
``cancel()``; a test's fake control plane returns a plain generator, so it wraps it in this.

* ``served``: the headers were sent as the stream was accepted (the control plane serves it).
* ``refused``: the call has already ended, with no headers; iterating it raises its error.
* ``silent``: no headers, ever, until the adapter cancels the call.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator


class SafetyCall:
    def __init__(self, stream: Iterator, mode: str = "served") -> None:
        if mode not in ("served", "refused", "silent"):
            raise ValueError(mode)
        self._stream = stream
        self.headers = threading.Event()
        self._ended = threading.Event()
        if mode == "served":
            self.headers.set()
        elif mode == "refused":
            self._ended.set()

    def initial_metadata(self) -> tuple:
        while not (self.headers.wait(0.005) or self._ended.is_set()):
            pass
        return ()

    def done(self) -> bool:
        return self._ended.is_set()

    def cancel(self) -> bool:
        self._ended.set()
        return True

    def __iter__(self) -> SafetyCall:
        return self

    def __next__(self):
        try:
            return next(self._stream)
        except BaseException:
            self._ended.set()
            raise
