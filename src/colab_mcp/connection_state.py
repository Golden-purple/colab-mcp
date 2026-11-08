# Copyright 2026 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Persists the WebSocket token/port between colab-mcp runs.

Colab's frontend stores `mcpProxyToken`/`mcpProxyPort` in the tab's
sessionStorage and auto-connects to them whenever the notebook (re)loads.
Re-using the same token and port across server restarts therefore lets a
Colab tab that was connected once reconnect to a brand new agent session
with a simple page reload, instead of opening a new notebook every time.
"""

import dataclasses
import json
import logging
import os
from pathlib import Path

DEFAULT_STATE_FILE = Path.home() / ".colab-mcp" / "connection.json"


@dataclasses.dataclass(frozen=True)
class ConnectionState:
    token: str
    port: int

    @property
    def connect_string(self) -> str:
        """The value Colab's 'Connect to a local Colab MCP server' dialog expects."""
        return f"{self.token}&{self.port}"


def load_state(path: Path = DEFAULT_STATE_FILE) -> ConnectionState | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        token, port = data["token"], int(data["port"])
        if not isinstance(token, str) or not token or not 0 < port < 65536:
            raise ValueError("invalid token/port")
        return ConnectionState(token=token, port=port)
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError) as e:
        logging.warning(f"Ignoring unreadable connection state file {path}: {e}")
        return None


def save_state(state: ConnectionState, path: Path = DEFAULT_STATE_FILE) -> None:
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(dataclasses.asdict(state)), encoding="utf-8")
        with_restricted_permissions(tmp)
        os.replace(tmp, path)
    except OSError as e:
        logging.warning(f"Could not save connection state to {path}: {e}")


def with_restricted_permissions(path: Path) -> None:
    # best effort: owner read/write only (largely a no-op on Windows)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
