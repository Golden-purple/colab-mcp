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

import pytest

from colab_mcp import session
from colab_mcp.connection_state import ConnectionState, load_state, save_state


def test_round_trip(tmp_path):
    path = tmp_path / "sub" / "connection.json"
    assert load_state(path) is None
    save_state(ConnectionState("tok", 4321), path)
    assert load_state(path) == ConnectionState("tok", 4321)


def test_connect_string():
    assert ConnectionState("tok", 4321).connect_string == "tok&4321"


@pytest.mark.parametrize(
    "content",
    ["not json", "{}", '{"token": "", "port": 1}', '{"token": "t", "port": 0}'],
)
def test_bad_state_ignored(tmp_path, content):
    path = tmp_path / "connection.json"
    path.write_text(content)
    assert load_state(path) is None


@pytest.mark.asyncio
async def test_session_proxy_reuses_saved_token_and_port(tmp_path):
    state_file = tmp_path / "connection.json"
    options = session.ConnectionOptions(state_file=state_file)

    first = session.ColabSessionProxy(options)
    await first.start_proxy_server()
    saved = load_state(state_file)
    assert saved == ConnectionState(first.wss.token, first.wss.port)
    await first.cleanup()

    second = session.ColabSessionProxy(options)
    await second.start_proxy_server()
    try:
        assert (second.wss.token, second.wss.port) == (saved.token, saved.port)
    finally:
        await second.cleanup()


@pytest.mark.asyncio
async def test_session_proxy_port_taken_keeps_other_instances_state(tmp_path):
    state_file = tmp_path / "connection.json"
    options = session.ConnectionOptions(state_file=state_file)

    owner = session.ColabSessionProxy(options)
    await owner.start_proxy_server()
    other = session.ColabSessionProxy(options)
    await other.start_proxy_server()
    try:
        assert other.wss.port != owner.wss.port
        # the running owner's saved port is left untouched
        assert load_state(state_file).port == owner.wss.port
    finally:
        await other.cleanup()
        await owner.cleanup()
