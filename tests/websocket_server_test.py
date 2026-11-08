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

import anyio
import asyncio
from colab_mcp.websocket_server import ColabWebSocketServer
from mcp.types import JSONRPCRequest, JSONRPCResponse, JSONRPCMessage
from mcp.shared.message import SessionMessage
import pytest
import websockets


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "origin_domain", ["https://colab.google.com", "https://colab.research.google.com"]
)
async def test_successful_connection(origin_domain):
    async with ColabWebSocketServer() as server:
        client = await websockets.connect(
            f"ws://localhost:{server.port}",
            origin=origin_domain,
            subprotocols=["mcp"],
            additional_headers={"Authorization": f"Bearer {server.token}"},
        )
        assert server.connection_live.is_set()
        assert server.connection_lock.locked()

        await client.close()
        await client.wait_closed()
        await asyncio.sleep(1)  # Allow server to update state

        assert not server.connection_live.is_set()
        assert not server.connection_lock.locked()


@pytest.mark.asyncio
async def test_unauthorized_origin_rejected():
    async with ColabWebSocketServer() as server:
        with pytest.raises(websockets.exceptions.InvalidStatus):
            await websockets.connect(
                f"ws://localhost:{server.port}",
                origin="https://wrong.com",
                subprotocols=["mcp"],
                additional_headers={"Authorization": f"Bearer {server.token}"},
            )
        assert not server.connection_live.is_set()


async def _connect(server):
    return await websockets.connect(
        f"ws://localhost:{server.port}",
        origin="https://colab.google.com",
        subprotocols=["mcp"],
        additional_headers={"Authorization": f"Bearer {server.token}"},
    )


@pytest.mark.asyncio
async def test_newer_connection_replaces_older():
    async with ColabWebSocketServer() as server:
        client1 = await _connect(server)
        assert server.connection_live.is_set()
        assert server.connection_generation == 1

        client2 = await _connect(server)

        # the stale/older tab is kicked out...
        with pytest.raises(
            websockets.exceptions.ConnectionClosed,
            match="Replaced by a newer Colab connection",
        ):
            await asyncio.wait_for(client1.recv(), timeout=2)

        # ...and the newer one is served
        for _ in range(20):
            if server.connection_generation == 2:
                break
            await asyncio.sleep(0.05)
        assert server.connection_generation == 2
        assert server.connection_live.is_set()
        pong = await client2.ping()
        assert await pong > 0
        await client2.close()


@pytest.mark.asyncio
async def test_disconnect_ends_read_stream():
    """In-flight MCP requests must fail fast (not hang) when the tab goes away."""
    async with ColabWebSocketServer() as server:
        client = await _connect(server)
        read_stream = server.read_stream
        closed = server.connection_closed
        await client.close()
        await asyncio.wait_for(closed.wait(), timeout=2)
        with pytest.raises(anyio.EndOfStream):
            await asyncio.wait_for(read_stream.receive(), timeout=1)


@pytest.mark.asyncio
async def test_reconnect_gets_fresh_streams():
    async with ColabWebSocketServer() as server:
        client = await _connect(server)
        first_stream = server.read_stream
        await client.close()
        await asyncio.wait_for(server.connection_closed.wait(), timeout=2)

        client = await _connect(server)
        assert server.connection_live.is_set()
        assert server.read_stream is not first_stream
        await client.send(
            JSONRPCResponse(jsonrpc="2.0", id="x", result={}).model_dump_json()
        )
        msg = await asyncio.wait_for(server.read_stream.receive(), timeout=1)
        assert msg.message.root.id == "x"
        await client.close()


@pytest.mark.asyncio
async def test_preferred_port_and_token_reused():
    async with ColabWebSocketServer() as first:
        port = first.port
    async with ColabWebSocketServer(port=port, token="saved-token") as server:
        assert server.port == port
        assert server.token == "saved-token"


@pytest.mark.asyncio
async def test_preferred_port_busy_falls_back():
    async with ColabWebSocketServer() as first:
        async with ColabWebSocketServer(port=first.port) as second:
            assert second.port not in (0, first.port)


@pytest.mark.asyncio
async def test_incoming_message_handling():
    async with ColabWebSocketServer() as server:
        client = await websockets.connect(
            f"ws://localhost:{server.port}",
            origin="https://colab.google.com",
            subprotocols=["mcp"],
            additional_headers={"Authorization": f"Bearer {server.token}"},
        )
        assert server.connection_live.is_set()

        test_message = JSONRPCResponse(
            jsonrpc="2.0",
            id="abc",
            result={"result": "success"},
            additional_headers={"Authorization": f"Bearer {server.token}"},
        )
        await client.send(test_message.model_dump_json())

        received_msg = await asyncio.wait_for(server.read_stream.receive(), timeout=1)
        test_json_message = JSONRPCMessage(test_message)
        assert received_msg.message == test_json_message

        await client.close()


@pytest.mark.asyncio
async def test_outgoing_message_handling():
    async with ColabWebSocketServer() as server:
        client = await websockets.connect(
            f"ws://localhost:{server.port}",
            origin="https://colab.google.com",
            subprotocols=["mcp"],
            additional_headers={"Authorization": f"Bearer {server.token}"},
        )
        assert server.connection_live.is_set()

        test_message = JSONRPCRequest(
            jsonrpc="2.0",
            id="abc",
            method="test_method",
            params={"bar": "baz"},
        )
        await server.write_stream.send(SessionMessage(test_message))

        received_msg_str = await asyncio.wait_for(client.recv(), timeout=1)
        received_msg = JSONRPCRequest.model_validate_json(received_msg_str)
        assert received_msg == test_message

        await client.close()


@pytest.mark.asyncio
async def test_malformed_incoming_message():
    async with ColabWebSocketServer() as server:
        client = await websockets.connect(
            f"ws://localhost:{server.port}",
            origin="https://colab.google.com",
            subprotocols=["mcp"],
            additional_headers={"Authorization": f"Bearer {server.token}"},
        )
        assert server.connection_live.is_set()

        await client.send("this is not json")

        received_item = await asyncio.wait_for(server.read_stream.receive(), timeout=1)
        assert isinstance(received_item, Exception)

        await client.close()


@pytest.mark.asyncio
async def test_bad_token():
    with pytest.raises(
        websockets.exceptions.InvalidStatus,
        check=lambda e: e.response.status_code == 403,
    ):
        async with ColabWebSocketServer() as server:
            await websockets.connect(
                f"ws://localhost:{server.port}",
                origin="https://colab.google.com",
                subprotocols=["mcp"],
                additional_headers={"Authorization": "Bearer bad_token"},
            )


@pytest.mark.asyncio
async def test_no_auth():
    with pytest.raises(
        websockets.exceptions.InvalidStatus,
        check=lambda e: e.response.status_code == 401,
    ):
        async with ColabWebSocketServer() as server:
            await websockets.connect(
                f"ws://localhost:{server.port}",
                origin="https://colab.google.com",
                subprotocols=["mcp"],
            )


@pytest.mark.asyncio
async def test_malformed_auth_header():
    with pytest.raises(
        websockets.exceptions.InvalidStatus,
        check=lambda e: e.response.status_code == 400,
    ):
        async with ColabWebSocketServer() as server:
            await websockets.connect(
                f"ws://localhost:{server.port}",
                origin="https://colab.google.com",
                subprotocols=["mcp"],
                additional_headers={"Authorization": f"Bearer?{server.token}"},
            )


@pytest.mark.asyncio
async def test_token_in_url():
    async with ColabWebSocketServer() as server:
        client = await websockets.connect(
            f"ws://localhost:{server.port}?access_token={server.token}",
            origin="https://colab.google.com",
            subprotocols=["mcp"],
        )
        assert server.connection_live.is_set()
        assert server.connection_lock.locked()

        await client.close()
        await client.wait_closed()
        await asyncio.sleep(1)  # Allow server to update state

        assert not server.connection_live.is_set()
        assert not server.connection_lock.locked()
