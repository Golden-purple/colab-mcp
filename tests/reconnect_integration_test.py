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

"""End-to-end: a fake Colab tab connects, goes away, and a new tab takes over."""

import asyncio
import contextlib
import json

import pytest
import websockets

from colab_mcp import session


async def fake_colab_tab(port: int, token: str, name: str, stop: asyncio.Event):
    """Minimal stand-in for Colab's frontend MCP server over the websocket."""
    async with websockets.connect(
        f"ws://localhost:{port}?access_token={token}",
        origin="https://colab.research.google.com",
        subprotocols=["mcp"],
    ) as ws:

        async def serve():
            async for raw in ws:
                msg = json.loads(raw)
                if "id" not in msg or "method" not in msg:
                    continue  # notifications / responses
                if msg["method"] == "initialize":
                    result = {
                        "protocolVersion": msg["params"]["protocolVersion"],
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": name, "version": "0"},
                    }
                elif msg["method"] == "tools/list":
                    result = {
                        "tools": [{"name": "whoami", "inputSchema": {"type": "object"}}]
                    }
                elif msg["method"] == "tools/call":
                    result = {"content": [{"type": "text", "text": name}]}
                else:
                    result = {}
                await ws.send(
                    json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": result})
                )

        serve_task = asyncio.create_task(serve())
        await stop.wait()
        serve_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await serve_task


async def call_whoami(proxy_client: session.ColabProxyClient) -> str:
    client = proxy_client.client_factory()
    async with client:
        result = await asyncio.wait_for(client.call_tool("whoami", {}), timeout=5)
    return result.content[0].text


@pytest.mark.asyncio
async def test_tab_reconnect_and_takeover():
    proxy = session.ColabSessionProxy(
        session.ConnectionOptions(persist_connection=False)
    )
    await proxy.start_proxy_server()
    proxy_client = proxy.middleware[0].proxy_client
    wss = proxy.wss
    try:
        # tab 1 connects and works
        stop1 = asyncio.Event()
        tab1 = asyncio.create_task(fake_colab_tab(wss.port, wss.token, "tab-1", stop1))
        assert await proxy_client.await_proxy_connection(5)
        assert await call_whoami(proxy_client) == "tab-1"

        # tab 1 is closed/reloaded -> proxy notices
        stop1.set()
        await tab1
        await asyncio.wait_for(wss.connection_closed.wait(), timeout=5)
        assert not proxy_client.is_connected()

        # tab 2 connects -> brand new MCP session, calls work again
        stop2 = asyncio.Event()
        tab2 = asyncio.create_task(fake_colab_tab(wss.port, wss.token, "tab-2", stop2))
        assert await proxy_client.await_proxy_connection(5)
        assert await call_whoami(proxy_client) == "tab-2"

        # tab 3 opens while tab 2 is still connected -> newest wins
        stop3 = asyncio.Event()
        tab3 = asyncio.create_task(fake_colab_tab(wss.port, wss.token, "tab-3", stop3))
        for _ in range(100):
            if wss.connection_generation == 3 and proxy_client.is_connected():
                break
            await asyncio.sleep(0.05)
        assert await call_whoami(proxy_client) == "tab-3"

        stop2.set()
        stop3.set()
        await asyncio.gather(tab2, tab3, return_exceptions=True)
    finally:
        await proxy.cleanup()
