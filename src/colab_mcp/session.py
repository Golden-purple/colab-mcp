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

import asyncio
from collections.abc import AsyncIterator
import contextlib
from contextlib import AsyncExitStack
import dataclasses
import logging
from pathlib import Path
from urllib.parse import quote
from fastmcp import FastMCP, Client
from fastmcp.client.transports import ClientTransport
from fastmcp.dependencies import CurrentContext
from fastmcp.server.context import Context
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.server.middleware.tool_injection import ToolInjectionMiddleware
from fastmcp.server.proxy import FastMCPProxy
from fastmcp.tools.tool import Tool, ToolResult
from mcp.client.session import ClientSession
from mcp.types import TextContent
import webbrowser

from colab_mcp.connection_state import (
    DEFAULT_STATE_FILE,
    ConnectionState,
    load_state,
    save_state,
)
from colab_mcp.websocket_server import ColabWebSocketServer, COLAB, SCRATCH_PATH

# Total time budget (secs) for one open_colab_browser_connection call.
UI_CONNECTION_TIMEOUT = 120.0
# How long (secs) to wait for an already-open Colab tab before opening a new one.
EXISTING_TAB_WAIT = 30.0
# How long (secs) a connected tab gets to answer the MCP initialize handshake.
SESSION_INIT_TIMEOUT = 30.0

FE_CONNECTED_KEY = "fe_connected"
PROXY_TOKEN_KEY = "proxy_token"
PROXY_PORT_KEY = "proxy_port"
INJECTED_TOOL_NAME = "open_colab_browser_connection"
CONNECT_COMMAND_NAME = "Connect to a local Colab MCP server"


@dataclasses.dataclass
class ConnectionOptions:
    # total budget for one connection attempt
    connect_timeout: float = UI_CONNECTION_TIMEOUT
    # part of that budget spent waiting for an already-open tab; 0 = open a new tab immediately
    existing_tab_wait: float = EXISTING_TAB_WAIT
    # Google account to open new notebooks with: index ("1") or email address
    authuser: str | None = None
    # reuse token/port across restarts so open tabs can reconnect by reloading
    persist_connection: bool = True
    state_file: Path = DEFAULT_STATE_FILE


def build_colab_url(token: str, port: int, authuser: str | None = None) -> str:
    query = f"?authuser={quote(authuser, safe='@')}" if authuser else ""
    return f"{COLAB}{SCRATCH_PATH}{query}#mcpProxyToken={token}&mcpProxyPort={port}"


class ColabTransport(ClientTransport):
    def __init__(self, wss: ColabWebSocketServer):
        self.wss = wss

    @contextlib.asynccontextmanager
    async def connect_session(self, **session_kwargs) -> AsyncIterator[ClientSession]:
        # always bind to the streams of the *current* websocket connection
        async with ClientSession(
            self.wss.read_stream, self.wss.write_stream, **session_kwargs
        ) as session:
            yield session

    def __repr__(self) -> str:
        return "<ColabSessionProxyTransport>"


class ColabProxyClient:
    """
    Keeps an MCP client session in sync with the websocket connection.

    A fresh client (and MCP initialize handshake) is created for every Colab
    tab that connects, and torn down when that tab goes away, so reconnecting
    or switching tabs never leaves the proxy talking to a stale session.
    """

    def __init__(self, wss: ColabWebSocketServer):
        self.wss = wss
        self.stubbed_mcp_client = Client(FastMCP())
        self.proxy_mcp_client: Client | None = None
        self._ready = asyncio.Event()
        self._session_generation: int | None = None
        self._supervisor_task: asyncio.Task | None = None

    def is_connected(self):
        return (
            self.wss.connection_live.is_set()
            and self.proxy_mcp_client is not None
            # the session must belong to the *current* tab, not one being torn down
            and self._session_generation == self.wss.connection_generation
        )

    async def await_proxy_connection(self, timeout: float = UI_CONNECTION_TIMEOUT):
        """Waits up to `timeout` secs for a fully initialized session. Returns whether connected."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while not self.is_connected():
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            if self._ready.is_set():
                # a previous tab's session is still shutting down
                await asyncio.sleep(min(0.05, remaining))
                continue
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._ready.wait(), timeout=remaining)
        return self.is_connected()

    def client_factory(self):
        if self.is_connected():
            return self.proxy_mcp_client
        # return a client mapped to a stubbed mcp server if there is no session proxy
        return self.stubbed_mcp_client

    async def _serve_one_connection(self):
        await self.wss.connection_live.wait()
        generation = self.wss.connection_generation
        closed = self.wss.connection_closed
        try:
            async with Client(
                ColabTransport(self.wss), init_timeout=SESSION_INIT_TIMEOUT
            ) as client:
                if self.wss.connection_generation == generation and not closed.is_set():
                    self.proxy_mcp_client = client
                    self._session_generation = generation
                    self._ready.set()
                    logging.info(f"Colab session #{generation} ready")
                    try:
                        await closed.wait()
                    finally:
                        # mark disconnected *before* the (possibly slow) client teardown
                        self._mark_disconnected()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # e.g. a stale tab that never answers the initialize handshake
            logging.warning(f"Colab session #{generation} failed: {e!r}")
            if self.wss.connection_generation == generation:
                await self.wss.close_active_connection("MCP handshake failed")
        finally:
            self._mark_disconnected()
        await closed.wait()

    def _mark_disconnected(self):
        self.proxy_mcp_client = None
        self._session_generation = None
        self._ready.clear()

    async def _supervise(self):
        while True:
            try:
                await self._serve_one_connection()
            except asyncio.CancelledError:
                raise
            except Exception:
                logging.exception("Unexpected error in Colab session supervisor")
                await asyncio.sleep(1)

    async def __aenter__(self):
        self._supervisor_task = asyncio.create_task(self._supervise())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._supervisor_task:
            self._supervisor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._supervisor_task


def _bool_result(value: bool, message: str) -> ToolResult:
    return ToolResult(
        content=[TextContent(type="text", text=f"{str(value).lower()}: {message}")],
        structured_content={"result": value},
    )


class ColabProxyMiddleware(Middleware):
    def __init__(
        self, proxy_client: ColabProxyClient, options: ConnectionOptions | None = None
    ):
        self.proxy_client = proxy_client
        self.options = options or ConnectionOptions()
        self.last_message_connected = self.proxy_client.is_connected()

    async def on_message(self, context: MiddlewareContext, call_next):
        """
        Check for a change to Colab session connectivity on any communication with this MCP server and
        notify the client when the connectivity status has changed.
        """
        context.fastmcp_context.set_state(
            FE_CONNECTED_KEY, self.proxy_client.is_connected()
        )
        context.fastmcp_context.set_state(PROXY_TOKEN_KEY, self.proxy_client.wss.token)
        context.fastmcp_context.set_state(PROXY_PORT_KEY, self.proxy_client.wss.port)

        result = await call_next(context)

        connected = self.proxy_client.is_connected()
        connection_state_changed = connected != self.last_message_connected
        self.last_message_connected = connected
        if connection_state_changed:
            await context.fastmcp_context.send_tool_list_changed()

        return result

    def _manual_connect_hint(self) -> str:
        wss = self.proxy_client.wss
        connect_string = ConnectionState(wss.token, wss.port).connect_string
        return (
            "To use a notebook that is already open: reload a Colab tab that was connected "
            f"before, or in any open notebook run '{CONNECT_COMMAND_NAME}' from the command "
            f"palette (Ctrl+Shift+P) and enter: {connect_string}"
        )

    async def on_call_tool(self, context, call_next):
        result = await call_next(context)
        if context.message.name != INJECTED_TOOL_NAME:
            return result
        if self.proxy_client.is_connected():
            return result
        return await self._connect(context.fastmcp_context)

    async def _connect(self, fctx) -> ToolResult:
        """
        1. give an already-open Colab tab a chance to (re)connect,
        2. otherwise open a new notebook in the default browser,
        all within one overall time budget, so a call can never hang forever.
        """
        opts = self.options
        loop = asyncio.get_running_loop()
        deadline = loop.time() + opts.connect_timeout
        hint = self._manual_connect_hint()

        existing_wait = max(0.0, min(opts.existing_tab_wait, opts.connect_timeout))
        if existing_wait > 0:
            await fctx.report_progress(
                progress=1,
                total=3,
                message=f"Waiting {existing_wait:.0f}s for an already-open Colab tab. {hint}",
            )
            if await self.proxy_client.await_proxy_connection(existing_wait):
                await fctx.report_progress(
                    progress=3, total=3, message="Connected to an existing Colab tab!"
                )
                return _bool_result(True, "connected to an already-open Colab notebook")

        remaining = deadline - loop.time()
        if remaining > 0:
            wss = self.proxy_client.wss
            webbrowser.open_new(build_colab_url(wss.token, wss.port, opts.authuser))
            await fctx.report_progress(
                progress=2,
                total=3,
                message=f"No existing tab connected; opened a new Colab notebook - waiting {remaining:.0f}s",
            )
            if await self.proxy_client.await_proxy_connection(remaining):
                await fctx.report_progress(
                    progress=3,
                    total=3,
                    message="The Colab UI is successfully connected!",
                )
                return _bool_result(True, "connected to a new Colab notebook")

        await fctx.report_progress(
            progress=3,
            total=3,
            message="Timeout while waiting for the user to connect.",
        )
        return _bool_result(
            False,
            f"no Colab tab connected within {opts.connect_timeout:.0f}s. {hint}. "
            f"Then call {INJECTED_TOOL_NAME} again to retry.",
        )


async def check_session_proxy_tool_fn(ctx: Context = CurrentContext()) -> bool:
    # The actual waiting / browser opening happens in ColabProxyMiddleware.on_call_tool,
    # which has access to progress reporting and the connection options.
    return bool(ctx.get_state(FE_CONNECTED_KEY))


check_session_proxy_tool = Tool.from_function(
    fn=check_session_proxy_tool_fn,
    name=INJECTED_TOOL_NAME,
    description=(
        "Opens a connection to a Google Colab browser session and unlocks notebook editing tools. "
        "First waits for an already-open Colab tab to connect (reloading a previously connected tab is enough), "
        "then opens a new notebook if none connects. Gives up after a bounded timeout and can simply be called again to retry. "
        "Returns a boolean representing whether the connection attempt succeeded"
    ),
)


class ColabSessionProxy:
    def __init__(self, options: ConnectionOptions | None = None):
        self.options = options or ConnectionOptions()
        self._exit_stack = AsyncExitStack()
        self.proxy_server: FastMCPProxy | None = None
        # list order matters, see: https://gofastmcp.com/servers/middleware#multiple-middleware
        self.middleware: list[Middleware] = []
        self.wss: ColabWebSocketServer | None = None

    async def _start_websocket_server(self) -> ColabWebSocketServer:
        opts = self.options
        saved = load_state(opts.state_file) if opts.persist_connection else None
        wss = await self._exit_stack.enter_async_context(
            ColabWebSocketServer(
                port=saved.port if saved else 0,
                token=saved.token if saved else None,
            )
        )
        if opts.persist_connection:
            if saved is None or saved.port == wss.port:
                save_state(ConnectionState(wss.token, wss.port), opts.state_file)
            else:
                # another colab-mcp instance owns the saved port; leave its state alone
                logging.info(
                    f"Saved port {saved.port} is in use by another instance; using {wss.port} for this session"
                )
        return wss

    async def start_proxy_server(self):
        self.wss = await self._start_websocket_server()
        proxy_client = await self._exit_stack.enter_async_context(
            ColabProxyClient(self.wss)
        )
        self.proxy_server = FastMCPProxy(
            client_factory=proxy_client.client_factory,
            instructions="Connects to a user's Google Colab session in a browser and allows for interactions with their Google Colab notebook",
        )
        # ColabProxyMiddleware must be first because it sets the fe_connected state
        self.middleware.append(ColabProxyMiddleware(proxy_client, self.options))
        self.middleware.append(
            ToolInjectionMiddleware(tools=[check_session_proxy_tool])
        )

    async def cleanup(self):
        await self._exit_stack.aclose()
