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
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
import asyncio
import logging
import mcp.types as types
from mcp.shared.message import SessionMessage
from pydantic_core import ValidationError
import secrets
import websockets
from websockets.asyncio.server import ServerConnection
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response
from websockets.typing import Subprotocol


COLAB = "https://colab.research.google.com"
COLAB_ALT_DOMAIN = "https://colab.google.com"
SCRATCH_PATH = "/notebooks/empty.ipynb"

# Close code/reason sent to an older tab when a newer Colab tab takes over.
REPLACED_CLOSE_CODE = 1000
REPLACED_CLOSE_REASON = "Replaced by a newer Colab connection"


class ColabWebSocketServer:
    """
    A WebSocket server designed to serve a single connection at a time
    from a Google Colab session (colab.google.com).

    Every accepted connection gets a fresh pair of memory streams, so the MCP
    client session built on top of it can be torn down and rebuilt whenever a
    tab disconnects or a newer tab takes over. A newer authenticated connection
    always replaces the current one, so a stale tab can never block reconnects.
    """

    def __init__(self, host="localhost", port: int = 0, token: str | None = None):
        self.host = host
        # port we'd like to bind (e.g. persisted from a previous run); 0 = any
        self.preferred_port = port
        self.port = 0
        self.connection_lock = asyncio.Lock()
        self.connection_live = asyncio.Event()
        # incremented for every accepted connection
        self.connection_generation = 0
        # replaced per connection; set once that connection has fully closed
        self.connection_closed = asyncio.Event()
        self.connection_closed.set()
        self.allowed_origins = [COLAB, COLAB_ALT_DOMAIN]
        self._server: websockets.Server | None = None
        self._active_ws: ServerConnection | None = None

        self.read_stream: MemoryObjectReceiveStream[SessionMessage | Exception]
        self._read_stream_writer: MemoryObjectSendStream[SessionMessage | Exception]
        self.write_stream: MemoryObjectSendStream[SessionMessage]
        self._write_stream_reader: MemoryObjectReceiveStream[SessionMessage]
        self._new_streams()
        self.token = token or secrets.token_urlsafe(16)

    def _new_streams(self):
        self._read_stream_writer, self.read_stream = anyio.create_memory_object_stream(
            0
        )
        self.write_stream, self._write_stream_reader = (
            anyio.create_memory_object_stream(0)
        )

    async def _read_from_socket(self, websocket):
        """Listens to the socket and puts messages into the read stream."""
        try:
            async for msg in websocket:
                try:
                    client_message = types.JSONRPCMessage.model_validate_json(msg)
                except ValidationError as exc:
                    await self._read_stream_writer.send(exc)
                    continue
                await self._read_stream_writer.send(SessionMessage(client_message))
        except ConnectionClosed as e:
            logging.info(f"Connection closed: {e}")
        except (anyio.ClosedResourceError, anyio.BrokenResourceError):
            # the MCP client session for this connection has gone away
            pass

    async def _write_to_socket(self, websocket):
        """Reads from the write stream and sends over the socket."""
        try:
            while True:
                # Wait for a message from the application
                msg = await self._write_stream_reader.receive()

                try:
                    json_obj = msg.message.model_dump_json(
                        by_alias=True, exclude_none=True
                    )
                    await websocket.send(json_obj)
                except ConnectionClosed:
                    break
        except (anyio.ClosedResourceError, anyio.EndOfStream):
            # server closed write stream
            pass

    def _validate_authorization(self, websocket: ServerConnection, request: Request):
        if request.path.find(f"access_token={self.token}") != -1:
            return None
        try:
            headers: Headers = request.headers
            auth_header = headers.get("Authorization")
            if not auth_header:
                return Response(401, "Missing authorization", Headers([]))
            scheme, token = auth_header.split(None, 1)
            if scheme.lower() != "bearer":
                return Response(400, "Invalid authorization header", Headers([]))
        except ValueError:
            return Response(400, "Invalid header format", Headers([]))
        if token == self.token:
            return None
        return Response(403, "Bad authorization token", Headers([]))

    async def close_active_connection(self, reason: str = "Closed by colab-mcp"):
        """Closes the currently connected tab, if any."""
        if self._active_ws is not None:
            await self._active_ws.close(code=REPLACED_CLOSE_CODE, reason=reason)

    async def _connection_handler(self, websocket: ServerConnection):
        """
        Handles incoming (already authenticated, origin-checked) connections.
        The newest connection wins: an existing connection is closed first.
        """
        if self.connection_lock.locked():
            logging.info(
                f"New Colab connection from {websocket.remote_address}; replacing the existing one"
            )
            await self.close_active_connection(REPLACED_CLOSE_REASON)

        async with self.connection_lock:
            self._active_ws = websocket
            self._new_streams()
            self.connection_generation += 1
            closed = asyncio.Event()
            self.connection_closed = closed
            self.connection_live.set()
            logging.info(
                f"Colab connection #{self.connection_generation} established from {websocket.remote_address}"
            )
            reading_task = asyncio.create_task(self._read_from_socket(websocket))
            writing_task = asyncio.create_task(self._write_to_socket(websocket))
            try:
                _, pending = await asyncio.wait(
                    [reading_task, writing_task], return_when=asyncio.FIRST_COMPLETED
                )
                for task in pending:
                    task.cancel()
            except Exception as e:
                logging.error(f"Unexpected error: {e}")
                reading_task.cancel()
                writing_task.cancel()
            finally:
                self.connection_live.clear()
                if self._active_ws is websocket:
                    self._active_ws = None
                # Ending the read stream makes the MCP client session fail any
                # in-flight requests with "Connection closed" instead of hanging.
                self._read_stream_writer.close()
                self._write_stream_reader.close()
                closed.set()
                logging.info("Colab connection closed")

    async def _serve(self, port: int):
        return await websockets.serve(
            self._connection_handler,
            host=self.host,
            port=port,
            subprotocols=[Subprotocol("mcp")],
            origins=self.allowed_origins,
            process_request=self._validate_authorization,
        )

    async def __aenter__(self):
        try:
            self._server = await self._serve(self.preferred_port)
        except OSError as e:
            if not self.preferred_port:
                raise
            logging.warning(
                f"Port {self.preferred_port} unavailable ({e}); falling back to a random port"
            )
            self._server = await self._serve(0)
        self.port = self._server.sockets[0].getsockname()[1]
        logging.info(f"Starting WebSocket server on ws://{self.host}:{self.port}")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        logging.info("Closing WebSocket server")
        if self._server:
            self._server.close()
            self.write_stream.close()
            self.read_stream.close()
            await self._server.wait_closed()
