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
from colab_mcp import session
from fastmcp.server.middleware import MiddlewareContext
import pytest
from unittest.mock import patch, AsyncMock, Mock


@pytest.fixture(autouse=True)
def mock_webbrowser(monkeypatch):
    import webbrowser

    mock_open = Mock()
    monkeypatch.setattr(webbrowser, "open_new", mock_open)
    return mock_open


@pytest.fixture
def mock_wss():
    """Provides a mock ColabWebSocketServer instance."""
    return MockColabWebSocketServer()


class MockColabWebSocketServer:
    def __init__(self):
        self.connection_live = asyncio.Event()
        self.read_stream = AsyncMock()
        self.write_stream = AsyncMock()
        self.token = "test-token"
        self.port = 1234
        self.connection_generation = 0
        self.connection_closed = asyncio.Event()
        self.close_active_connection = AsyncMock()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture
def mock_proxy_client(mock_wss):
    client = Mock(spec=session.ColabProxyClient)
    client.wss = mock_wss
    client.is_connected.return_value = False
    return client


class TestColabProxyMiddleware:
    @pytest.mark.asyncio
    async def test_connection_live(self, mock_proxy_client):
        """Tests connection state change from disconnected to connected."""
        middleware = session.ColabProxyMiddleware(mock_proxy_client)
        mock_proxy_client.is_connected.return_value = True
        context = Mock(spec=MiddlewareContext)
        context.fastmcp_context.set_state = Mock()
        context.fastmcp_context.send_tool_list_changed = AsyncMock()
        call_next = AsyncMock()

        await middleware.on_message(context, call_next)

        call_next.assert_called_once_with(context)
        context.fastmcp_context.set_state.assert_any_call("fe_connected", True)
        context.fastmcp_context.set_state.assert_any_call("proxy_token", "test-token")
        context.fastmcp_context.set_state.assert_any_call("proxy_port", 1234)
        assert middleware.last_message_connected is True
        context.fastmcp_context.send_tool_list_changed.assert_called_once()

    @pytest.mark.asyncio
    async def test_connection_not_live(self, mock_proxy_client):
        """Tests connection state change from connected to disconnected."""
        mock_proxy_client.is_connected.return_value = True
        middleware = session.ColabProxyMiddleware(mock_proxy_client)
        mock_proxy_client.is_connected.return_value = False
        context = Mock(spec=MiddlewareContext)
        context.fastmcp_context.set_state = Mock()
        context.fastmcp_context.send_tool_list_changed = AsyncMock()
        call_next = AsyncMock()

        await middleware.on_message(context, call_next)

        call_next.assert_called_once_with(context)
        context.fastmcp_context.set_state.assert_any_call("fe_connected", False)
        assert middleware.last_message_connected is False
        context.fastmcp_context.send_tool_list_changed.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_connection_change(self, mock_proxy_client):
        """Tests no connection state change."""
        mock_proxy_client.is_connected.return_value = True
        middleware = session.ColabProxyMiddleware(mock_proxy_client)
        context = Mock(spec=MiddlewareContext)
        context.fastmcp_context.set_state = Mock()
        context.fastmcp_context.send_tool_list_changed = AsyncMock()
        call_next = AsyncMock()

        await middleware.on_message(context, call_next)

        call_next.assert_called_once_with(context)
        context.fastmcp_context.set_state.assert_any_call("fe_connected", True)
        assert middleware.last_message_connected is True
        context.fastmcp_context.send_tool_list_changed.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_call_tool_existing_tab_connects(
        self, mock_proxy_client, mock_webbrowser
    ):
        """An already-open tab connecting during the grace period wins; no new tab is opened."""
        middleware = session.ColabProxyMiddleware(mock_proxy_client)
        context = Mock()
        context.fastmcp_context.report_progress = AsyncMock()
        context.message.name = session.INJECTED_TOOL_NAME
        mock_proxy_client.await_proxy_connection = AsyncMock(return_value=True)
        call_next = AsyncMock()

        result = await middleware.on_call_tool(context, call_next)

        mock_proxy_client.await_proxy_connection.assert_called_once_with(
            session.EXISTING_TAB_WAIT
        )
        mock_webbrowser.assert_not_called()
        assert result.structured_content == {"result": True}

    @pytest.mark.asyncio
    async def test_on_call_tool_opens_new_tab_after_grace(
        self, mock_proxy_client, mock_webbrowser
    ):
        options = session.ConnectionOptions(
            connect_timeout=1.0, existing_tab_wait=0.3, authuser="me@example.com"
        )
        middleware = session.ColabProxyMiddleware(mock_proxy_client, options)
        context = Mock()
        context.fastmcp_context.report_progress = AsyncMock()
        context.message.name = session.INJECTED_TOOL_NAME
        outcomes = iter([False, True])

        async def fake_wait(timeout):
            if timeout == 0.3:
                await asyncio.sleep(timeout)  # nobody connected during the grace period
            return next(outcomes)

        mock_proxy_client.await_proxy_connection = AsyncMock(side_effect=fake_wait)
        call_next = AsyncMock()

        result = await middleware.on_call_tool(context, call_next)

        assert result.structured_content == {"result": True}
        mock_webbrowser.assert_called_once()
        url = mock_webbrowser.call_args.args[0]
        assert "authuser=me@example.com" in url
        assert "#mcpProxyToken=test-token&mcpProxyPort=1234" in url
        # the second wait only gets what is left of the total budget
        second_wait = mock_proxy_client.await_proxy_connection.call_args_list[1].args[0]
        assert 0.5 < second_wait <= 0.7

    @pytest.mark.asyncio
    async def test_on_call_tool_timeout(self, mock_proxy_client, mock_webbrowser):
        middleware = session.ColabProxyMiddleware(mock_proxy_client)
        context = Mock()
        context.fastmcp_context.report_progress = AsyncMock()
        context.message.name = session.INJECTED_TOOL_NAME
        mock_proxy_client.is_connected.return_value = False
        mock_proxy_client.await_proxy_connection = AsyncMock(return_value=False)
        call_next = AsyncMock()

        result = await middleware.on_call_tool(context, call_next)

        assert mock_proxy_client.await_proxy_connection.call_count == 2
        assert result.structured_content == {"result": False}
        text = result.content[0].text
        assert "test-token&1234" in text
        assert session.INJECTED_TOOL_NAME in text

    @pytest.mark.asyncio
    async def test_on_call_tool_zero_existing_wait(
        self, mock_proxy_client, mock_webbrowser
    ):
        options = session.ConnectionOptions(existing_tab_wait=0)
        middleware = session.ColabProxyMiddleware(mock_proxy_client, options)
        context = Mock()
        context.fastmcp_context.report_progress = AsyncMock()
        context.message.name = session.INJECTED_TOOL_NAME
        mock_proxy_client.await_proxy_connection = AsyncMock(return_value=True)

        result = await middleware.on_call_tool(context, AsyncMock())

        mock_webbrowser.assert_called_once()
        mock_proxy_client.await_proxy_connection.assert_called_once()
        assert result.structured_content == {"result": True}


class TestCheckSessionProxyToolFn:
    @pytest.mark.asyncio
    async def test_connected(self):
        ctx = Mock()
        ctx.get_state.side_effect = (
            lambda k: True if k == session.FE_CONNECTED_KEY else None
        )
        assert await session.check_session_proxy_tool_fn(ctx) is True

    @pytest.mark.asyncio
    async def test_disconnected_has_no_side_effects(self, mock_webbrowser):
        ctx = Mock()
        ctx.get_state.return_value = False
        assert await session.check_session_proxy_tool_fn(ctx) is False
        mock_webbrowser.assert_not_called()


def test_build_colab_url():
    assert session.build_colab_url("t", 5) == (
        "https://colab.research.google.com/notebooks/empty.ipynb#mcpProxyToken=t&mcpProxyPort=5"
    )
    assert "?authuser=1#" in session.build_colab_url("t", 5, "1")


class TestColabProxyClient:
    def test_is_connected(self, mock_wss):
        client = session.ColabProxyClient(mock_wss)
        assert client.is_connected() is False
        mock_wss.connection_live.set()
        assert client.is_connected() is False
        client.proxy_mcp_client = Mock()
        client._session_generation = mock_wss.connection_generation
        assert client.is_connected() is True

    def test_client_factory_connection_live(self, mock_wss):
        mock_wss.connection_live.set()
        client = session.ColabProxyClient(mock_wss)
        client.proxy_mcp_client = Mock()
        client._session_generation = mock_wss.connection_generation

        assert client.client_factory() is client.proxy_mcp_client

    def test_client_factory_connection_not_live(self, mock_wss):
        client = session.ColabProxyClient(mock_wss)
        assert client.client_factory() is client.stubbed_mcp_client

    @pytest.mark.asyncio
    async def test_await_proxy_connection_times_out(self, mock_wss):
        client = session.ColabProxyClient(mock_wss)
        assert await client.await_proxy_connection(0.05) is False

    @staticmethod
    def _fake_client_cls(entered: list):
        class FakeClient:
            def __init__(self, transport, **kwargs):
                self.transport = transport
                self.kwargs = kwargs

            async def __aenter__(self):
                entered.append(self)
                return self

            async def __aexit__(self, *exc):
                return False

        return FakeClient

    @pytest.mark.asyncio
    async def test_new_session_per_connection(self, mock_wss):
        """Each websocket connection gets its own initialized client session."""
        entered = []
        with patch("colab_mcp.session.Client", self._fake_client_cls(entered)):
            client = session.ColabProxyClient(mock_wss)
            async with client:
                for generation in (1, 2):
                    mock_wss.connection_generation = generation
                    mock_wss.connection_closed = asyncio.Event()
                    mock_wss.connection_live.set()
                    assert await client.await_proxy_connection(1) is True
                    assert client.proxy_mcp_client is entered[-1]
                    assert (
                        entered[-1].kwargs["init_timeout"]
                        == session.SESSION_INIT_TIMEOUT
                    )
                    # tab goes away
                    mock_wss.connection_live.clear()
                    mock_wss.connection_closed.set()
                    for _ in range(20):
                        await asyncio.sleep(0.01)
                        if client.proxy_mcp_client is None:
                            break
                    assert client.proxy_mcp_client is None
                    assert client.is_connected() is False
        assert len(entered) == 2

    @pytest.mark.asyncio
    async def test_failed_handshake_drops_connection(self, mock_wss):
        class FailingClient:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                raise TimeoutError("no initialize response")

            async def __aexit__(self, *exc):
                return False

        with patch("colab_mcp.session.Client", FailingClient):
            client = session.ColabProxyClient(mock_wss)
            async with client:
                mock_wss.connection_generation = 1
                mock_wss.connection_closed = asyncio.Event()
                mock_wss.connection_live.set()
                for _ in range(20):
                    await asyncio.sleep(0.01)
                    if mock_wss.close_active_connection.await_count:
                        break
                mock_wss.close_active_connection.assert_awaited()
                assert client.is_connected() is False
                mock_wss.connection_live.clear()
                mock_wss.connection_closed.set()


class TestColabTransport:
    @pytest.mark.asyncio
    @patch("colab_mcp.session.ClientSession")
    async def test_connect_session(self, mock_client_session, mock_wss):
        transport = session.ColabTransport(mock_wss)
        mock_client_session.return_value.__aenter__ = AsyncMock()
        async with transport.connect_session(foo="bar") as client_session:
            assert (
                client_session
                == mock_client_session.return_value.__aenter__.return_value
            )

        mock_client_session.assert_called_once_with(
            mock_wss.read_stream, mock_wss.write_stream, foo="bar"
        )


class TestColabSessionProxy:
    @pytest.mark.asyncio
    @patch("colab_mcp.session.ToolInjectionMiddleware")
    @patch("colab_mcp.session.ColabWebSocketServer")
    @patch("colab_mcp.session.ColabProxyClient")
    @patch("colab_mcp.session.ColabProxyMiddleware")
    async def test_start_proxy_server(
        self,
        mock_colab_proxy_middleware,
        mock_colab_proxy_client,
        mock_colab_web_socket_server,
        mock_tool_injection_middleware,
    ):
        mock_colab_web_socket_server.return_value.__aenter__ = AsyncMock()
        mock_colab_proxy_client.return_value.__aenter__ = AsyncMock()
        proxy = session.ColabSessionProxy(
            session.ConnectionOptions(persist_connection=False)
        )
        await proxy.start_proxy_server()
        mock_colab_proxy_client.assert_called_once()
        assert proxy.proxy_server is not None
        mock_colab_proxy_middleware.assert_called_once()
        mock_tool_injection_middleware.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup(self):
        proxy = session.ColabSessionProxy()
        proxy._exit_stack = AsyncMock()
        await proxy.cleanup()
        proxy._exit_stack.aclose.assert_called_once()
