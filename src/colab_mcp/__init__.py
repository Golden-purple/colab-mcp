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

import argparse
import asyncio
import datetime
import logging
from pathlib import Path
import tempfile
import sys

from fastmcp import FastMCP
from fastmcp.utilities import logging as fastmcp_logger

from colab_mcp.connection_state import DEFAULT_STATE_FILE
from colab_mcp.session import (
    EXISTING_TAB_WAIT,
    UI_CONNECTION_TIMEOUT,
    ColabSessionProxy,
    ConnectionOptions,
)


mcp = FastMCP(name="ColabMCP")


def init_logger(logdir):
    log_filename = datetime.datetime.now().strftime(
        f"{logdir}/colab-mcp.%Y-%m-%d_%H-%M-%S.log"
    )
    logging.basicConfig(
        format="%(asctime)s %(levelname)s:%(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        filename=log_filename,
        level=logging.INFO,  # Set the minimum logging level to capture
    )
    fastmcp_logger.get_logger("colab-mcp").info("logging to %s" % log_filename)


def parse_args(v):
    parser = argparse.ArgumentParser(
        description="ColabMCP is an MCP server that lets you interact with Colab."
    )
    parser.add_argument(
        "-l",
        "--log",
        help="if set, use this directory as a location for logfiles (if unset, will log to %s/colab-mcp-logs/)"
        % tempfile.gettempdir(),
        action="store",
        default=tempfile.mkdtemp(prefix="colab-mcp-logs-"),
    )
    parser.add_argument(
        "-p",
        "--enable-proxy",
        help="if set, enable the runtime proxy (enabled by default).",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--authuser",
        help="Google account used when a new Colab notebook is opened: account index "
        "(0, 1, ...) or email address. Defaults to the browser's default account.",
        default=None,
    )
    parser.add_argument(
        "--connect-timeout",
        help="total seconds one connection attempt may take before giving up (default: %(default)s)",
        type=float,
        default=UI_CONNECTION_TIMEOUT,
    )
    parser.add_argument(
        "--existing-tab-wait",
        help="seconds to wait for an already-open Colab tab to (re)connect before opening "
        "a new notebook; 0 opens a new notebook immediately (default: %(default)s)",
        type=float,
        default=EXISTING_TAB_WAIT,
    )
    parser.add_argument(
        "--no-persist",
        help="use a fresh random token/port every run (open tabs can't reconnect by reloading)",
        action="store_true",
    )
    parser.add_argument(
        "--state-file",
        help="where the reusable token/port are stored (default: %(default)s)",
        type=Path,
        default=DEFAULT_STATE_FILE,
    )
    return parser.parse_args(v)


async def main_async():
    args = parse_args(sys.argv[1:])
    init_logger(args.log)

    if args.enable_proxy:
        logging.info("enabling session proxy tools")
        session_mcp = ColabSessionProxy(
            ConnectionOptions(
                connect_timeout=args.connect_timeout,
                existing_tab_wait=args.existing_tab_wait,
                authuser=args.authuser,
                persist_connection=not args.no_persist,
                state_file=args.state_file,
            )
        )
        await session_mcp.start_proxy_server()
        mcp.mount(session_mcp.proxy_server)
        for middleware in session_mcp.middleware:
            mcp.add_middleware(middleware)

    try:
        await mcp.run_async()

    finally:
        if args.enable_proxy:
            await session_mcp.cleanup()


def main() -> None:
    asyncio.run(main_async())
