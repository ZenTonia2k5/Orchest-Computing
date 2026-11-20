"""
Persistent raw-TCP connections: one TCP + TLS session per peer, reused across messages,
recovered transparently after the peer restarts, never retried after a timeout.
"""

import socket
import threading
import time

import pytest
import torch

from orchest.network import transport
from orchest.network.protocol import MessageType
from orchest.network.transport import RawTCPTransportServer, TensorTransportClient


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _server(port, handler):
    srv = RawTCPTransportServer(host="127.0.0.1", port=port)
    srv.on_tensor_received_cb = handler
    srv.start()
    time.sleep(0.1)
    return srv


def _echo(msg_type, meta, tensor):
    return ({"echo": meta.get("i")}, tensor if tensor is not None else torch.zeros(1))


def _call(port, i, timeout=5.0):
    return TensorTransportClient.send_and_receive_tensor(
        "127.0.0.1", port, MessageType.FORWARD_TENSOR, {"i": i}, tensor=torch.full((1, 4), float(i)), timeout=timeout,
    )


@pytest.fixture(autouse=True)
def _clean_pool():
    TensorTransportClient.close_all_connections()
    yield
    TensorTransportClient.close_all_connections()


def test_messages_reuse_one_connection():
    port = _free_port()
    srv = _server(port, _echo)
    try:
        before = transport._POOL.opened
        for i in range(20):
            ok, meta, out = _call(port, i)
            assert ok and meta["echo"] == i and out[0, 0].item() == i
        assert transport._POOL.opened - before == 1
        assert len(srv._conns) == 1
    finally:
        srv.stop()


def test_send_packet_uses_framed_ack_on_the_same_connection():
    port = _free_port()
    seen = []
    srv = _server(port, lambda t, m, x: seen.append(t) or None)
    try:
        before = transport._POOL.opened
        for _ in range(5):
            assert TensorTransportClient.send_packet("127.0.0.1", port, MessageType.RELEASE_SESSION, {"session_id": "s"})
        ok, _, _ = _call(port, 1)  # a following tensor request still parses cleanly
        assert transport._POOL.opened - before == 1
        assert seen.count(MessageType.RELEASE_SESSION) == 5
    finally:
        srv.stop()


def test_stale_pooled_connection_is_replaced_after_restart():
    port = _free_port()
    srv = _server(port, _echo)
    assert _call(port, 1)[0]
    srv.stop()
    time.sleep(0.2)
    srv = _server(port, _echo)  # same address, new process state: the pooled socket is dead
    try:
        ok, meta, _ = _call(port, 2)
        assert ok and meta["echo"] == 2
    finally:
        srv.stop()


def test_handler_error_returns_error_and_keeps_connection():
    port = _free_port()

    def handler(t, meta, x):
        if meta.get("i") == 1:
            raise ValueError("boom")
        return _echo(t, meta, x)

    srv = _server(port, handler)
    try:
        before = transport._POOL.opened
        ok, meta, out = _call(port, 1)
        assert not ok and "boom" in meta["error"] and out is None
        ok, meta, _ = _call(port, 2)
        assert ok and meta["echo"] == 2
        assert transport._POOL.opened - before == 1
    finally:
        srv.stop()


def test_concurrent_requests_get_separate_connections():
    port = _free_port()

    def slow(t, meta, x):
        time.sleep(0.2)
        return _echo(t, meta, x)

    srv = _server(port, slow)
    results = {}
    try:
        threads = [threading.Thread(target=lambda i=i: results.__setitem__(i, _call(port, i))) for i in range(4)]
        t0 = time.perf_counter()
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        assert time.perf_counter() - t0 < 0.7  # served in parallel, not one after another
        assert all(results[i][0] and results[i][1]["echo"] == i for i in range(4))
    finally:
        srv.stop()


def test_timeout_is_not_retried():
    port = _free_port()
    calls = []

    def sleepy(t, meta, x):
        calls.append(meta.get("i"))
        time.sleep(1.0)
        return _echo(t, meta, x)

    srv = _server(port, sleepy)
    try:
        assert _call(port, 1)[0]  # warms a pooled connection
        calls.clear()
        ok, meta, _ = _call(port, 2, timeout=0.3)
        assert not ok and "timed out" in meta["error"].lower()
        time.sleep(1.2)
        assert calls == [2]  # sent exactly once
    finally:
        srv.stop()
