import socket

import pytest

from orchest.network import transport
from orchest.network.protocol import MessageType


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def no_certs(tmp_path, monkeypatch):
    monkeypatch.setattr(transport, "CERTS_DIR", str(tmp_path))
    monkeypatch.delenv(transport.ALLOW_INSECURE_ENV, raising=False)
    return tmp_path


def test_raw_tcp_server_fails_closed_without_certs(no_certs):
    server = transport.RawTCPTransportServer(host="127.0.0.1", port=_free_port())
    with pytest.raises(transport.InsecureTransportError):
        server.start()
    assert server._server_sock is None


def test_raw_tcp_client_refuses_plaintext_without_certs(no_certs):
    raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(transport.InsecureTransportError):
            transport._wrap_client_socket(raw)
    finally:
        raw.close()

    ok, meta, tensor = transport.RawTCPTransportClient.send_and_receive_tensor(
        "127.0.0.1", _free_port(), MessageType.FORWARD_TENSOR, {"stage_index": 0}, timeout=1.0,
    )
    assert not ok and tensor is None
    assert "insecure" in meta["error"].lower()


def test_grpc_client_refuses_plaintext_without_certs(no_certs):
    pytest.importorskip("grpc")
    with pytest.raises(transport.InsecureTransportError):
        transport._read_tls_material("worker", "gRPC client")


def test_explicit_opt_in_allows_insecure_demo_mode(no_certs, monkeypatch):
    monkeypatch.setenv(transport.ALLOW_INSECURE_ENV, "1")
    assert transport._build_tls_context("host", transport.ssl.Purpose.CLIENT_AUTH, "test") is None

    server = transport.RawTCPTransportServer(host="127.0.0.1", port=_free_port())
    server.start()
    try:
        assert server.ssl_context is None
    finally:
        server.stop()


def test_corrupt_certificate_is_rejected(no_certs):
    for name in ("ca.crt", "worker.crt", "worker.key"):
        (no_certs / name).write_text("not a certificate")
    with pytest.raises(transport.InsecureTransportError):
        transport._build_tls_context("worker", transport.ssl.Purpose.SERVER_AUTH, "test")
