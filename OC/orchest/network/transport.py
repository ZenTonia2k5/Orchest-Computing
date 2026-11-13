import logging
import socket
import struct
import threading
import time
import os
import ssl
from typing import Callable, Optional, Dict, Any, Tuple
from concurrent import futures

import torch
from .protocol import TensorPacket, MessageType

# Resolve certificates directory path dynamically
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CERTS_DIR = os.path.join(BASE_DIR, "certs")

def read_zmq_key(filename: str) -> bytes:
    path = os.path.join(CERTS_DIR, filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read().strip()
    return b""


# ==============================================================================
# 0. Transport security policy (fail closed by default)
# ==============================================================================

logger = logging.getLogger("OrchestTransport")

ALLOW_INSECURE_ENV = "ORCHEST_ALLOW_INSECURE_TRANSPORT"


class InsecureTransportError(RuntimeError):
    """Raised when secure transport cannot be established and insecure fallback is not allowed."""


def insecure_transport_allowed() -> bool:
    """
    Insecure (plaintext) transport is only permitted when explicitly opted in via
    ORCHEST_ALLOW_INSECURE_TRANSPORT=1, e.g. for local demos. Research runs that claim
    secure execution must leave it unset so a missing or broken certificate setup fails
    instead of silently downgrading to plaintext.
    """
    return os.environ.get(ALLOW_INSECURE_ENV, "0").strip().lower() in ("1", "true", "yes")


def _handle_missing_security(component: str, detail: str) -> None:
    if insecure_transport_allowed():
        logger.warning(f"{component}: {detail}. Continuing WITHOUT encryption ({ALLOW_INSECURE_ENV}=1).")
        return
    raise InsecureTransportError(
        f"{component}: {detail}. Refusing to fall back to insecure transport "
        f"(set {ALLOW_INSECURE_ENV}=1 only for local demos)."
    )


def _cert_paths(role: str) -> Tuple[str, str, str]:
    return (
        os.path.join(CERTS_DIR, "ca.crt"),
        os.path.join(CERTS_DIR, f"{role}.crt"),
        os.path.join(CERTS_DIR, f"{role}.key"),
    )


def _missing_cert_files(role: str) -> list:
    return [os.path.basename(p) for p in _cert_paths(role) if not os.path.exists(p)]


def _build_tls_context(role: str, purpose: ssl.Purpose, component: str) -> Optional[ssl.SSLContext]:
    """Returns an mTLS context, None when insecure mode is allowed and certs are unusable, or raises."""
    missing = _missing_cert_files(role)
    if missing:
        _handle_missing_security(component, f"missing certificate files {', '.join(missing)}")
        return None
    ca_path, crt_path, key_path = _cert_paths(role)
    try:
        context = ssl.create_default_context(purpose)
        context.load_cert_chain(certfile=crt_path, keyfile=key_path)
        context.load_verify_locations(cafile=ca_path)
        context.check_hostname = False
        context.verify_mode = ssl.CERT_REQUIRED
        return context
    except Exception as e:
        _handle_missing_security(component, f"failed to load mTLS context ({e})")
        return None


def _wrap_client_socket(s: socket.socket) -> socket.socket:
    context = _build_tls_context("worker", ssl.Purpose.SERVER_AUTH, "RawTCP client")
    if context is None:
        return s
    return context.wrap_socket(s, server_hostname="host_master")


def _read_tls_material(role: str, component: str) -> Optional[Tuple[bytes, bytes, bytes]]:
    """Returns (ca, cert, key) bytes, None when insecure mode is allowed and certs are unusable, or raises."""
    missing = _missing_cert_files(role)
    if missing:
        _handle_missing_security(component, f"missing certificate files {', '.join(missing)}")
        return None
    ca_path, crt_path, key_path = _cert_paths(role)
    try:
        with open(ca_path, "rb") as f:
            ca = f.read()
        with open(crt_path, "rb") as f:
            crt = f.read()
        with open(key_path, "rb") as f:
            key = f.read()
        return ca, crt, key
    except Exception as e:
        _handle_missing_security(component, f"failed to read mTLS material ({e})")
        return None

# ==============================================================================
# 1. Raw TCP Sockets Transport (Default) with mTLS
# ==============================================================================

class RawTCPTransportServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 5555, secret_key: Optional[bytes] = None):
        self.host = host
        self.port = port
        self.secret_key = secret_key
        self._server_sock: Optional[socket.socket] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.on_tensor_received_cb: Optional[
            Callable[[MessageType, Dict[str, Any], Optional[torch.Tensor]], None]
        ] = None
        self.ssl_context: Optional[ssl.SSLContext] = None

    def start(self):
        if self._running and self._server_sock is not None:
            return
        self._running = True
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):
            try:
                self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except Exception:
                pass
        self._server_sock.bind((self.host, self.port))
        self._server_sock.listen(128)

        # Configure mTLS SSL Context (fails closed unless insecure mode is explicitly allowed)
        try:
            self.ssl_context = _build_tls_context("host", ssl.Purpose.CLIENT_AUTH, "RawTCP server")
        except InsecureTransportError:
            self._running = False
            self._server_sock.close()
            self._server_sock = None
            raise

        def _accept_loop():
            while self._running:
                try:
                    self._server_sock.settimeout(1.0)
                    try:
                        conn, addr = self._server_sock.accept()
                        if self.ssl_context:
                            conn = self.ssl_context.wrap_socket(conn, server_side=True)
                    except (socket.timeout, ssl.SSLError):
                        continue
                    client_thread = threading.Thread(
                        target=self._handle_client, args=(conn, addr), daemon=True
                    )
                    client_thread.start()
                except Exception:
                    break

        self._thread = threading.Thread(target=_accept_loop, daemon=True)
        self._thread.start()

    def _recv_exact(self, conn: socket.socket, num_bytes: int) -> bytes:
        data = bytearray()
        while len(data) < num_bytes:
            try:
                chunk = conn.recv(min(num_bytes - len(data), 65536))
                if not chunk:
                    break
                data.extend(chunk)
            except Exception:
                break
        return bytes(data)

    def _handle_client(self, conn: socket.socket, addr: Tuple[str, int]):
        try:
            conn.settimeout(10.0)
            header_head = self._recv_exact(conn, TensorPacket.HEADER_SIZE)
            if len(header_head) < TensorPacket.HEADER_SIZE:
                return

            magic, msg_type_code, header_len, tensor_len, body_crc32 = struct.unpack(
                TensorPacket.HEADER_STRUCT, header_head
            )
            if magic != TensorPacket.MAGIC:
                return

            total_body_len = header_len + tensor_len
            raw_body = self._recv_exact(conn, total_body_len)
            if len(raw_body) < total_body_len:
                return

            raw_full = header_head + raw_body
            msg_type, meta, tensor = TensorPacket.unpack(raw_full, secret_key=self.secret_key)

            resp_tensor = None
            resp_meta = {"status": "OK"}
            if self.on_tensor_received_cb:
                cb_out = self.on_tensor_received_cb(msg_type, meta, tensor)
                if isinstance(cb_out, torch.Tensor):
                    resp_tensor = cb_out
                elif isinstance(cb_out, tuple) and len(cb_out) >= 2:
                    if isinstance(cb_out[0], dict):
                        resp_meta.update(cb_out[0])
                    resp_tensor = cb_out[1]

            if resp_tensor is not None:
                resp_pkt = TensorPacket.pack(
                    MessageType.FORWARD_TENSOR, resp_meta, tensor=resp_tensor, secret_key=self.secret_key
                )
            else:
                resp_pkt = TensorPacket.pack(MessageType.ACK, resp_meta)
            conn.sendall(resp_pkt)
        except Exception:
            pass
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def stop(self):
        self._running = False
        if self._server_sock:
            try:
                self._server_sock.close()
            except Exception:
                pass
            self._server_sock = None
        if self._thread and self._thread.is_alive():
            try:
                self._thread.join(timeout=1.0)
            except Exception:
                pass
            self._thread = None


class RawTCPTransportClient:
    @staticmethod
    def send_packet(
        target_ip: str,
        target_port: int,
        msg_type: MessageType,
        meta: Dict[str, Any],
        tensor: Optional[torch.Tensor] = None,
        secret_key: Optional[bytes] = None,
        timeout: float = 5.0,
    ) -> bool:
        packet_bytes = TensorPacket.pack(msg_type, meta, tensor, secret_key=secret_key)
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(timeout)

            # Wrap client socket with mTLS (raises InsecureTransportError instead of downgrading)
            s = _wrap_client_socket(s)

            s.connect((target_ip, target_port))
            s.sendall(packet_bytes)

            ack = s.recv(1024)
            s.close()
            return len(ack) > 0
        except Exception:
            return False

    @classmethod
    def send_and_receive_tensor(
        cls,
        target_ip: str,
        target_port: int,
        msg_type: MessageType,
        meta: Dict[str, Any],
        tensor: Optional[torch.Tensor] = None,
        secret_key: Optional[bytes] = None,
        timeout: float = 45.0,
        device: str = "cpu",
    ) -> Tuple[bool, Dict[str, Any], Optional[torch.Tensor]]:
        packet_bytes = TensorPacket.pack(msg_type, meta, tensor, secret_key=secret_key)
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(timeout)

            # Wrap client socket with mTLS (raises InsecureTransportError instead of downgrading)
            s = _wrap_client_socket(s)

            s.connect((target_ip, target_port))
            s.sendall(packet_bytes)

            header_head = cls._recv_exact_sock(s, TensorPacket.HEADER_SIZE)
            if len(header_head) < TensorPacket.HEADER_SIZE:
                s.close()
                return False, {"error": "Incomplete header received"}, None

            magic, msg_type_code, header_len, tensor_len, body_crc32 = struct.unpack(
                TensorPacket.HEADER_STRUCT, header_head
            )
            raw_body = cls._recv_exact_sock(s, header_len + tensor_len)
            s.close()
            if len(raw_body) < header_len + tensor_len:
                return False, {"error": "Incomplete body received"}, None

            raw_full = header_head + raw_body
            res_msg_type, res_meta, res_tensor = TensorPacket.unpack(
                raw_full, secret_key=secret_key, device=device
            )
            ok = (res_tensor is not None) and ("error" not in res_meta)
            return ok, res_meta, res_tensor
        except Exception as e:
            return False, {"error": str(e)}, None

    @staticmethod
    def _recv_exact_sock(s: socket.socket, num_bytes: int) -> bytes:
        data = bytearray()
        while len(data) < num_bytes:
            try:
                chunk = s.recv(min(num_bytes - len(data), 65536))
                if not chunk:
                    break
                data.extend(chunk)
            except Exception:
                break
        return bytes(data)


# ==============================================================================
# 2. ZeroMQ Transport with Curve Security
# ==============================================================================

class ZMQTransportServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 5555, secret_key: Optional[bytes] = None):
        self.host = host
        self.port = port
        self.secret_key = secret_key
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.on_tensor_received_cb: Optional[
            Callable[[MessageType, Dict[str, Any], Optional[torch.Tensor]], None]
        ] = None

    def start(self):
        if self._running and self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        def _loop():
            import zmq
            ctx = zmq.Context.instance()
            sock = ctx.socket(zmq.REP)

            # Configure Curve security on server socket (fails closed unless explicitly allowed)
            secret_key_bytes = read_zmq_key("host_zmq.key")
            public_key_bytes = read_zmq_key("host_zmq.pub")
            if secret_key_bytes and public_key_bytes:
                sock.curve_secretkey = secret_key_bytes
                sock.curve_publickey = public_key_bytes
                sock.curve_server = 1
            else:
                try:
                    _handle_missing_security("ZMQ server", "missing CurveZMQ host keys")
                except InsecureTransportError as e:
                    logger.error(str(e))
                    self._running = False
                    sock.close()
                    return

            sock.bind(f"tcp://{self.host}:{self.port}")
            sock.setsockopt(zmq.RCVTIMEO, 1000)

            while self._running:
                try:
                    raw_full = sock.recv()
                    msg_type, meta, tensor = TensorPacket.unpack(raw_full, secret_key=self.secret_key)
                    if self.on_tensor_received_cb:
                        self.on_tensor_received_cb(msg_type, meta, tensor)
                    ack_pkt = TensorPacket.pack(MessageType.ACK, {"status": "OK"})
                    sock.send(ack_pkt)
                except zmq.Again:
                    continue
                except Exception:
                    try:
                        sock.send(TensorPacket.pack(MessageType.ACK, {"status": "ERROR"}))
                    except Exception:
                        pass
            try:
                sock.close()
            except Exception:
                pass

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            try:
                self._thread.join(timeout=1.5)
            except Exception:
                pass
            self._thread = None


class ZMQTransportClient:
    @staticmethod
    def send_packet(
        target_ip: str,
        target_port: int,
        msg_type: MessageType,
        meta: Dict[str, Any],
        tensor: Optional[torch.Tensor] = None,
        secret_key: Optional[bytes] = None,
        timeout: float = 5.0,
    ) -> bool:
        import zmq
        packet_bytes = TensorPacket.pack(msg_type, meta, tensor, secret_key=secret_key)
        try:
            ctx = zmq.Context.instance()
            sock = ctx.socket(zmq.REQ)

            # Configure Curve security on client socket (fails closed unless explicitly allowed)
            server_public_bytes = read_zmq_key("host_zmq.pub")
            client_secret_bytes = read_zmq_key("worker_zmq.key")
            client_public_bytes = read_zmq_key("worker_zmq.pub")
            if server_public_bytes and client_secret_bytes and client_public_bytes:
                sock.curve_serverkey = server_public_bytes
                sock.curve_secretkey = client_secret_bytes
                sock.curve_publickey = client_public_bytes
            else:
                try:
                    _handle_missing_security("ZMQ client", "missing CurveZMQ keys")
                except InsecureTransportError as e:
                    logger.error(str(e))
                    sock.close()
                    return False

            sock.connect(f"tcp://{target_ip}:{target_port}")
            sock.setsockopt(zmq.SNDTIMEO, int(timeout * 1000))
            sock.setsockopt(zmq.RCVTIMEO, int(timeout * 1000))
            sock.setsockopt(zmq.LINGER, 0)
            
            sock.send(packet_bytes)
            ack = sock.recv()
            sock.close()
            return len(ack) > 0
        except Exception:
            return False


# ==============================================================================
# 3. gRPC Transport with mTLS
# ==============================================================================

class GRPCTensorServicer:
    def __init__(self, on_tensor_received_cb, secret_key):
        self.on_tensor_received_cb = on_tensor_received_cb
        self.secret_key = secret_key

    def SendPacket(self, request, context):
        import grpc
        from orchest.network import tensor_service_pb2 as pb2
        try:
            raw_full = request.data
            msg_type, meta, tensor = TensorPacket.unpack(raw_full, secret_key=self.secret_key)
            if self.on_tensor_received_cb:
                self.on_tensor_received_cb(msg_type, meta, tensor)
            ack_pkt = TensorPacket.pack(MessageType.ACK, {"status": "OK"})
            return pb2.PacketResponse(data=ack_pkt)
        except Exception as e:
            err_pkt = TensorPacket.pack(MessageType.ACK, {"status": "ERROR", "error": str(e)})
            return pb2.PacketResponse(data=err_pkt)


class GRPCTransportServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 5555, secret_key: Optional[bytes] = None):
        self.host = host
        self.port = port
        self.secret_key = secret_key
        self.on_tensor_received_cb: Optional[
            Callable[[MessageType, Dict[str, Any], Optional[torch.Tensor]], None]
        ] = None
        self._server = None

    def start(self):
        if self._server is not None:
            return
        import grpc
        from orchest.network import tensor_service_pb2_grpc as pb2_grpc
        self._server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=16),
            options=[
                ('grpc.max_receive_message_length', 100 * 1024 * 1024),
                ('grpc.max_send_message_length', 100 * 1024 * 1024)
            ]
        )
        servicer = GRPCTensorServicer(self.on_tensor_received_cb, self.secret_key)
        pb2_grpc.add_TensorServiceServicer_to_server(servicer, self._server)

        # Configure gRPC mTLS credentials (fails closed unless insecure mode is explicitly allowed)
        try:
            material = _read_tls_material("host", "gRPC server")
        except InsecureTransportError:
            self._server = None
            raise
        if material is not None:
            ca_cert, host_cert, host_key = material
            credentials = grpc.ssl_server_credentials(
                [(host_key, host_cert)],
                root_certificates=ca_cert,
                require_client_auth=True
            )
            self._server.add_secure_port(f"{self.host}:{self.port}", credentials)
        else:
            self._server.add_insecure_port(f"{self.host}:{self.port}")

        self._server.start()

    def stop(self):
        if self._server:
            try:
                self._server.stop(0.5)
            except Exception:
                pass
            self._server = None


class GRPCTransportClient:
    @staticmethod
    def send_packet(
        target_ip: str,
        target_port: int,
        msg_type: MessageType,
        meta: Dict[str, Any],
        tensor: Optional[torch.Tensor] = None,
        secret_key: Optional[bytes] = None,
        timeout: float = 5.0,
    ) -> bool:
        import grpc
        from orchest.network import tensor_service_pb2 as pb2
        from orchest.network import tensor_service_pb2_grpc as pb2_grpc
        packet_bytes = TensorPacket.pack(msg_type, meta, tensor, secret_key=secret_key)
        try:
            message_options = [
                ('grpc.max_receive_message_length', 100 * 1024 * 1024),
                ('grpc.max_send_message_length', 100 * 1024 * 1024)
            ]
            # Configure gRPC mTLS credentials (raises InsecureTransportError instead of downgrading)
            material = _read_tls_material("worker", "gRPC client")
            if material is not None:
                ca_cert, worker_cert, worker_key = material
                credentials = grpc.ssl_channel_credentials(
                    root_certificates=ca_cert,
                    private_key=worker_key,
                    certificate_chain=worker_cert
                )
                # Override target name to match host cert SAN to prevent hostname mismatch errors
                channel = grpc.secure_channel(
                    f"{target_ip}:{target_port}", credentials,
                    options=[('grpc.ssl_target_name_override', 'host_master')] + message_options,
                )
            else:
                channel = grpc.insecure_channel(f"{target_ip}:{target_port}", options=message_options)

            stub = pb2_grpc.TensorServiceStub(channel)
            req = pb2.PacketRequest(data=packet_bytes)
            resp = stub.SendPacket(req, timeout=timeout)
            channel.close()
            return len(resp.data) > 0
        except Exception:
            return False


# ==============================================================================
# 4. Master Abstract Wrappers
# ==============================================================================

class TensorTransportServer:
    """
    Unified Server wrapper routing connections to raw_tcp, zmq, or grpc backends.
    """
    def __init__(self, host: str = "0.0.0.0", port: int = 5555, secret_key: Optional[bytes] = None, transport_type: str = "raw_tcp"):
        self.transport_type = transport_type
        if transport_type == "zmq":
            self.delegate = ZMQTransportServer(host, port, secret_key)
        elif transport_type == "grpc":
            self.delegate = GRPCTransportServer(host, port, secret_key)
        else:
            self.delegate = RawTCPTransportServer(host, port, secret_key)

    @property
    def on_tensor_received_cb(self):
        return self.delegate.on_tensor_received_cb

    @on_tensor_received_cb.setter
    def on_tensor_received_cb(self, val):
        self.delegate.on_tensor_received_cb = val

    def start(self):
        self.delegate.start()

    def stop(self):
        self.delegate.stop()


class TensorTransportClient:
    """
    Unified Client wrapper dispatching messages using raw_tcp, zmq, or grpc channels.
    """
    @staticmethod
    def send_packet(
        target_ip: str,
        target_port: int,
        msg_type: MessageType,
        meta: Dict[str, Any],
        tensor: Optional[torch.Tensor] = None,
        secret_key: Optional[bytes] = None,
        timeout: float = 5.0,
        transport_type: str = "raw_tcp"
    ) -> bool:
        if transport_type == "zmq":
            return ZMQTransportClient.send_packet(target_ip, target_port, msg_type, meta, tensor, secret_key, timeout)
        elif transport_type == "grpc":
            return GRPCTransportClient.send_packet(target_ip, target_port, msg_type, meta, tensor, secret_key, timeout)
        else:
            return RawTCPTransportClient.send_packet(target_ip, target_port, msg_type, meta, tensor, secret_key, timeout)

    @classmethod
    def send_and_receive_tensor(
        cls,
        target_ip: str,
        target_port: int,
        msg_type: MessageType,
        meta: Dict[str, Any],
        tensor: Optional[torch.Tensor] = None,
        secret_key: Optional[bytes] = None,
        timeout: float = 45.0,
        transport_type: str = "raw_tcp",
        device: str = "cpu",
    ) -> Tuple[bool, Dict[str, Any], Optional[torch.Tensor]]:
        return RawTCPTransportClient.send_and_receive_tensor(
            target_ip=target_ip,
            target_port=target_port,
            msg_type=msg_type,
            meta=meta,
            tensor=tensor,
            secret_key=secret_key,
            timeout=timeout,
            device=device,
        )
