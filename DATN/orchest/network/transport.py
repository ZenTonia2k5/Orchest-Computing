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

        # Configure mTLS SSL Context
        try:
            ca_path = os.path.join(CERTS_DIR, "ca.crt")
            host_crt_path = os.path.join(CERTS_DIR, "host.crt")
            host_key_path = os.path.join(CERTS_DIR, "host.key")
            if os.path.exists(ca_path) and os.path.exists(host_crt_path):
                context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
                context.load_cert_chain(certfile=host_crt_path, keyfile=host_key_path)
                context.load_verify_locations(cafile=ca_path)
                context.verify_mode = ssl.CERT_REQUIRED
                context.check_hostname = False
                self.ssl_context = context
        except Exception:
            self.ssl_context = None

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

            if self.on_tensor_received_cb:
                self.on_tensor_received_cb(msg_type, meta, tensor)

            ack_pkt = TensorPacket.pack(MessageType.ACK, {"status": "OK"})
            conn.sendall(ack_pkt)
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

            # Wrap client socket with mTLS Context
            try:
                ca_path = os.path.join(CERTS_DIR, "ca.crt")
                worker_crt_path = os.path.join(CERTS_DIR, "worker.crt")
                worker_key_path = os.path.join(CERTS_DIR, "worker.key")
                if os.path.exists(ca_path) and os.path.exists(worker_crt_path):
                    context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
                    context.load_cert_chain(certfile=worker_crt_path, keyfile=worker_key_path)
                    context.load_verify_locations(cafile=ca_path)
                    context.check_hostname = False
                    context.verify_mode = ssl.CERT_REQUIRED
                    s = context.wrap_socket(s, server_hostname="host_master")
            except Exception:
                pass

            s.connect((target_ip, target_port))
            s.sendall(packet_bytes)

            ack = s.recv(1024)
            s.close()
            return len(ack) > 0
        except Exception:
            return False


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

            # Configure Curve security on server socket
            try:
                secret_key_bytes = read_zmq_key("host_zmq.key")
                public_key_bytes = read_zmq_key("host_zmq.pub")
                if secret_key_bytes and public_key_bytes:
                    sock.curve_secretkey = secret_key_bytes
                    sock.curve_publickey = public_key_bytes
                    sock.curve_server = 1
            except Exception:
                pass

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

            # Configure Curve security on client socket
            try:
                server_public_bytes = read_zmq_key("host_zmq.pub")
                client_secret_bytes = read_zmq_key("worker_zmq.key")
                client_public_bytes = read_zmq_key("worker_zmq.pub")
                if server_public_bytes and client_secret_bytes and client_public_bytes:
                    sock.curve_serverkey = server_public_bytes
                    sock.curve_secretkey = client_secret_bytes
                    sock.curve_publickey = client_public_bytes
            except Exception:
                pass

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

        # Configure gRPC mTLS credentials
        secured = False
        try:
            ca_path = os.path.join(CERTS_DIR, "ca.crt")
            host_crt_path = os.path.join(CERTS_DIR, "host.crt")
            host_key_path = os.path.join(CERTS_DIR, "host.key")
            if os.path.exists(ca_path) and os.path.exists(host_crt_path):
                with open(ca_path, "rb") as f:
                    ca_cert = f.read()
                with open(host_key_path, "rb") as f:
                    host_key = f.read()
                with open(host_crt_path, "rb") as f:
                    host_cert = f.read()

                credentials = grpc.ssl_server_credentials(
                    [(host_key, host_cert)],
                    root_certificates=ca_cert,
                    require_client_auth=True
                )
                self._server.add_secure_port(f"{self.host}:{self.port}", credentials)
                secured = True
        except Exception:
            pass

        if not secured:
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
            channel = None
            # Configure gRPC mTLS credentials
            try:
                ca_path = os.path.join(CERTS_DIR, "ca.crt")
                worker_crt_path = os.path.join(CERTS_DIR, "worker.crt")
                worker_key_path = os.path.join(CERTS_DIR, "worker.key")
                if os.path.exists(ca_path) and os.path.exists(worker_crt_path):
                    with open(ca_path, "rb") as f:
                        ca_cert = f.read()
                    with open(worker_key_path, "rb") as f:
                        worker_key = f.read()
                    with open(worker_crt_path, "rb") as f:
                        worker_cert = f.read()

                    credentials = grpc.ssl_channel_credentials(
                        root_certificates=ca_cert,
                        private_key=worker_key,
                        certificate_chain=worker_cert
                    )
                    # Override target name to match host cert SAN to prevent hostname mismatch errors
                    channel_options = [
                        ('grpc.ssl_target_name_override', 'host_master'),
                        ('grpc.max_receive_message_length', 100 * 1024 * 1024),
                        ('grpc.max_send_message_length', 100 * 1024 * 1024)
                    ]
                    channel = grpc.secure_channel(f"{target_ip}:{target_port}", credentials, options=channel_options)
            except Exception:
                pass

            if channel is None:
                channel = grpc.insecure_channel(
                    f"{target_ip}:{target_port}",
                    options=[
                        ('grpc.max_receive_message_length', 100 * 1024 * 1024),
                        ('grpc.max_send_message_length', 100 * 1024 * 1024)
                    ]
                )

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
