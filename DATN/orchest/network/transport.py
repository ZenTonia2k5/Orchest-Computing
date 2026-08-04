import socket
import struct
import threading
import time
from typing import Callable, Optional, Dict, Any, Tuple

import torch
from .protocol import TensorPacket, MessageType


class TensorTransportServer:
    """
    High-Performance TCP Socket Server for receiving intermediate binary activation tensors.
    """

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

    def start(self):
        self._running = True
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind((self.host, self.port))
        self._server_sock.listen(128)

        def _accept_loop():
            while self._running:
                try:
                    conn, addr = self._server_sock.accept()
                    client_thread = threading.Thread(
                        target=self._handle_client, args=(conn, addr), daemon=True
                    )
                    client_thread.start()
                except Exception:
                    break

        self._thread = threading.Thread(target=_accept_loop, daemon=True)
        self._thread.start()

    def _recv_exact(self, conn: socket.socket, num_bytes: int) -> bytes:
        data = b""
        while len(data) < num_bytes:
            chunk = conn.recv(min(num_bytes - len(data), 65536))
            if not chunk:
                break
            data += chunk
        return data

    def _handle_client(self, conn: socket.socket, addr: Tuple[str, int]):
        try:
            # Read 8 byte header
            header_head = self._recv_exact(conn, 8)
            if len(header_head) < 8:
                conn.close()
                return

            magic, msg_type_code, header_len = struct.unpack("!4sHH", header_head)
            if magic != TensorPacket.MAGIC:
                conn.close()
                return

            # Read full packet body
            raw_body = self._recv_exact(conn, header_len)
            # Peek if there is raw tensor bytes coming
            # Read 4-byte payload length prefix if stream mode
            tensor_data_len_head = conn.recv(4)
            if len(tensor_data_len_head) == 4:
                (tensor_len,) = struct.unpack("!I", tensor_data_len_head)
                raw_tensor = self._recv_exact(conn, tensor_len)
            else:
                raw_tensor = b""

            raw_full = header_head + raw_body + raw_tensor
            msg_type, meta, tensor = TensorPacket.unpack(raw_full, secret_key=self.secret_key)

            if self.on_tensor_received_cb:
                self.on_tensor_received_cb(msg_type, meta, tensor)

            # Send ACK
            ack_pkt = TensorPacket.pack(MessageType.ACK, {"status": "OK"})
            conn.sendall(ack_pkt)
        except Exception:
            pass
        finally:
            conn.close()

    def stop(self):
        self._running = False
        if self._server_sock:
            self._server_sock.close()


class TensorTransportClient:
    """
    High-Performance Socket Client for forwarding activation tensors to next pipeline node.
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
    ) -> bool:
        packet_bytes = TensorPacket.pack(msg_type, meta, tensor, secret_key=secret_key)
        
        # Split packet into header frame and tensor payload frame for socket streaming
        header_len = struct.unpack("!H", packet_bytes[6:8])[0]
        header_part = packet_bytes[: 8 + header_len]
        tensor_part = packet_bytes[8 + header_len :]

        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(timeout)
            s.connect((target_ip, target_port))

            # Send header part
            s.sendall(header_part)
            # Send tensor payload length + tensor bytes
            if tensor_part:
                s.sendall(struct.pack("!I", len(tensor_part)))
                s.sendall(tensor_part)

            # Wait for ACK
            ack = s.recv(1024)
            s.close()
            return len(ack) > 0
        except Exception as e:
            return False
