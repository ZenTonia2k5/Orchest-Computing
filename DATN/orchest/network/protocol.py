import io
import json
import struct
from enum import Enum
from typing import Dict, Any, Tuple, Optional

import torch

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False


class MessageType(str, Enum):
    BEACON = "BEACON"
    FORWARD_TENSOR = "FORWARD_TENSOR"
    ACK = "ACK"
    HEARTBEAT = "HEARTBEAT"
    DISPATCH_SLICE = "DISPATCH_SLICE"
    INFERENCE_COMPLETE = "INFERENCE_COMPLETE"


def serialize_tensor(tensor: torch.Tensor) -> bytes:
    """
    Serializes a PyTorch Tensor directly into binary buffer without precision loss.
    """
    buffer = io.BytesIO()
    # Save tensor to byte stream
    torch.save(tensor.detach().cpu(), buffer)
    return buffer.getvalue()


def deserialize_tensor(bdata: bytes, device: str = "cpu") -> torch.Tensor:
    """
    Deserializes raw binary buffer back into PyTorch Tensor on target device.
    """
    buffer = io.BytesIO(bdata)
    tensor = torch.load(buffer, map_location=device, weights_only=True)
    return tensor


class TensorPacket:
    """
    High-Performance Binary Tensor Packet framing protocol:
    Header: [4 bytes Magic "ORCH"][2 bytes MsgType][4 bytes HeaderLen][JSON Header Bytes][Binary Tensor Bytes]
    """

    MAGIC = b"ORCH"

    @classmethod
    def pack(
        cls,
        msg_type: MessageType,
        meta: Dict[str, Any],
        tensor: Optional[torch.Tensor] = None,
        secret_key: Optional[bytes] = None,
    ) -> bytes:
        header_dict = {"msg_type": msg_type.value, "meta": meta}
        header_bytes = json.dumps(header_dict).encode("utf-8")

        tensor_bytes = serialize_tensor(tensor) if tensor is not None else b""

        # Edge Security: Optional Encryption of intermediate tensor data
        if secret_key and HAS_CRYPTO and tensor_bytes:
            aesgcm = AESGCM(secret_key)
            nonce = b"\x00" * 12  # Simple fixed nonce for LAN streaming prototype
            tensor_bytes = aesgcm.encrypt(nonce, tensor_bytes, None)

        msg_type_code = list(MessageType).index(msg_type)

        header_len = len(header_bytes)
        packet_header = struct.pack("!4sHH", cls.MAGIC, msg_type_code, header_len)

        return packet_header + header_bytes + tensor_bytes

    @classmethod
    def unpack(
        cls, raw_bytes: bytes, secret_key: Optional[bytes] = None, device: str = "cpu"
    ) -> Tuple[MessageType, Dict[str, Any], Optional[torch.Tensor]]:
        if len(raw_bytes) < 8:
            raise ValueError("Invalid packet length")

        magic, msg_type_code, header_len = struct.unpack("!4sHH", raw_bytes[:8])
        if magic != cls.MAGIC:
            raise ValueError(f"Invalid magic header: {magic}")

        msg_type = list(MessageType)[msg_type_code]
        header_bytes = raw_bytes[8 : 8 + header_len]
        tensor_bytes = raw_bytes[8 + header_len :]

        header_dict = json.loads(header_bytes.decode("utf-8"))
        meta = header_dict.get("meta", {})

        tensor = None
        if tensor_bytes:
            if secret_key and HAS_CRYPTO:
                aesgcm = AESGCM(secret_key)
                nonce = b"\x00" * 12
                tensor_bytes = aesgcm.decrypt(nonce, tensor_bytes, None)

            tensor = deserialize_tensor(tensor_bytes, device=device)

        return msg_type, meta, tensor
