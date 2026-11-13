import io
import json
import struct
import os
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
    RELEASE_SESSION = "RELEASE_SESSION"
    STAGE_STATUS = "STAGE_STATUS"


def serialize_tensor(tensor: torch.Tensor) -> bytes:
    """
    Serializes a PyTorch Tensor directly into binary buffer without precision loss.
    """
    buffer = io.BytesIO()
    # Save contiguous tensor to byte stream
    torch.save(tensor.detach().contiguous().cpu(), buffer)
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
    High-Performance Binary Tensor Packet framing protocol with CRC32:
    Header: [4 bytes Magic "ORCH"][2 bytes MsgTypeCode (uint16)][4 bytes HeaderLen (uint32)][4 bytes TensorLen (uint32)][4 bytes BodyCRC32 (uint32)]
    Body:   [JSON Header UTF-8 Bytes][Binary (Optional Encrypted) Tensor Bytes]
    Total Header Size = 18 bytes fixed.
    """

    MAGIC = b"ORCH"
    HEADER_STRUCT = "!4sHIII"
    HEADER_SIZE = struct.calcsize(HEADER_STRUCT)  # 18 bytes

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

        # Edge Security: Optional AES-GCM-256 Encryption of intermediate tensor data
        if secret_key and HAS_CRYPTO and tensor_bytes:
            aesgcm = AESGCM(secret_key)
            nonce = b"\x00" * 12  # Standard 96-bit nonce for LAN streaming prototype
            tensor_bytes = aesgcm.encrypt(nonce, tensor_bytes, None)

        msg_type_code = list(MessageType).index(msg_type)
        header_len = len(header_bytes)
        tensor_len = len(tensor_bytes)

        # Compute CRC32 over the packet body
        import zlib
        body_bytes = header_bytes + tensor_bytes
        body_crc32 = zlib.crc32(body_bytes) & 0xffffffff

        packet_header = struct.pack(
            cls.HEADER_STRUCT, cls.MAGIC, msg_type_code, header_len, tensor_len, body_crc32
        )

        return packet_header + body_bytes

    @classmethod
    def unpack(
        cls, raw_bytes: bytes, secret_key: Optional[bytes] = None, device: str = "cpu"
    ) -> Tuple[MessageType, Dict[str, Any], Optional[torch.Tensor]]:
        if len(raw_bytes) < cls.HEADER_SIZE:
            raise ValueError(f"Invalid packet length: {len(raw_bytes)} < {cls.HEADER_SIZE}")

        magic, msg_type_code, header_len, tensor_len, body_crc32 = struct.unpack(
            cls.HEADER_STRUCT, raw_bytes[: cls.HEADER_SIZE]
        )
        if magic != cls.MAGIC:
            raise ValueError(f"Invalid magic header: {magic}")

        msg_type = list(MessageType)[msg_type_code]
        header_end = cls.HEADER_SIZE + header_len
        body_bytes = raw_bytes[cls.HEADER_SIZE : header_end + tensor_len]

        # Verify CRC32 Checksum
        import zlib
        computed_crc32 = zlib.crc32(body_bytes) & 0xffffffff
        if computed_crc32 != body_crc32:
            raise ValueError(f"CRC32 Checksum Mismatch: packet body corrupted in transit (expected {body_crc32}, got {computed_crc32})")

        header_bytes = raw_bytes[cls.HEADER_SIZE : header_end]
        tensor_bytes = raw_bytes[header_end : header_end + tensor_len]

        header_dict = json.loads(header_bytes.decode("utf-8"))
        meta = header_dict.get("meta", {})

        tensor = None
        if tensor_bytes:
            if secret_key and HAS_CRYPTO:
                aesgcm = AESGCM(secret_key)
                nonce = b"\x00" * 12
                tensor_bytes = aesgcm.decrypt(nonce, tensor_bytes, None)

            tensor = deserialize_tensor(tensor_bytes, device=device)
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                raise ValueError("Tensor payload integrity validation failed: contains NaN or Inf values.")

        return msg_type, meta, tensor

