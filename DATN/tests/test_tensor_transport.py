import time
import pytest
import torch
from orchest.network.protocol import TensorPacket, MessageType, serialize_tensor, deserialize_tensor
from orchest.network.transport import TensorTransportServer, TensorTransportClient


def test_tensor_serialization():
    t = torch.randn(2, 4, 128, dtype=torch.float32)
    bdata = serialize_tensor(t)
    t_restored = deserialize_tensor(bdata)

    assert torch.equal(t, t_restored)


def test_tensor_transport_socket_transmission():
    received_packets = []

    def _on_recv(msg_type, meta, tensor):
        received_packets.append((msg_type, meta, tensor))

    server = TensorTransportServer(host="127.0.0.1", port=59555)
    server.on_tensor_received_cb = _on_recv
    server.start()

    time.sleep(0.2)

    sample_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    meta_info = {"prompt_id": "req_123", "layer_stage": 1}

    ok = TensorTransportClient.send_packet(
        target_ip="127.0.0.1",
        target_port=59555,
        msg_type=MessageType.FORWARD_TENSOR,
        meta=meta_info,
        tensor=sample_tensor,
    )

    time.sleep(0.5)
    server.stop()

    assert ok is True
    assert len(received_packets) == 1
    msg_type, meta, tensor = received_packets[0]
    assert msg_type == MessageType.FORWARD_TENSOR
    assert meta["prompt_id"] == "req_123"
    assert torch.equal(tensor, sample_tensor)
