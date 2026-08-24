import time
import os
import pytest
import torch
from orchest.network.protocol import TensorPacket, MessageType, serialize_tensor, deserialize_tensor
from orchest.network.transport import TensorTransportServer, TensorTransportClient


def test_tensor_serialization():
    t = torch.randn(2, 4, 128, dtype=torch.float32)
    bdata = serialize_tensor(t)
    t_restored = deserialize_tensor(bdata)

    assert torch.equal(t, t_restored)


def test_tensor_packet_encryption():
    key = os.urandom(32)  # 256-bit AES key
    sample_tensor = torch.randn(4, 64, dtype=torch.float32)
    meta = {"layer": 5, "model": "Qwen2.5-0.5B"}

    packed = TensorPacket.pack(
        msg_type=MessageType.FORWARD_TENSOR,
        meta=meta,
        tensor=sample_tensor,
        secret_key=key,
    )
    
    msg_type, meta_out, tensor_out = TensorPacket.unpack(packed, secret_key=key)
    assert msg_type == MessageType.FORWARD_TENSOR
    assert meta_out["layer"] == 5
    assert torch.equal(sample_tensor, tensor_out)


def test_packet_without_tensor():
    meta = {"action": "heartbeat", "status": "READY"}
    packed = TensorPacket.pack(
        msg_type=MessageType.HEARTBEAT,
        meta=meta,
        tensor=None,
    )
    msg_type, meta_out, tensor_out = TensorPacket.unpack(packed)
    assert msg_type == MessageType.HEARTBEAT
    assert meta_out["action"] == "heartbeat"
    assert tensor_out is None


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

    # 1. Send packet WITH tensor
    ok1 = TensorTransportClient.send_packet(
        target_ip="127.0.0.1",
        target_port=59555,
        msg_type=MessageType.FORWARD_TENSOR,
        meta=meta_info,
        tensor=sample_tensor,
    )

    # 2. Send packet WITHOUT tensor (control signal, e.g. ACK or DISPATCH)
    ok2 = TensorTransportClient.send_packet(
        target_ip="127.0.0.1",
        target_port=59555,
        msg_type=MessageType.DISPATCH_SLICE,
        meta={"model": "test"},
        tensor=None,
    )

    time.sleep(0.5)
    server.stop()

    assert ok1 is True
    assert ok2 is True
    assert len(received_packets) == 2

    # Verify first packet
    msg_type1, meta1, tensor1 = received_packets[0]
    assert msg_type1 == MessageType.FORWARD_TENSOR
    assert meta1["prompt_id"] == "req_123"
    assert torch.equal(tensor1, sample_tensor)

    # Verify second packet
    msg_type2, meta2, tensor2 = received_packets[1]
    assert msg_type2 == MessageType.DISPATCH_SLICE
    assert tensor2 is None


def test_tensor_transport_zmq_transmission():
    received_packets = []

    def _on_recv(msg_type, meta, tensor):
        received_packets.append((msg_type, meta, tensor))

    server = TensorTransportServer(host="127.0.0.1", port=59556, transport_type="zmq")
    server.on_tensor_received_cb = _on_recv
    server.start()

    time.sleep(0.2)

    sample_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    meta_info = {"prompt_id": "req_zmq", "layer_stage": 1}

    ok1 = TensorTransportClient.send_packet(
        target_ip="127.0.0.1",
        target_port=59556,
        msg_type=MessageType.FORWARD_TENSOR,
        meta=meta_info,
        tensor=sample_tensor,
        transport_type="zmq",
    )

    time.sleep(0.5)
    server.stop()

    assert ok1 is True
    assert len(received_packets) == 1

    msg_type1, meta1, tensor1 = received_packets[0]
    assert msg_type1 == MessageType.FORWARD_TENSOR
    assert meta1["prompt_id"] == "req_zmq"
    assert torch.equal(tensor1, sample_tensor)


def test_tensor_transport_grpc_transmission():
    import pytest
    pytest.importorskip("grpc")
    received_packets = []

    def _on_recv(msg_type, meta, tensor):
        received_packets.append((msg_type, meta, tensor))

    server = TensorTransportServer(host="127.0.0.1", port=59557, transport_type="grpc")
    server.on_tensor_received_cb = _on_recv
    server.start()

    time.sleep(0.5)

    sample_tensor = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32)
    meta_info = {"prompt_id": "req_grpc", "layer_stage": 2}

    ok1 = TensorTransportClient.send_packet(
        target_ip="127.0.0.1",
        target_port=59557,
        msg_type=MessageType.FORWARD_TENSOR,
        meta=meta_info,
        tensor=sample_tensor,
        transport_type="grpc",
    )

    time.sleep(0.5)
    server.stop()

    assert ok1 is True
    assert len(received_packets) == 1

    msg_type1, meta1, tensor1 = received_packets[0]
    assert msg_type1 == MessageType.FORWARD_TENSOR
    assert meta1["prompt_id"] == "req_grpc"
    assert torch.equal(tensor1, sample_tensor)


