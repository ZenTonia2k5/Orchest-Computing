# Walkthrough: Cryptographic Security Upgrade (mTLS, ZMQ Curve, Allowlist, CRC32)

We have successfully implemented the **Security Architecture** requirements specified in Section 5 of the project proposal. The system now enforces mutual authentication, node authorization, payload encryption, and packet integrity validation.

---

## 🛠️ Security Features Implemented

### 1. Packet Integrity Verification (CRC32 Checksum)
*   **Header Update**: Extended [`protocol.py`](file:///d:/BKU/Orchest-Computing/DATN/orchest/network/protocol.py) header layout (`HEADER_STRUCT = "!4sHIII"`, 18 bytes fixed) to append a 4-byte CRC32 body checksum.
*   **Automatic Validation**: `TensorPacket.pack` computes the CRC32 of the body in real-time, and `TensorPacket.unpack` recalculates and validates it. If any corruption or tampering is detected, a `ValueError` is raised and the payload is discarded.

### 2. mutual TLS (mTLS) & Curve Encryption
*   **Credential Generator**: Added [`generate_certs.py`](file:///d:/BKU/Orchest-Computing/DATN/scripts/generate_certs.py) to generate X.509 certificates and keys under `DATN/certs/` for the Root CA, Host, and Workers.
*   **Raw TCP mTLS**: Wrapped connections in [`transport.py`](file:///d:/BKU/Orchest-Computing/DATN/orchest/network/transport.py) with Python's standard `ssl.SSLContext`, enforcing `CERT_REQUIRED` and verifying peer certificates.
*   **gRPC mTLS**: Wrapped the gRPC server and client channels with CA trust certificates and client-side private credentials. Handled address verification dynamically using `grpc.ssl_target_name_override = 'host_master'`.
*   **ZeroMQ Curve Security**: Enabled native ZeroMQ **Curve** elliptic curve encryption by configuring server/client sockets with generated curve secret/public keys.

### 3. Node Registration Allowlist
*   **Authorization check**: Added an allowlist filter in `HostOrchestrator._on_beacon` ([`host.py`](file:///d:/BKU/Orchest-Computing/DATN/orchest/executor/host.py)). Incoming worker nodes that emit discovery beacons but are not in the allowed list are rejected from registering.
*   **CLI integration**: Exposed this check via the `--allowlist` command line parameter in [`host_node.py`](file:///d:/BKU/Orchest-Computing/DATN/apps/host_node.py).

---

## 🚀 How to Run

### 1. Generate Certificates & Keys
Run the generator script (only needed once to create the `certs/` folder):
```bash
python scripts/generate_certs.py
```

### 2. Start the Host Node with an Allowlist
Start the Host with ZMQ transport and restrict access to specific worker IDs (e.g. `worker_01` and `worker_02`):
```bash
python apps/host_node.py --transport zmq --allowlist worker_01,worker_02
```

Workers broadcasting standard UDP beacons will only be registered in the cluster topology if their Node ID matches the allowed list. Non-allowed nodes will be logged as security warnings:
```
[WARNING] HostOrchestrator: Unauthorized Worker Node registration attempt: [worker_unknown] at IP: 192.168.1.55. Node ignored.
```

---

## 🧪 Verification & Test Results

### 1. New Unit Tests
Added unit tests to [`test_fault_tolerance.py`](file:///d:/BKU/Orchest-Computing/DATN/tests/test_fault_tolerance.py) validating the allowlist authorization:
*   `test_host_orchestrator_allowlist()`

### 2. Test Execution
All 17 test cases in the test suite passed successfully:
```
tests/test_dp_partitioner.py::test_dp_partitioning_heterogeneous PASSED
tests/test_dp_partitioner.py::test_ratio_partitioning PASSED
tests/test_fault_tolerance.py::test_fault_tolerance_monitor_timeout PASSED
tests/test_fault_tolerance.py::test_host_orchestrator_failure_pruning PASSED
tests/test_fault_tolerance.py::test_host_orchestrator_allowlist PASSED
tests/test_micro_batcher.py::test_micro_batch_splitting PASSED
tests/test_micro_batcher.py::test_micro_batcher_pipeline_worker PASSED
tests/test_network_discovery.py::test_udp_discovery_service PASSED
tests/test_profiler.py::test_node_profiler_metrics PASSED
tests/test_profiler.py::test_model_profiler_layers PASSED
tests/test_swarm_inference.py::test_sliced_pipeline_forward_pass_equivalence PASSED
tests/test_tensor_transport.py::test_tensor_serialization PASSED
tests/test_tensor_transport.py::test_tensor_packet_encryption PASSED
tests/test_tensor_transport.py::test_packet_without_tensor PASSED
tests/test_tensor_transport.py::test_tensor_transport_socket_transmission PASSED
tests/test_tensor_transport.py::test_tensor_transport_zmq_transmission PASSED
tests/test_tensor_transport.py::test_tensor_transport_grpc_transmission PASSED

============================= 17 passed in 15.95s =============================
```
