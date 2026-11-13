import os
import sys

import pytest

OC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if OC_ROOT not in sys.path:
    sys.path.insert(0, OC_ROOT)


@pytest.fixture(scope="session", autouse=True)
def _ensure_test_certificates():
    """
    Certificates are not committed. Transports fail closed without them, so generate a
    throwaway local set for the test session when the certs directory is missing.
    """
    from orchest.network.transport import CERTS_DIR

    required = ["ca.crt", "host.crt", "host.key", "worker.crt", "worker.key"]
    if not all(os.path.exists(os.path.join(CERTS_DIR, name)) for name in required):
        scripts_dir = os.path.join(OC_ROOT, "scripts")
        sys.path.insert(0, scripts_dir)
        try:
            import generate_certs

            generate_certs.main()
        finally:
            sys.path.remove(scripts_dir)
    yield
