import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests")
    exit_code = pytest.main(["-v", test_dir])
    sys.exit(exit_code)


