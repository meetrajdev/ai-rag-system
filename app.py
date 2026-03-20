"""
Compatibility shim.

If you previously started the server with `uvicorn app:app`, keep doing that.
The real application now lives in `main.py`.
"""

from main import app  # noqa: F401
