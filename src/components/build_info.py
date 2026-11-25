"""Version and commit of the running orchestrator.

Baked into the image by the release workflow (Dockerfile build args CHABO_VERSION /
CHABO_COMMIT). Any other build or a local run reports "dev" / "unknown", so it never
pretends to be a release.
"""
import os
from typing import Dict


def build_info() -> Dict[str, str]:
    """{"version": ..., "commit": ...} of this orchestrator."""
    return {
        "version": os.getenv("CHABO_VERSION") or "dev",
        "commit": os.getenv("CHABO_COMMIT") or "unknown",
    }
