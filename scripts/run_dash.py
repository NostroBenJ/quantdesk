"""Launch the research dashboard from any working directory.

    python scripts/run_dash.py

Exists because the launch config lives at the Downloads root while the package
lives in quantdesk/, so `python -m quantdesk.dash.app` only resolves when the
cwd happens to be right. This puts the package on sys.path explicitly.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The dashboard resolves runs/ and data/ relative to the package root, so make
# the cwd deterministic rather than inheriting whatever launched us.
import os  # noqa: E402

os.chdir(ROOT)

from quantdesk.dash.app import main  # noqa: E402

if __name__ == "__main__":
    # Binds to loopback by default and that default should survive going to a
    # server. The dashboard has no authentication of any kind - it is a research
    # tool, not a product - so it must never listen on a public interface.
    #
    # For remote access use Tailscale (free, private, works from a phone) and
    # bind to the machine's tailnet address, or an SSH tunnel:
    #     ssh -L 8010:127.0.0.1:8010 user@host
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8010
    host = sys.argv[2] if len(sys.argv) > 2 else "127.0.0.1"
    if host not in ("127.0.0.1", "localhost", "::1") and not host.startswith("100."):
        print(
            f"refusing to bind {host}: the dashboard has no auth. Use a tailnet "
            "address (100.x.y.z) or an SSH tunnel.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    main(host=host, port=port)
