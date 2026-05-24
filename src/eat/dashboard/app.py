"""Dashboard entry point: `eat-dashboard` -> streamlit run home.py.

We can't call into Streamlit's machinery directly without bootstrapping its
script runner, so the entry point shells out to `streamlit run`. The actual
page lives next to this file as `home.py` plus the `pages/` folder.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from eat.config import get_settings


def main() -> int:
    s = get_settings()
    here = Path(__file__).parent
    home = here / "home.py"
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(home),
        "--server.address", s.dashboard_host,
        "--server.port", str(s.dashboard_port),
        "--browser.gatherUsageStats", "false",
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(here.parent.parent))
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    sys.exit(main())
