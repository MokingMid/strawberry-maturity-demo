"""WSGI 入口 —— 供 Render / gunicorn / uWSGI 使用。

本地开发也可以直接跑：
    python app.py            # 然后打开 http://127.0.0.1:8000
    gunicorn app:app         # 生产
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "demo"))

from demo.web.app import app  # noqa: E402,F401


def main() -> int:
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    print(f"草莓成熟度检测 Demo → http://{host}:{port}")
    app.run(host=host, port=port, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
