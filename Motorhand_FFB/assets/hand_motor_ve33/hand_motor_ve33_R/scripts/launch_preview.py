"""使用当前 Python 环境启动本侧网页预览。"""
import runpy
from pathlib import Path

root = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    runpy.run_path(str(root / "scripts/viewer_server.py"), run_name="__main__")
